
module NeuroBootcamp

using PyCall
const animation = PyNULL()
function __init__()
    # silence the old "QApplication: invalid style override passed, ignoring
    #  it." warning...
    Base.eval(Main, :(ENV["QT_STYLE_OVERRIDE"] = ""))
    copy!(animation, pyimport("matplotlib.animation"))
end

using PyPlot

include("./LifConfig.jl")
using .LifConfig

include("./Lif.jl")
using .Lif

include("./Networks.jl")
using .Networks

include("./Histogram/Histogram.jl")

include("./Spk/Spk.jl")
using .Spk

include("./SpkPlot/SpkPlot.jl")
using .SpkPlot

import Base.run

export build_demo, build_network, SquareWave, SineWave, WhiteNoise, run_sim
export plot_raster, plot_raster!, plot_cycle_mean, plot_cycle_mean!, plot_xcorr,
    psth, get_xcorr, f1xfm, cycle_mean

abstract type Stimulus end
# ============================================================================ #
mutable struct ChunkArray
    nel::Int
    k::Int
    last::Float64
    d::Vector{Float64}
    # ------------------------------------------------------------------------ #
    function ChunkArray(n::Int)
        self = new()
        self.nel = n
        self.k = 1
        self.d = fill(NaN, n)
        self.last = NaN
        return self
    end
    # ------------------------------------------------------------------------ #
end
# ---------------------------------------------------------------------------- #
function add!(x::ChunkArray, v::Float64)
    x.last = v;
    if x.k < x.nel
        x.d[x.k] = v
        x.k += 1
    else
        x.d[1:end-1] = x.d[2:end]
        x.d[end] = v
    end
end
# ============================================================================ #
mutable struct LiveDemo
    net::LIFNetwork
    speed::Int
    fig::PyPlot.Figure
    ax::PyCall.PyObject
end
"""
    LiveDemo(net::Network, stim::Stimulus, speed::Int=5)
"""
function LiveDemo(net::LIFNetwork, speed::Int=5)
    h = default_figure()
    demo = LiveDemo(net, speed, h, h.axes[1])
    close(h)
    return demo
end
# ---------------------------------------------------------------------------- #
function reset!(demo::LiveDemo)
    # if we always close the figure we avoid the "not responding to keypresses"
    # issue *AND* we get focus back on the figure when run() exits...
    close(demo.fig)
    demo.fig = default_figure()
    demo.ax = demo.fig.axes[1]
    Networks.reset!(demo.net)
end
# ============================================================================ #
"""
    build_network([
            (1=>3, .8),
            (2=>3, .6)
        ], xi=0.0)
"""
function build_network(inp::Vector{Tuple{Pair{T,T}, F}}, xi::Real=0.0) where {T<:Integer, F<:Real}
    ncell = 0
    for x in inp
        ncell = max(ncell, maximum(x[1]))
    end

    net = LIFNetwork(ncell, xi)

    for x in net
        Lif.set!(x, :threshold, 1.0)
        Lif.set!(x, :rm, 1.0)
        Lif.set!(x, :tau, 4.5)
        Lif.set!(x, :vspike, 4.0)
    end

    connect!(net)
    for x in inp
        connect!(net, Tuple(x[1]), (Synapses.Static, x[2], 2.0, 2.0))
    end

    return net
end
# ============================================================================ #
"""
    build_demo([
            (1=>3, .8),
            (2=>3, .6)
        ], xi=0.0)
"""
function build_demo(inp::Vector{Tuple{Pair{T,T}, F}}, xi::Real=0.0) where {T<:Integer, F<:Real}
    return LiveDemo(build_network(inp, xi), 5)
end
"""
    build_demo([1=>3, 2=>3], xi=0.0)
"""
function build_demo(pairs::Vector{Pair{T,T}}, xi::Real=0.0) where {T<:Integer}
    return build_demo([(x, 1.8) for x in pairs], xi)
end
"""
    build_demo(2, xi=0.0)
"""
function build_demo(ncell::Integer, xi::Real=0.0)
    return build_demo([(x=>ncell, 1.8) for x in 1:(ncell-1)], xi)
end
# ============================================================================ #
mutable struct SquareWave <: Stimulus
    amp::Vector{Float64}
    on::Float64
    off::Float64
    tlast::Float64
    state::Bool
end
"""
    SquareWave(demo::LiveDemo, amp::Float64=1.0, on::Float64=20.0,
        off::Float64=30.0)
"""
function SquareWave(demo::LiveDemo, amp::Float64=1.0, on::Float64=20.0,
    off::Float64=30.0)

    return SquareWave([amp; zeros(Float64, length(demo.net)-1)], 20.0, 20.0,
        0.0, false)
end
function reset!(sq::SquareWave)
    sq.state = false
    sq.tlast = 0.0
end
function getstim(sq::SquareWave, id::Integer, t::Time)
    tnext = sq.tlast + (sq.state ? sq.on : sq.off)
    if t >= tnext
        sq.tlast = t
        sq.state = !sq.state
    end
    if sq.state && (0 < id <= length(sq.amp))
        return sq.amp[id]
    else
        return 0.0
    end
end
# ============================================================================ #
mutable struct SineWave <: Stimulus
    amp::Vector{Float64}
    freq::Vector{Float64}
end
"""
    SineWave(demo::LiveDemo, amp::Float64=1.0, freq::Float64=25.0)
"""
function SineWave(demo::LiveDemo, amp::Float64=1.0, freq::Float64=25.0)
    # frequency needs to be converted to cycles-per-ms for simulations
    return SineWave([amp; zeros(Float64, length(demo.net)-1)],
        fill(freq/1000.0, length(demo.net)))
end
reset!(sw::SineWave) = nothing
function getstim(sw::SineWave, id::Integer, t::Time)
    if 0 < id <= length(sw.amp)
        return sw.amp[id] * ((0.5 * sin(2.0*pi*sw.freq[id]*t)) + 0.5)
    end
    return 0.0
end
# ============================================================================ #
mutable struct WhiteNoise <: Stimulus
    last_frame::Float64
    last_value::Float64
    ifi::Float64
    mu::Float64
    sigma::Float64
    # history::Vector{Float64}
end
"""
    WhiteNoise(mu::Real=0.7, sigma::Real=0.3, ifi::Real=4.0)
"""
function WhiteNoise(mu::Real=0.7, sigma::Real=0.3, ifi::Real=4.0)
    return WhiteNoise(-Inf, sigma * randn() + mu, ifi, mu, sigma)#, Float64[])
end
function reset!(wn::WhiteNoise)
    wn.last_frame = -Inf
end
function getstim(wn::WhiteNoise, id::Integer, t::Time)
    if id == 1
        if t >= wn.last_frame + wn.ifi
            wn.last_frame = t
            wn.last_value = wn.sigma * randn() + wn.mu
            # push!(wn.history, wn.last_value)
        end
        return wn.last_value
    else
        return 0.0
    end
end
# ============================================================================ #
mutable struct Keyboard <: Stimulus
    state::Vector{Bool}
    amp::Vector{Float64}
    inc::Float64
end
# ---------------------------------------------------------------------------- #
Keyboard(demo::LiveDemo) = Keyboard(length(demo.net))
Keyboard(n::Integer) = Keyboard(falses(n), ones(Float64, n), 0.1)
Keyboard(a::AbstractArray{T,1}) where {T<:Real} = Keyboard(falses(a), a, 0.1)
# ============================================================================ #
function keypressed(s::Keyboard, key::String)
    if key in ["1", "2", "3"]
        id = parse(Int8, key)
        if 0 < id <= length(s.state)
            s.state[id] = !s.state[id]
        end
    elseif key == "down"
        s.amp .-= s.inc
    elseif key == "up"
        s.amp .+= s.inc
    end
end
# ============================================================================ #
function reset!(s::Keyboard)
    for k in eachindex(s.state)
        s.state[k] = false
    end
end
# ============================================================================ #
function getstim(s::Keyboard, id::Integer, t::Time)
    if 0 < id <= length(s.state)
        return s.state[id] ? s.amp[id] : 0.0
    else
        return 0.0
    end
end
# ============================================================================ #
keypressed(s::Stimulus, key::String) = nothing
# ============================================================================ #
"""
    run(demo::LiveDemo, duration::Real=+Inf)
    run(demo::LiveDemo, stim::Stimulus, duration::Real=+Inf)
"""
function run(demo::LiveDemo, duration::Real=+Inf)
    return run(demo, Keyboard(length(demo.net)), duration)
end
# ---------------------------------------------------------------------------- #
function run(demo::LiveDemo, stimgen::Stimulus, duration::Real=+Inf)

    TMAX = 50.0 # max time of x axis in MS

    dt = 0.05 # timestep in MS

    npt = round(Int64, TMAX / dt)

    ncell = length(demo.net)

    if isinf(duration)
        ncall = round(Int64, 200.0 / (dt * demo.speed))
        repeat = true
    else
        ncall = round(Int64, duration / (dt * demo.speed))
        repeat = false
    end

    TNOW = 0.0

    time = ChunkArray(npt)

    data = ChunkArray[ChunkArray(npt) for k in 1:ncell]
    stim = ChunkArray[ChunkArray(npt) for k in 1:ncell]

    reset!(demo)
    reset!(stimgen)

    sca(demo.ax)
    xlim(0, TMAX)
    ylim(-2.0, 4.0)

    local col::Vector{String}
    if ncell == 2
        col = ["blue", "red"]
    elseif ncell == 3
        col = ["blue", "cyan", "red"]
    elseif ncell == 4
        col = ["blue", "cyan", "lightblue", "red"]
    else
        error("Not enough colors!")
    end

    data_line = Vector{PyCall.PyObject}(undef, ncell)
    stim_line = Vector{PyCall.PyObject}(undef, ncell)
    for k = 1:ncell
        data_line[k] = plot([], [], color=col[k], linewidth=3)[1]
        stim_line[k] = plot([], [], color=col[k], linewidth=3)[1]
    end

    ts = [Vector{Float64}() for x in 1:ncell]

    fstim(id::Integer, t::Time) = getstim(stimgen, id, t)

    # ------------------------------------------------------------------------ #
    function run_demo(k)
        for kt = 1:demo.speed
            add!(time, TNOW)
            vm, spk = Networks.update!(demo.net, fstim, dt, TNOW)
            for kc = 1:ncell
                offset = .2 * (kc - 1)
                add!(data[kc], vm[kc] - offset)
                add!(stim[kc], fstim(kc, TNOW) - (1.8 - offset))
                if spk[kc]
                    push!(ts[kc], TNOW * 0.001)
                end
            end
            TNOW += dt
        end

        for kc = 1:ncell
            data_line[kc].set_data(time.d, data[kc].d)
            stim_line[kc].set_data(time.d, stim[kc].d)
        end

        if TNOW > TMAX
            demo.ax.set_xlim(time.d[1], time.last)
        end

        demo.fig.canvas.draw()
    end
    # ------------------------------------------------------------------------ #

    anim = animation.FuncAnimation(demo.fig, run_demo, frames=ncall,
        interval=10, repeat=repeat, blit=false)

    # ------------------------------------------------------------------------ #
    function key_press(event)
        if event.key == " "
            anim.event_source.stop()
        elseif event.key == "enter"
            anim.event_source.start()
        else
            keypressed(stimgen, event.key)
        end
    end
    # ------------------------------------------------------------------------ #

    if repeat
        plt.connect("key_press_event", key_press)
    end

    return ts
end
# ============================================================================ #

include("./RunSim.jl")

end # end module
