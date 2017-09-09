
module LiveDemo

using Networks, PyCall, LifConfig
using Lif: set!

@pyimport matplotlib.animation as animation

using PyPlot, Plot

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
mutable struct StimGen <: Stimulus
    output::Float64
    interval::Vector{Float64}
    amp::Float64
    tlast::Float64
    id::Vector{Int}
    state::Int

    function StimGen(interval::Vector{Float64}, amp::Float64, id::Vector{Int}=[1])
        self = new()
        self.output = 0.0
        self.amp = amp
        self.interval = interval
        self.tlast = 0.0
        self.id = id
        self.state = 1

        return self
    end
end
# ---------------------------------------------------------------------------- #
function generate!(s::StimGen, id::Int64, t::Float64)

    out::Float64

    if id in s.id

        if (t >= (s.tlast + s.interval[s.state]))
            s.tlast = t
            s.output = abs(s.output - s.amp)
            s.state = abs(s.state - 3)
        end

        out = s.output

    else
        out = 0.0
    end

    return out
end
# ---------------------------------------------------------------------------- #
function generate2!(s::StimGen, id::Int64, t::Float64, f::Float64,
    phase::Float64)
    out::Float64

    if id in s.id
        out = 0.5*(sin(2.0*pi*(t*1e-3)*f+phase))+0.5
    else
        out = 0.0
    end

    return float(out > 0.5)
end
# ============================================================================ #
mutable struct Demo
    net::LIFNetwork
    speed::Int
    fig::PyPlot.Figure
    ax::PyCall.PyObject
end
"""
    Demo(net::Network, stim::Stimulus, speed::Int=5)
"""
function Demo(net::LIFNetwork, speed::Int=5)
    h = default_figure()
    return Demo(net, speed, h, h[:axes][1])
end
# ---------------------------------------------------------------------------- #
function reset!(demo::Demo)
    # if we always close the figure we avoid the "not responding to keypresses"
    # issue *AND* we get focus back on the figure when run() exits...
    close(demo.fig)
    demo.fig = default_figure()
    demo.ax = demo.fig[:axes][1]
    Networks.reset!(demo.net)
end
# ============================================================================ #
"""
    build_demo([
            (1=>3, .8),
            (2=>3, .6)
        ])
"""
function build_demo{T<:Integer, F<:Real}(inp::Vector{Tuple{Pair{T,T}, F}})
    ncell = 0
    for x in inp
        ncell = max(ncell, maximum(x[1]))
    end

    net = LIFNetwork(ncell, 0.0)

    for x in net
        set!(x, :threshold, 1.8)
        set!(x, :rm, 3.5)
        set!(x, :tau, 4.5)
        set!(x, :vspike, 4.0)
    end

    connect!(net)
    for x in inp
        connect!(net, Tuple(x[1]), (Synapses.Static, x[2], 2.0, 2.0))
    end
    return Demo(net, 5)
end
"""
    build_demo([1=>3, 2=>3])
"""
function build_demo{T<:Integer}(pairs::Vector{Pair{T,T}})
    return build_demo([(x, 1.8) for x in pairs])
end
"""
    build_demo(2)
"""
function build_demo(ncell::Integer)
    return build_demo([(x=>ncell, 1.8) for x in 1:(ncell-1)])
end
# ============================================================================ #
mutable struct StimState <: Stimulus
    state::Vector{Bool}
    amp::Vector{Float64}
    inc::Float64
end
# ---------------------------------------------------------------------------- #
StimState(demo::Demo) = StimState(length(demo.net))
StimState(n::Integer) = StimState(falses(n), ones(Float64, n), 0.1)
StimState{T<:Real}(a::AbstractArray{T,1}) = StimState(falses(a), a, 0.1)
# ============================================================================ #
function keypressed(s::StimState, key::String)
    if key in ["1", "2", "3"]
        id = parse(Int8, key)
        if 0 < id <= length(s.state)
            s.state[id] = !s.state[id]
        end
    end
end
# ============================================================================ #
function getstim(s::StimState, id::Integer, t::Time)
    if 0 < id <= length(s.state)
        return s.state[id] ? s.amp[id] : 0.0
    else
        return 0.0
    end
end
# ============================================================================ #
function run(demo::Demo, duration::Real=+Inf)
    return run(demo, StimState(length(demo.net)), duration)
end
# ---------------------------------------------------------------------------- #
function run(demo::Demo, stimgen::Stimulus, duration::Real=+Inf)

    const TMAX = 50.0 # max time of x axis in MS

    const dt = 0.05 # timestep in MS

    const npt = round(Int64, TMAX / dt)

    const ncell = length(demo.net)

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

    data_line = Vector{PyCall.PyObject}(ncell)
    stim_line = Vector{PyCall.PyObject}(ncell)
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
            data_line[kc][:set_data](time.d, data[kc].d)
            stim_line[kc][:set_data](time.d, stim[kc].d)
        end

        if TNOW > TMAX
            demo.ax[:set_xlim](time.d[1], time.last)
        end

        demo.fig[:canvas][:draw]()
    end
    # ------------------------------------------------------------------------ #

    anim = animation.FuncAnimation(demo.fig, run_demo, frames=ncall, interval=10, repeat=repeat, blit=false)

    # ------------------------------------------------------------------------ #
    function key_press(event)
        if event[:key] == " "
            anim[:event_source][:stop]()
        elseif event[:key] == "enter"
            anim[:event_source][:start]()
        else
            keypressed(stimgen, event[:key])
        end
    end
    # ------------------------------------------------------------------------ #

    if repeat
        plt[:connect]("key_press_event", key_press)
    end

    return ts
end
# ============================================================================ #

end # end module
