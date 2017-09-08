
module LiveDemo

using Networks, PyCall
using Lif: set!

@pyimport matplotlib.animation as animation

using PyPlot, Plot

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
mutable struct StimGen
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
mutable struct StimState
    state::Vector{Bool}
    amp::Float64
    inc::Float64
end
StimState(n::Integer) = StimState(falses(n), 1.0, 0.1)
function setstate!(s::StimState, id::Integer)
    if 0 < id <= length(s.state)
        s.state[id] = !s.state[id]
    end
end
function getstim(s::StimState, id::Integer)
    if 0 < id <= length(s.state)
        return s.state[id] ? s.amp : 0.0
    else
        return 0.0
    end
end
# ============================================================================ #
function demo(dur::Float64, xi::Float64, amp::Float64, wi::Float64, speed::Int,
    interval::Vector{Float64}, fig=nothing, ncell=2)

    TMAX = 50.0 # max time of x axis in MS

    dt = 0.05 # timestep in MS

    npt = round(Int64, TMAX / dt)

    KN = speed

    ncall = round(Int64, dur / (dt * KN))

    TNOW = 0.0

    time = ChunkArray(npt)

    data = ChunkArray[ChunkArray(npt) for k in 1:ncell]
    stim = ChunkArray[ChunkArray(npt) for k in 1:ncell]

    fig = default_figure(fig)
    ax = fig[:axes][1]

    xlim(0, TMAX)
    ylim(-2.0, 4.0)

    local col::Vector{AbstractString}
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

    net = LIFNetwork(ncell, xi)

    for x in net
        set!(x, :threshold, 1.8)
        set!(x, :rm, 3.5)
        set!(x, :tau, 4.5)
        set!(x, :vspike, 4.0)
    end

    connect!(net)
    for k = 1:ncell-1
        connect!(net, (k, ncell), (Synapses.Static, wi, 3.0, 2.0))
    end

    # sg = StimGen(interval, amp, collect(1:ncell-1))
    # if ncell < 3
    #     fstim(id, t) = generate!(sg, id, t)
    # else
    #     sg2 = StimGen([interval[1]+5.0, interval[2]], amp, collect(1:ncell-1))
    #     fstim(id, t) = begin
    #         if id == 1
    #             ij = generate!(sg, id, t)
    #         elseif (id == 2) && (t >= 60.0)
    #             ij = generate!(sg2, id, t)
    #         else
    #             ij = 0.0
    #         end
    #     end
    # end

    sg = StimState(ncell)

    fstim(id, t) = getstim(sg, id)

    ts = [Vector{Float64}() for x in 1:ncell]

    # ------------------------------------------------------------------------ #
    function run_demo(k)

        for kt = 1:KN
            add!(time, TNOW)
            vm, spk = Networks.update!(net, fstim, dt, TNOW)
            for kc = 1:ncell
                add!(data[kc], vm[kc] - (.2 * (kc - 1)))
                add!(stim[kc], fstim(kc, TNOW) - 1.8)
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
            ax[:set_xlim](time.d[1], time.last)
        end

        fig[:canvas][:draw]()

    end
    # ------------------------------------------------------------------------ #
    stim_on(event) = setstate!(sg, event[:button])
    # ------------------------------------------------------------------------ #

    plt[:connect]("button_press_event", stim_on)

    anim = animation.FuncAnimation(fig, run_demo, frames=ncall, interval=10, repeat=true, blit=false)

    # ------------------------------------------------------------------------ #
    function end_demo(event)
        if event[:key] == "escape"
            anim[:event_source][:stop]()
        elseif event[:key] == "up"
            sg.amp *= (1 + sg.inc)
        elseif event[:key] == "down"
            sg.amp *= (1 - sg.inc)
        end
    end
    # ------------------------------------------------------------------------ #

    plt[:connect]("key_press_event", end_demo)

    return fig, ts

end
# ============================================================================ #
function run(n = 1, fig = nothing)
    dur = 95.0
    xi = 0.0
    amp = 0.5
    wi = 0.0
    speed = 3
    interval = [20.0, 20.0]
    ncell = 2

    if n == 2
        speed = 5
        amp = 0.7
        interval = [10.0, 6.0]
        dur = 55.0

    elseif n == 3
        speed = 5
        amp = 0.7
        interval = [20.0, 6.0]
        wi = 1.8

    elseif n == 4
        speed = 5
        amp = 0.7
        interval = [20.0, 6.0]
        wi = 1.8
        # xi = 0.05
        dur = 200.0
        ncell = 3

    elseif n == 5
        speed = 5
        amp = 0.8
        interval = [20.0, 6.0]
        wi = 1.8
        xi = 0.05
        dur = 200.0
        ncell = 3

    end

    fig = demo(dur, xi, amp, wi, speed, interval, fig, ncell)

    # print("Press [ENTER] to begin >> ")
    # readline(STDIN)

    plt[:show]()

    return fig
end
# ============================================================================ #

end # end module
