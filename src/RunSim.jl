using Printf

export run_sim

# ============================================================================ #
"""
run_sim(d::LiveDemo, amp::Real=1.0, dur::Real=4.0, nrep::Integer=10)
"""
function run_sim(d::LiveDemo, amp::Real=1.0, dur::Real=4.0, nrep::Integer=10)
    return run_sim(d.net, amp, dur, nrep)
end
# ---------------------------------------------------------------------------- #
"""
run_sim(net::LIFNetwork, amp::Real, dur::Real, nrep::Integer, tstart::Time=0.0,
    record_vm::Bool=false)
"""
function run_sim(net::LIFNetwork, amp::Real, dur::Real,
    nrep::Integer, tstart::Time=0.0, record_vm::Bool=false)

    #dur should be in seconds
    if dur > 100.0
        error("Simulation duration is really long... check your units!")
    end

    ncell = length(net)

    evt_ts = tstart:dur:tstart + dur*(nrep-1)

    fstim(id::Integer, t::Float64) = begin
        if id == 1
            return amp*((0.5*sin(2.0*pi*(t*1e-3)*4.0))+0.5)
        else
            return 0.0
        end
    end

    ts = [Vector{Float64}() for x in 1:ncell]

    vm = [zeros(1) for x in 1:ncell]
    if record_vm
        rec = collect(1:ncell)
    else
        rec = Int[]
    end

    @inbounds for kr = 1:nrep

        d = simulate(net, fstim, dur*1e3, rec)

        @inbounds for k = 1:ncell
            append!(ts[k], d[:ts][k] .+ evt_ts[kr])
            if record_vm
                append!(vm[k], d[:vm][:,k])
            end
        end
    end

    return ts, vm, evt_ts
end
# =========================================================================== #
