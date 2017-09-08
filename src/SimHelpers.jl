module SimHelpers

using LifConfig, Networks

export sine_stim, naka_rushton, run_sim_par, read_sim_file

const default_dir = joinpath(splitdir(@__FILE__)[1], "data")

# ============================================================================ #
function sine_stim(id::Integer, t::Time, set::Vector{Int}=[], amp::Float64=.05, f::Float64=4.0)
    if id in set
        phase = -pi/2.0
        ij = amp*((0.5*sin(2.0*pi*(t*1e-3)*f+phase))+0.5)
    else
        ij = 0.0
    end
    return ij
end
# ============================================================================ #
function naka_rushton(x::Vector{Float64}, p::Vector)
    return p[1] * (x.^p[2] ./ (x.^p[2] + p[3].^p[2])) + p[4]
end
# ============================================================================ #
function progress(msg)
    print("\u1b[1G")   # go to first column
    print_with_color(:yellow, msg)
    print("\u1b[K")    # clear the rest of the line
end
# ============================================================================ #
"""
`run_stim_par(net::LIFNetwork, stim::Function, iv::Vector, dur_s::Float64,
    nrep::Integer, odir::AbstractString=<auto>, tstart::Float64=0.0)`
* net    - LIFNetwork
* stim   - a function that accepts **3** inputs, a neuron id, the current time,
           and the element of `iv` for the current trial
* iv     - Vector of items to serve as trailing input to `stim()` for each trial
* dur_s  - simulation duration in **SECONDS**
* nrep   - number of repeats of each element of `iv` to run
* odir   - (<auto>) output directory for temporart simulation data files
* tstart - (0.0) stimulation start time in **SECONDS**
"""
function run_sim_par(net::LIFNetwork, stim::Function, iv::Vector, dur::Float64,
    nrep::Integer, odir::AbstractString="", tstart::Time=0.0)

    if isempty(odir)
        odir = joinpath(default_dir, Dates.format(now(), "yyyymmddHHMM"))
    end

    !isdir(odir) && mkdir(odir)

    #dur should be in seconds
    if dur > 100.0
        error("Simulation duration is really long... check your units!")
    end

    ncell = length(net)

    ntrial = length(iv) * nrep
    evt_ts = tstart:dur:tstart + dur*(ntrial-1)

    sim_ids = [joinpath(odir, @sprintf("%03d.dat", k)) for k in 1:length(iv)]

    @sync @parallel for kt in eachindex(iv)

        local fstim(id, t) = stim(id, t, iv[kt])

        local ts = [Vector{Float64}() for x in 1:ncell]

        @inbounds for kr = 1:nrep

            local d = simulate(net, fstim, dur*1e3)

            local ke = (kt - 1) * nrep + kr

            @inbounds for k = 1:ncell
                append!(ts[k], d[:ts][k] + evt_ts[ke])
            end
        end

        write_sim_file(sim_ids[kt], ts)
    end


    ts = [Vector{Time}() for k in 1:ncell]
    for f in sim_ids
        tmp = read_sim_file(f)
        for k in eachindex(tmp)
            append!(ts[k], tmp[k])
        end
    end

    return ts, evt_ts
end
# =========================================================================== #
function write_sim_file(ofile::AbstractString, ts::Vector{Vector{Time}})
    open(ofile, "w") do io
        write(io, length(ts))
        for k in eachindex(ts)
            write(io, length(ts[k]))
        end

        for k in eachindex(ts)
            write(io, ts[k])
        end
    end
end
# =========================================================================== #
function read_sim_file(ifile::AbstractString)
    open(ifile, "r") do io
        ncell = read(io, Int64)
        nspk = read(io, Int64, ncell)

        ts = [Vector{Float64}(k) for k in nspk]
        for k = 1:ncell
            read!(io, ts[k])
        end

        return ts
    end
end
# =========================================================================== #

end # end module
