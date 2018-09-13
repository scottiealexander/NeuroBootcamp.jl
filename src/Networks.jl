module Networks

using Printf, ..Lif, ..LifConfig

include("./Synapses.jl")
using .Synapses

export LIFNetwork, simulate, reset!, connect!

# by exporting the Synapses module, where 'using' is used to import
# this module, Synapses will be imported as if 'import' had been used
export Synapses
# ============================================================================ #
abstract type NeuralNetwork end

# ============================================================================ #
# LIFNetwork
# ============================================================================ #
mutable struct LIFNetwork{S<:BaseSynapse, N<:BaseNeuron} <: NeuralNetwork
    neurons::Array{N,1}
    synapses::Array{S,2}
end
# ---------------------------------------------------------------------------- #
function LIFNetwork(x::Array{T,1}) where {T<:BaseNeuron}
    n = length(x)
    LIFNetwork(x, Array{BaseSynapse,2}(n, n))
end
# ---------------------------------------------------------------------------- #
function LIFNetwork(n::Integer, xi::Float64=0.2, typ::Type{T}=LIFNeuron) where {T<:BaseNeuron}
    x = Array{T,1}(undef, n)
    for k in eachindex(x)
        x[k] = typ()
        Lif.set!(x[k], :xi, xi)
    end
    LIFNetwork(x, Array{BaseSynapse,2}(undef, n, n))
end
# ---------------------------------------------------------------------------- #
function connect!(net::LIFNetwork)
    for k in 1:length(net)^2
        net.synapses[k] = Synapse()
    end
end
# # ---------------------------------------------------------------------------- #
# function connect!(net::LIFNetwork, args...)
#     d = Dict(args)
#     idx = keys(d)
#     n = length(net)
#
#     for r = 1:n
#         for c = 1:n
#             if (r,c) in idx
#                 net.synapses[r,c] = Synapse(d[(r,c)]...)
#             elseif !isassigned(net.synapses, r, c)
#                 net.synapses[r,c] = Synapse()
#             else
#                 @show(net.synapses[r,c])
#             end
#         end
#     end
# end
# ---------------------------------------------------------------------------- #
function connect!(net::LIFNetwork, k::Tuple{Integer, Integer}, p::Tuple)
    net.synapses[k...] = Synapse(p...)
end
# ---------------------------------------------------------------------------- #
Base.iterate(net::LIFNetwork) = (net[1], 1)
function Base.iterate(net::LIFNetwork, state::Integer)
    if state <= length(net)
        return (net[state], state+1)
    else
        return nothing
    end
end
Base.IteratorSize(net::LIFNetwork) = HasLength()
Base.IteratorEltype(net::LIFNetwork) = HasEltype()
Base.eltype(net::LIFNetwork) = eltype(net.neurons)
Base.length(net::LIFNetwork) = length(net.neurons)

Base.getindex(net::LIFNetwork, k::Integer) = net.neurons[k]
Base.getindex(net::LIFNetwork, k::UnitRange) = net.neurons[k]
Base.getindex(net::LIFNetwork, k::AbstractArray) = net.neurons[k]
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, net::LIFNetwork)
    str = @sprintf("%dx1 Array{LIFNeuron,1}:\n", length(net.neurons))
    print(io, str)
    print(io, net.neurons)
    print(io, "\n\n")
    print(io, net.synapses)
end
# ---------------------------------------------------------------------------- #
function reset!(net::LIFNetwork)
    for x in net.neurons
        #NOTE: not sure why we have to specify the module here...
        Lif.reset!(x)
    end
    for s in net.synapses
        #NOTE: not sure why we have to specify the module here...
        Synapses.reset!(s)
    end
end
# ---------------------------------------------------------------------------- #
function collect_input(net::LIFNetwork, k::Integer, dt::Time, tnow::Time)
    totl = 0.0
    #input is received column-wise
    @inbounds for s in net.synapses[:,k]
        if isconnected(s)
            totl += receive!(net[k], s, dt, tnow)
        end
    end
    return totl
end
# ---------------------------------------------------------------------------- #
function distribute_output(net::LIFNetwork, k::Integer, spk::Bool, tnow::Time)
    #output is distributed row-wise
    @inbounds for s in net.synapses[k,:]
        send!(net[k], s, spk, tnow)
    end
end
# ---------------------------------------------------------------------------- #
# function update!(net::LIFNetwork, stim::Vector{Float64}, dt::Time, tnow::Time)
function update!(net::LIFNetwork, fstim::Function, dt::Time, tnow::Time)
    vm = zeros(length(net),1)
    spk = falses(length(net),1)

    @inbounds for k = 1:length(net)

        #get synaptic and stimulus input
        inp = collect_input(net, k, dt, tnow) + fstim(k, tnow)

        # update Vm
        vm[k], spk[k] = step!(net[k], inp, dt, tnow)

        #propagate output to synapses
        distribute_output(net, k, spk[k], tnow)

        if spk[k]
            vm[k] = net[k].p.vspike
        end
    end

    return vm, spk
end
# ============================================================================ #
# helper functions
# ============================================================================ #
# include("NetworksExt.jl")
# ============================================================================ #
function default_stim(id::Integer, t::Time)
    return 0.0
end
# ============================================================================ #
function simulate(net::T, fstim::Function, duration::Real=500.0,
    record::Vector{Int64}=Int64[]) where {T<:NeuralNetwork}

    tnow = 0.0
    dt = 0.043
    record_vm = false

    reset!(net)

    d = Dict{Symbol, AbstractArray}()

    times = 0.0:dt:duration
    if !isa(record, Array)
        record = [record]
    end
    if !isempty(record) && eltype(record) <: Integer
        filter!(x -> (x > 0 && x <= length(net)), record)
        d[:vm] = zeros(length(times),length(record))
        record_vm = true
    end
    d[:ts] = [Vector{Time}() for k in 1:length(net)]

    for k in eachindex(times)

        vm, spk = update!(net, fstim, dt, times[k])
        if record_vm
            d[:vm][k,:] = vm[record]
        end

        for j in 1:length(net)
            if spk[j]
                #NOTE: timestamps are stored in *SECONDS*
                push!(d[:ts][j], times[k]*1e-3)
            end
        end
    end

    d[:time] = times

    return d
end
# ============================================================================ #
# function test(n=2, xi=0.5)
#
#     net = LIFNetwork(n,xi)
#     for x in net
#         Lif.set!(x, :xi, 0.0)
#     end
#     connect!(net, (1,2)=>(Synapses.Static, .54, 3.0, 2.0))
#     stim(id, t) = t > 100.0 && id == 1 ? 0.2 : 0.0
#
#     d = simulate(net, fstim=stim, duration=500, record=[1])
#     return d
# end
# # ============================================================================ #

end #END MODULE
