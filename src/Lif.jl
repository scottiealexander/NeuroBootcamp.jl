module Lif

export BaseNeuron, LIFNeuron, TSNeuron,
       show, reset!, step! #is_refractory, is_spiking

using LifConfig

import Base: show

abstract type BaseNeuron end
abstract type BaseParameters end
# ============================================================================ #
mutable struct LIFParameters <: BaseParameters
    # parameters in mV
    vrest::Float      #resting potential (mV)
    vreset::Float     #post-spike reset (mV)
    vspike::Float     #spike amplitude (mV)
    threshold::Float  #spike threshold (mV)

    # misc parameters
    rm::Float         #membrane resistance (Mohm)
    #cm::Float         #membrane capacitance (uF)
    tau::Float        #membrane time constant (ms)
    refractory_period::Float #refractory period (ms)
    xi::Float         #noise factor (std of Gaussian) (mV)

    # Ohm = kg * m^2 * s^-3 * A^-2
    # Farad = s^4 * A^2 * m^-2 * kg^-1

    function LIFParameters()
        return new(
            0.0,        #vrest
            -0.2,       #vreset
            1.22 * 6.0, #vspike
            1.22,       #threshold
            2.5,        #rm
            4.5,        #tau
            3.0,        #refractory_period
            0.2         #xi
        )
    end
end
# ============================================================================ #
mutable struct LIFNeuron <: BaseNeuron
    p::LIFParameters
    vm::Float
    refractory_start::Float
    function LIFNeuron()
        p = LIFParameters()
        return new(p, p.vrest, -Inf)
    end
end
# ---------------------------------------------------------------------------- #
function set!(x::LIFNeuron, f::Symbol, v::Float)
    if f in fieldnames(typeof(x))
        setfield!(x, f, v)
    elseif f in fieldnames(typeof(x.p))
        setfield!(x.p, f, v)
    else
        warn("Failed to set field '" * string(f) * "': field not found")
    end
end
# ---------------------------------------------------------------------------- #
function show(io::IO, x::LIFNeuron)
    print(io,x.vm)
end
# ---------------------------------------------------------------------------- #
function reset!(x::LIFNeuron)
    x.vm = x.p.vrest
    x.refractory_start = -Inf
end
# ---------------------------------------------------------------------------- #
function is_refractory(x::LIFNeuron, tnow::Number)
    return tnow <= (x.refractory_start + x.p.refractory_period)
end
# ---------------------------------------------------------------------------- #
function is_spiking(x::LIFNeuron)
    return x.vm > x.p.threshold
end
# ---------------------------------------------------------------------------- #
function step!(x::LIFNeuron, inp::Float, dt::Float, tnow::Float)
    spike = false
    if !is_refractory(x, tnow)

        # ######################
        # # NOISE ADDED HERE   #
        # ######################
        xi = randn()*x.p.xi / sqrt(dt)

        # NOTE: the correct equation is (Vrest - Vm + (Isyn * Rm)) / tau
        # see Vm equation derivation in quals2 proposal file vmeq_derivation.md
        # x.vm += (((x.p.vrest - x.vm + (inp * x.p.rm)) / x.p.tau) + xi)  * dt

        vinf = x.p.vrest + inp * x.p.rm
        x.vm = vinf + (x.vm - vinf) * exp(-dt / x.p.tau) + (xi*dt)


        if is_spiking(x)
            x.vm = x.p.vreset
            x.refractory_start = tnow
            spike = true
        end
    else
        x.vm = x.p.vrest #+ xi
    end

    return x.vm, spike
end
# ============================================================================ #
# abstract IndexGenerator
#
# # cycle through indicies 1:N
# type SerialIndex <: IndexGenerator
#     idx::UnitRange
#     ptr::Int
#     SerialIndex(n::Int) = new(1:n, 1)
# end
# # cycle through a permuted verison of 1:N, re-shuffling after reaching the end
# type ShuffledIndex <: IndexGenerator
#     idx::Vector{Int}
#     ptr::Int
#     ShuffledIndex(n::Int) = new(randperm(n), 1)
# end
# # pick a random index with replacement
# type RandomIndex <: IndexGenerator
#     idx::UnitRange
#     cur::Int
#     function RandomIndex(n::Int)
#         self = new(1:n, 0)
#         increment!(self)
#         return self
#     end
# end
# # ---------------------------------------------------------------------------- #
# reset!(x::ShuffledIndex) = shuffle!(x.idx)
# reset!(x::IndexGenerator) = nothing
# # ---------------------------------------------------------------------------- #
# function increment!(x::IndexGenerator)
#     if x.ptr == length(x.idx)
#         x.ptr = 1
#         reset!(x)
#     else
#         x.ptr += 1
#     end
# end
# function increment!(x::RandomIndex)
#     x.cur = rand(x.idx, 1)[1]
# end
# index(x::IndexGenerator) = x.idx[x.ptr]
# index(x::RandomIndex) = x.cur
# # ============================================================================ #
# type TSParameters <: BaseParameters
#     vspike::Float
#     TSParameters() = new(1.0)
# end
# type TSNeuron{T<:IndexGenerator} <: BaseNeuron
#     ts::Vector{Vector{Float}}
#     p::TSParameters
#     ptr::Int
#     idx::T
#     kt::Int
#     function TSNeuron{T<:IndexGenerator}(ts::Vector{Vector{Float}},
#         idx::T=SerialIndex(length(ts)))
#
#         #NOTE timestamps are in seconds but simulations run in ms
#         return new(ts * 1e3, TSParameters(), 1, idx, 0)
#     end
# end
# # ---------------------------------------------------------------------------- #
# function reset!(x::TSNeuron)
#     x.ptr = 1
#     x.kt = index(x.idx)
# end
# # ---------------------------------------------------------------------------- #
# function set!(x::TSNeuron, f::Symbol, v::Float)
#     #nothing to set
# end
# # ---------------------------------------------------------------------------- #
# function show(io::IO, x::TSNeuron)
#     print(io, @sprintf("TSNeuron(%d)", length(x.ts)))
# end
# # ---------------------------------------------------------------------------- #
# function step!(x::TSNeuron, inp, dt, tnow)
#     spike = false
#     if x.ptr > 0 && ((tnow + dt) > x.ts[x.kt][x.ptr])
#         spike = true
#         x.ptr = x.ptr < length(x.ts[x.kt]) ? x.ptr+1 : -1
#     end
#     return 0.0, spike
# end
# ============================================================================ #
end #END MODULE

# vinf = x.p.vrest + stim * x.p.rm
# x.vm = vinf + (x.vm - vinf) * exp(-dt / x.p.tau)

# x.vm += dt * ((x.p.vrest - x.vm) + (inp * x.p.rm) + xi) / x.p.tau
