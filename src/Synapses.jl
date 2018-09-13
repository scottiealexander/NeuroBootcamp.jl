module Synapses

using ..Lif, ..LifConfig

export BaseSynapse, Synapse, PlasticSynapse, update!, send!, receive!, reset!, show, isconnected

import Base: show, getindex, setindex!

# =========================================================================== #

abstract type BaseSynapse end
abstract type DynamicSynapse <: BaseSynapse end

# =========================================================================== #
# macro pcheck(var)
#     quote
#         @assert(!isnan($var) && !isinf($var))
#     end
# end
# =========================================================================== #
# enumeration of concrete synpase types
# =========================================================================== #
@enum SynapseType Static DC Depressing Facilitating Tsodyks
# =========================================================================== #
# struct like container for extra synapse parameters
# =========================================================================== #
mutable struct SynapseParameters
    weight::Float
    delay::Time
    tau::Time
    aux::Dict{Symbol, Any}
end
SynapseParameters(weight, delay, tau) = SynapseParameters(weight, delay, tau, Dict{Symbol, Any}())
# --------------------------------------------------------------------------- #
getindex(s::SynapseParameters, k::Symbol) = return s.aux[k]
# --------------------------------------------------------------------------- #
setindex!(s::SynapseParameters, x::Any, k::Symbol) = (s.aux[k] = x)
# --------------------------------------------------------------------------- #
function show(io::IO, s::SynapseParameters)
    #TODO, FIXME: this formatting turns out quite poor...
    fn = [:weight, :delay, :tau]
    str = ""
    for f in fn
        str *= string(f) * ": " * string(getfield(s, f)) * "\n"
    end
    for f in keys(s.aux)
        str *= string(f) * ": " * string(s[f]) * "\n"
    end
    print(io, str[1:end-1])
end
# =========================================================================== #
# interal synapse factory function
# =========================================================================== #
function build_synapse!(s::T, weight, delay, tau; args...) where T<:BaseSynapse
    s.weight = weight
    s.queue = Array{Float,1}()
    s.onset = 0.0 #+Inf
    s.p = SynapseParameters(weight, delay, tau, Dict(args))

    return s
end
# =========================================================================== #
# external synapse factory function
# =========================================================================== #
function Synapse(typ::SynapseType, args...)
    local out::BaseSynapse
    if typ == Static
        out = StaticSynapse(args...)
    elseif typ == DC
        out = DCSynapse(args...)
    elseif typ == Depressing
        out = DepressingSynapse(args...)
    elseif typ == Facilitating
        out = FacilitatingSynapse(args...)
    elseif typ == Tsodyks
        out = TsodyksSynapse(args...)
    else
        error("Invalid synapse type")
    end

    return out
end
# ---------------------------------------------------------------------------- #
function Synapse()
    return StaticSynapse()
end
# =========================================================================== #
# static synapse type
# =========================================================================== #
mutable struct StaticSynapse <: BaseSynapse
    weight::Float         #current weight
    queue::Array{Float,1} #spike queue
    onset::Float          #current spike onset time
    p::SynapseParameters    #extra parameters

    function StaticSynapse(weight, delay, tau)
        return build_synapse!(new(), weight, delay, tau)
    end
end
StaticSynapse() = StaticSynapse(0.0, +Inf, 0.0)
# =========================================================================== #
# DC synapse type - gap junction
# =========================================================================== #
mutable struct DCSynapse <: BaseSynapse
    weight::Float         #current weight
    queue::Array{Float,1} #gap junction queue
    onset::Float          #current spike onset time
    p::SynapseParameters    #extra parameters

    function DCSynapse(weight, delay)
        return build_synapse!(new(), weight, delay, 0.0)
    end
end
DCSynapse() = DCSynapse(0.0, 0.0)
# =========================================================================== #
# Depressing synapse type
# =========================================================================== #
mutable struct DepressingSynapse <: DynamicSynapse
    weight::Float         #current weight
    queue::Array{Float,1} #spike queue
    onset::Float          #current spike onset time
    p::SynapseParameters    #extra parameters

    function DepressingSynapse(weight, delay, tau, dx, taux)
        return build_synapse!(new(), weight, delay, tau, x=1.0, dx=dx, taux=taux)
    end
end
DepressingSynapse() = DepressingSynapse(0.0, +Inf, 0.0, 0.0, 0.0)
# =========================================================================== #
# Facilitating synapse type
# =========================================================================== #
mutable struct FacilitatingSynapse <: DynamicSynapse
    weight::Float         #current weight
    queue::Array{Float,1} #spike queue
    onset::Float          #current spike onset time
    p::SynapseParameters    #extra parameters

    function FacilitatingSynapse(weight, delay, tau, dx, taux)
        return build_synapse!(new(), weight, delay, tau, x=0.0, dx=dx, taux=taux)
    end
end
FacilitatingSynapse() = FacilitatingSynapse(0.0, +Inf, 0.0, 0.0, 0.0)
# =========================================================================== #
# Tsodyks synapse type
# =========================================================================== #
mutable struct TsodyksSynapse <: DynamicSynapse
    weight::Float         #current weight
    queue::Array{Float,1} #spike queue
    onset::Float          #current spike onset time
    p::SynapseParameters    #extra parameters

    function TsodyksSynapse(weight, delay, tau, U, taud, tauf)
        return build_synapse!(new(), weight, delay, tau, U=U, taud=taud,
            tauf=tauf, x=1.0, y=0.0, u=U)
    end
end
TsodyksSynapse() = TsodyksSynapse(0.0, +Inf, 0.0, 0.0, 0.0, 0.0)
# =========================================================================== #
# BaseSynapse methods
# =========================================================================== #
function send!(pre::N, s::T, spk::Bool, tnow::Time) where {T<:BaseSynapse, N<:BaseNeuron}
    if spk
        #send a spike to the synapse
        push!(s.queue, tnow + s.p.delay)
    end
end
# --------------------------------------------------------------------------- #
function receive!(post::N, s::T, dt::Time, tnow::Time) where {T<:BaseSynapse, N<:BaseNeuron}
    #receive 'input' from the current state of the synapse
    if !isempty(s.queue) && (tnow >= s.queue[1])
        s.onset = popfirst!(s.queue)
        spike = true
    else
        spike = false
    end

    # NOTE: this update *MUST* be performed before calculation of the current
    # <out> below
    update!(s, dt, spike, tnow)

    # alpha synapse
    # t = (tnow - s.onset) / s.p.tau
    # out = s.weight * t * exp(1.0-t)

    #
    # out = s.weight * (1.0 + 1.2*exp(-t/15.0) - exp(-t/20.0))
    # biexp(t, a, b) = (a*b/(a-b)) * (exp(-t.*b) - exp(-t.*a))
    # biexp2(t, τa, τb) = (1.0/(τb-τa)) * (exp(-t./τb) - exp(-t./τa))
    #   where τa = 1/a and τb = 1/b

    # @pcheck(s.weight); @pcheck(tnow); @pcheck(s.onset); @pcheck(s.p.tau)

    # exponential synapse
    if s.onset > 0.0
        out = s.weight * exp(-(tnow-s.onset)/s.p.tau)
        # out = s.weight
    else
        out = 0.0
    end

    # update!(s, dt, spike, tnow)

    return out
end
# --------------------------------------------------------------------------- #
function reset!(s::T) where {T<:BaseSynapse}
    s.onset = 0.0
    s.queue = Array{Float,1}()
    s.weight = s.p.weight
end
# --------------------------------------------------------------------------- #
function update!(s::T, dt::Time, spike::Bool, tnow::Time) where {T<:BaseSynapse}
    #nothing to do, synaptic weight is static
end
# --------------------------------------------------------------------------- #
function isconnected(s::T) where {T<:BaseSynapse}
    if s.p.weight == 0.0 || isinf(s.p.delay)
        b = false
    else
        b = true
    end
    return b
end
# --------------------------------------------------------------------------- #
function show(io::IO, s::T) where {T<:BaseSynapse}
    #TODO, FIXME: this formatting is quite poor...
    show(io, s.p.weight)
end
# =========================================================================== #
# DynamicSynapse methods
# =========================================================================== #
function update!(s::DepressingSynapse, dt::Time, spike::Bool, tnow::Time)
    if spike
        s.weight = s.p.weight * s.p[:x]
        s.p[:x] *= 1.0 - s.p[:dx]
    end
    s.p[:x] += dt * ((1.0 - s.p[:x]) / s.p[:taux])
    # s.p[:x] *= (1.0 - (s.p[:dx] * spike))
end
# =========================================================================== #
function update!(s::FacilitatingSynapse, dt::Time, spike::Bool, tnow::Time)
    # NOTE: for a facilitating synapse, the weight should represent the maximum
    # output that the synapse can produce (i.e. an asymptotic limit on <x>), so
    # we then the release probability to begin growing <x> (which asymptotes at
    # 1)
    error("FacilitatingSynapse update() function is not complete")
    if spike
        s.weight = s.p.weight * (1.0 + s.p[:x])
        s.p[:x] += s.p[:dx]
    end

    s.p[:x] += dt * ((-s.p[:x]) / s.p[:taux])
end
# =========================================================================== #
# function update!(s::FDSynapse, dt::Time, spike::Bool, tnow::Time)
#     if spike
#         s.weight = s.p.weight * (1.0 + s.p[:x])
#     end
#     # (1.0/(τb-τa)) * (exp(-t./τb) - exp(-t./τa))
#     s.p[:x] =
# end
# =========================================================================== #
function update2!(s::TsodyksSynapse, dt::Time, spike::Bool, tnow::Time)
    # this is just the forward Euler approximation of update!
    if spike
        s.weight = s.p.weight * s.p[:u] * s.p[:x]

        # NOTE: it appears that this update can occur either before or after the
        # integration, so long as it is triggered by a spike
        # NOTE: <x> depends on <u> and *MUST* be updated first
        s.p[:x] *= 1.0 - s.p[:u]
        s.p[:u] += s.p[:U] * (1.0 - s.p[:u])
    end

    s.p[:u] += ((s.p[:U] - s.p[:u]) / s.p[:tauf]) * dt
    s.p[:x] += ((1.0 - s.p[:x]) / s.p[:taud])  * dt
end
# =========================================================================== #
function update!(s::TsodyksSynapse, dt::Time, spike::Bool, tnow::Time)

    s.p[:u] = s.p[:U] + (s.p[:u] - s.p[:U]) * exp(-dt / s.p[:tauf])
    s.p[:x] = 1.0 + (s.p[:x] - 1.0) * exp(-dt / s.p[:taud])

    if spike
        s.weight = s.p.weight * s.p[:u] * s.p[:x]
        tmp = 1.0 - s.p[:u]
        s.p[:x] *= tmp
        s.p[:u] += s.p[:U] * tmp
    end
end
# =========================================================================== #
# function update!(s::TsodyksSynapse1, dt::Time, spike::Bool, tnow::Time)
# # https://github.com/nest/nest-simulator/blob/master/models/tsodyks_connection.h
# #   tauf: [ms] time constant for fascilitation
# #   taud: [ms] time constant for recovery
# #   U:    asymptotic value of probability of release
# #   x:    amount of resources in recovered state
# #   y:    amount of resources in active state
# #   u:    actual probability of release
# #
# #  U     - maximum probability of release [0,1]
# #  tau   - time constant of synaptic current in ms
# #  tauf  - time constant for facilitation in ms
# #  taud  - time constant for depression in ms
# #  x     - initial fraction of synaptic vesicles in the readily
# #          releasable pool [0,1]
# #  y     - initial fraction of synaptic vesicles in the synaptic
# #          cleft [0,1]
# # SEE ALSO:
# #https://github.com/nest/nest-simulator/blob/master/models/tsodyks2_connection.h
#
#     # NOTE: exact solution requires tnow.... which we don't actually have
#     h = -(tnow - s.onset)
#
#     pu = s.p[:tauf] == 0.0 ? 0.0 : exp(h / s.p[:tauf])
#     py = exp(h / s.p.tau)
#     pz = s.p[:taud] == 0.0 ? 0.0 : exp(h / s.p[:taud])
#
#     pxy = ((pz - 1.0) * s.p[:taud] - (py - 1.0) * s.p.tau) / (s.p.tau - s.p[:taud])
#
#     pxz = 1.0 - pz
#
#     z = 1.0 - s.p[:x] - s.p[:y]
#
#     @pcheck(pu); @pcheck(py); @pcheck(pz); @pcheck(pxy); @pcheck(pxz); @pcheck(z)
#
#     # NOTE: order is critical here
#     s.p[:u] *= pu
#     s.p[:x] += pxy * s.p[:y] + pxz * z
#     s.p[:y] *= py
#
#     s.p[:u] += s.p[:U] * (1.0 - s.p[:u])
#
#     dw = s.p[:u] * s.p[:x]
#
#     s.p[:x] -= dw
#     s.p[:y] += dw
#
#     s.weight = s.p.weight * dw
#
# end
# =========================================================================== #
# DCSynapse methods
# =========================================================================== #
function send!(pre::N, s::DCSynapse, spk::Bool, tnow::Time) where {N<:BaseNeuron}
    push!(s.queue, pre.vm)
    s.onset = tnow #NOTE: this is not needed...
end
# --------------------------------------------------------------------------- #
function receive!(post::N, s::DCSynapse, dt::Time, tnow::Time) where {N<:BaseNeuron}

    if length(s.queue) < floor(s.p.delay/dt) || tnow < dt #floor(1.0/dt)
        out = 0.0
    else
        out = (shift!(s.queue) - post.vm)*s.weight
    end

    return out
end
# =========================================================================== #
function test2(dt=0.05)
    tnow = 0.0
    dur = 1200.0

    pre = LIFNeuron()
    post = LIFNeuron()
    Lif.set!(pre, :xi, 0.0)
    Lif.set!(post, :xi, 0.0)
    Lif.set!(post, :tau, 10.0)

    #                    wi   delay tau   U   taud  tauf
    s = Synapse(Tsodyks, 0.05, 3.0, 3.0, 0.4, 100.0, 10.0)

    @show(s.p)

    times = tnow:dt:dur
    inp = zeros(length(times),)
    x = zeros(length(times),)
    w = zeros(length(times),)
    post_vm = zeros(length(times),)

    spikes = 33.0:33.0:1000.0
    nspike = length(spikes)
    inc = 2
    next_spike = spikes[1]

    for k = 1:length(times)

        if times[k] >= next_spike
            send!(pre, s, true, times[k])
            if inc <= nspike
                next_spike = spikes[inc]
                inc += 1
            else
                next_spike = NaN
            end
        else
            send!(pre, s, false, times[k])
        end

        inp[k] = receive!(post, s, dt, times[k])

        step!(pre, 0.0, dt, times[k])
        step!(post, inp[k], dt, times[k])
        post_vm[k] = post.vm
        x[k] = s.p[:x]
        w[k] = s.weight
    end

    return times, inp, x, w, post_vm
end
# =========================================================================== #
function test(dt=0.043)
    tnow = 0.0
    dur = 150.0

    #2 noise-less neurons
    pre = LIFNeuron()
    Lif.set!(pre, :xi, 0.0)
    # pre.p.tau = 1.0
    post = LIFNeuron()
    Lif.set!(post, :xi, 0.0)

    # s = DCSynapse(1.0, 0.1)
    # s = Synapse(.38, 3.0, 2.0, 0.4, 60)

    #                        wi  delay tau  dx   taux
    # s = Synapse(Facilitating, 0.05, 3.0, 2.0, 0.4, 40.0)

    # NOTE: taud (and tauf) *MUST* be > dt
    #   to approximate a static synapse, set U = 1, taud = 0.05, (tauf doesn't
    #   matter)
    #                    wi   delay tau   U   taud  tauf
    s = Synapse(Tsodyks, 0.05, 3.0, 2.0, 0.4, 100.0, 20.0)

    @show(s.p)

    times = tnow:dt:dur
    inp = zeros(length(times),)
    x = zeros(length(times),)
    w = zeros(length(times),)
    # u = zeros(length(times),)
    post_vm = zeros(length(times),)

    isi = 5.0
    nspike = 5
    spikes = [25.0:isi:(25.0+isi*(nspike-1)); 100.0]
    nspike += 1
    inc = 2
    next_spike = spikes[1]

    for k = 1:length(times)

        if times[k] >= next_spike
            send!(pre, s, true, times[k])
            if inc <= nspike
                next_spike = spikes[inc]
                inc += 1
            else
                next_spike = NaN
            end
        else
            send!(pre, s, false, times[k])
        end

        inp[k] = receive!(post, s, dt, times[k])

        step!(pre, 0.0, dt, times[k])
        step!(post, inp[k], dt, times[k])
        post_vm[k] = post.vm
        x[k] = s.p[:x]
        w[k] = s.weight
        # u[k] = s.p[:u]
    end

    return times, inp, x, w, post_vm
end
# =========================================================================== #

end #END MODULE

# delegation:
# https://github.com/JuliaLang/DataStructures.jl/blob/master/src/delegate.jl

# NOTE: synaptic plasticity
#   http://www.scholarpedia.org/article/Short-term_plasticity
#   http://www.briansimulator.org/docs/stp.html
#   Markram et al (1998). Differential signaling via the same axon of
#   neocortical pyramidal neurons, PNAS 95(9):5323-8
#
#   weight *= u*x # "Synaptic weights are modulated by the product u*x (in 0..1)
#                 #  which is taken before updating the variables"
#   dx/dt = (1-x)/taux
#   du/dt = (U-u)/tauu
#   x *= (1-u)    #NOTE: order of update matters, x is first
#   u += U*(1-u)
#
# where:
#   x = fraction of resources available following release
#   u = fraciton of available resources ready for useful
#   U = increment of u produced by a spike
#
# gamma distribution
# (x.^(k-1).*exp(-x/theta))./(theta^k*gamma(k))


################################################################################
# dt = 0.05
# tau1 = 10.0
# tau2 = 20.0
# t = 0:dt:1999*dt
# d1 = 1.2
# d2 = 1.0
# x = zeros(length(t))
# for k in eachindex(x)
#    d1 += dt * ((-d1 / tau1))
#    d2 += dt * ((-d2) / tau2)
#    x[k] = d1 - d2
# end

# dt = 0.05
# tau = 10.0
# t = 0:dt:1999*dt
# d = 1.2
# x = zeros(length(t))
# for k in eachindex(x)
#     d += dt * ((-d + 1.0) / tau)
#     x[k] = d
# end
