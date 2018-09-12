# TODO, NOTE
# We should be doing each f1xfm **AFTER** each trial as been cycle averaged

export f1xfm, cycle_mean
# ============================================================================ #
@inline calcf1(cs::Vector{C}, d::Matrix{T}) where {C<:Complex, T<:Number} =
    abs(vec(cs' * d))
# ---------------------------------------------------------------------------- #
@inline calcf1(cs::Vector{C}, d::Vector{T}) where {C<:Complex, T<:Number} =
    abs(dot(cs, d))
# ---------------------------------------------------------------------------- #
@inline f1basis(npt::Int) = exp(-im*2.0*pi*linspace(0.0, 1.0, npt))
# ============================================================================ #
function f1xfm(d::Array{T}, f1::AbstractFloat, dur::AbstractFloat) where {T<:Number}
    bpc = floor(Int, size(d, 1) / (dur * f1))
    tmp = cycle_mean(d, bpc)
    return calcf1(f1basis(bpc), tmp) * (2.0 / bpc)
end
# ============================================================================ #
function f1xfm(d::Vector{Matrix{T}}, f1::AbstractFloat,
    dur::AbstractFloat) where {T<:Number}

    siz = size(d[1], 1)
    check_size(x::Matrix{T}) = size(x, 1) == siz

    out = [Vector{Float64}(size(d[k], 2)) for k in 1:length(d)]

    if !all(check_size, d[2:end])
        # psth matricies have different number of bins so the f1basis must be
        # caclulated for each matrix
        @inbounds for k = 1:length(d)
            out[k] = f1xfm(d[k], f1, dur)
        end
    else
        # our sinusodial basis can be reused so pre-allocate
        bpc = floor(Int, size(d[1], 1) / (dur * f1))
        cs = f1basis(bpc)

        @inbounds for k = 1:length(d)
            out[k] = calcf1(cs, cycle_mean(d[k], bpc)) * (2.0 / bpc)
        end
    end

    return out
end
# ============================================================================ #
function f1xfm(d::Vector{Matrix{T}}, f1::Vector{F},
    dur::AbstractFloat) where {T<:Number, F<:AbstractFloat}

    length(f1) != length(d) && error("Number of temporal frequencies and number of trial groups *MUST* match")

    out = Vector{Vector{Float64}}(length(d))

    @inbounds for k = 1:length(d)
        out[k] = f1xfm(d[k], f1[k], dur)
    end

    return out
end
# ============================================================================ #
function cycle_pad(npt::Int, bpc::Integer)
    ncycle = floor(Int64, npt / bpc)
    if npt % bpc > 0
        ncycle += 1
        npad = (bpc * ncycle) - npt
    else
        npad = 0
    end
    return npad, ncycle
end
# ---------------------------------------------------------------------------- #
function cycle_mean(d::Vector{T}, bpc::Integer) where {T<:Number}
    pad, ncycle = cycle_pad(length(d), bpc)
    den = [fill(ncycle, bpc-pad); fill(ncycle-1, pad)]
    return vec(sum(reshape(cat(1, d, zeros(T, pad)), bpc, ncycle), 2)) ./ den
end
# ---------------------------------------------------------------------------- #
function cycle_mean(d::Matrix{T}, bpc::Integer) where {T<:Number}
    pad, ncycle = cycle_pad(size(d, 1), bpc)
    ntrial = size(d, 2)

    den = [fill(ncycle, bpc-pad); fill(ncycle-1, pad)]
    tmp = cat(1, d, zeros(pad, ntrial))

    out = Matrix{T}(bpc, ntrial)
    @inbounds @fastmath for k in 1:ntrial
        out[:,k] = sum(reshape(tmp[:,k], bpc, ncycle), 2) ./ den
    end

    return out
end
# ============================================================================ #
