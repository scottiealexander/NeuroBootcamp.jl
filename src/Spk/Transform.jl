# TODO, NOTE
# We should be doing each f1xfm **AFTER** each trial as been cycle averaged

export f1xfm, cycle_mean
# ============================================================================ #
@inline calcf1(cs::Vector{C}, d::Matrix{T}) where {C<:Complex,T<:Real} =
    vec(cs' * d) .* (2.0 / length(cs))
# ---------------------------------------------------------------------------- #
@inline calcf1(cs::Vector{C}, d::Vector{T}) where {C<:Complex,T<:Real} =
    dot(cs, d) .* (2.0 / length(cs))
# ---------------------------------------------------------------------------- #
@inline f1basis(npt::Int) = exp.(-im*2.0*pi*range(0.0, stop=1.0, length=npt))
# ============================================================================ #
function f1xfm(d::Array{T}, f1::AbstractFloat, dur::AbstractFloat) where {T<:Real}
    bpc = floor(Int, size(d, 1) / (dur * f1))
    tmp = cycle_mean(d, bpc)
    return calcf1(f1basis(bpc), tmp)
end
# ============================================================================ #
function f1xfm(d::Vector{Matrix{T}}, f1::AbstractFloat,
    dur::AbstractFloat) where {T<:Real}

    siz = size(d[1], 1)
    check_size(x::Matrix{T}) = size(x, 1) == siz

    out = [Vector{Float64}(undef, size(d[k], 2)) for k in 1:length(d)]

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
            out[k] = calcf1(cs, cycle_mean(d[k], bpc))
        end
    end

    return out
end
# ============================================================================ #
function f1xfm(d::Vector{Matrix{T}}, f1::Vector{F},
    dur::AbstractFloat) where {T<:Real, F<:AbstractFloat}

    length(f1) != length(d) && error("Number of temporal frequencies and number of trial groups *MUST* match")

    out = Vector{Vector{Float64}}(undef, length(d))

    @inbounds for k = 1:length(d)
        out[k] = f1xfm(d[k], f1[k], dur)
    end

    return out
end
# ============================================================================ #
function cycle_pad(npt::Int, bpc::Integer)
    ncycle = floor(Int, npt / bpc)
    if npt % bpc > 0
        ncycle += 1
        npad = (bpc * ncycle) - npt
    else
        npad = 0
    end
    return npad, ncycle
end
# ---------------------------------------------------------------------------- #
function cycle_mean(d::Vector{T}, bpc::Integer) where {T<:Real}
    pad, ncycle = cycle_pad(length(d), bpc)
    den = [fill(ncycle, bpc-pad); fill(ncycle-1, pad)]
    return vec(
                sum(
                    reshape(cat(d, zeros(T, pad), dims=1), bpc, ncycle),
                    dims=2
                )
            ) ./ den
end
# ---------------------------------------------------------------------------- #
function cycle_mean(d::Matrix{T}, bpc::Integer) where {T<:Real}
    pad, ncycle = cycle_pad(size(d, 1), bpc)
    ntrial = size(d, 2)

    den = [fill(ncycle, bpc-pad); fill(ncycle-1, pad)]
    tmp = cat(d, zeros(pad, ntrial), dims=1)

    out = Matrix{T}(undef, bpc, ntrial)
    @inbounds @fastmath for k in 1:ntrial
        out[:,k] = sum(reshape(tmp[:,k], bpc, ncycle), dims=2) ./ den
    end

    return out
end
# ============================================================================ #
