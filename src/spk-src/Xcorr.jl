
export get_xcorr, get_shift, xpsd

# ============================================================================ #
function get_xcorr{T<:AbstractFloat}(ts1::Vector{T}, ts2::Vector{T},
    bs::AbstractFloat, tm::AbstractFloat)

    nbin = round(Int64, tm/bs)
    bins = -nbin:nbin

    xc = psth(ts2, ts1, bins, bs)[1]
    xc = vec(sum(xc,2))
    t = (-tm:bs:tm) - (bs/2.0)

    return xc, t
end
# ============================================================================ #
function get_xcorr{T<:AbstractFloat}(ts::Vector{T}, bs::AbstractFloat,
    tm::AbstractFloat)

    xc, t = get_xcorr(ts, ts, bs, tm)
    kc = floor(Int, length(xc) / 2) + 1
    xc[kc] = T(0.0)

    return xc, t
end
# ============================================================================ #
function get_shift{T<:AbstractFloat}(ts1::Vector{T}, ts2::Vector{T},
    bs::AbstractFloat, tm::AbstractFloat, nshift::Integer, tf::AbstractFloat)

    len = round(Int64, (tm*2.0)/bs + 1)
    sp = zeros(len,)
    for k = 1:nshift
        sp = sp .+ get_xcorr(ts1, ts2+(Float64(k)/tf), bs, tm)[1]
    end
    return sp ./ Float64(nshift)
end
# ============================================================================ #
function xpsd{T<:AbstractFloat}(ts1::Vector{T}, ts2::Vector{T},
    bin_size::AbstractFloat, tmax::AbstractFloat, fmax::AbstractFloat)

    p, f = xpsd(ts1, ts2, bin_size, tmax)
    b = f .<= fmax

    return p[b], f[b]
end
# ---------------------------------------------------------------------------- #
function xpsd{T<:AbstractFloat}(ts1::Vector{T}, ts2::Vector{T},
    bin_size::AbstractFloat, tmax::AbstractFloat)

    xc, t = get_xcorr(ts1, ts2, bin_size, tmax)
    npt = nextpow(2, length(xc))
    xc = cat(1, xc, zeros(npt-length(xc),))
    nf = floor(Int, npt/2.0) + 1

    p = fft(xc)

    return abs2(p[1:nf]), (0:nf-1) / (npt * bin_size)
end
# ============================================================================ #
