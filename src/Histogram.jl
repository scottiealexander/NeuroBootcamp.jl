module Histogram

# simple histogram functions copied from Julia v0.4.7 to avoid having to
# pull in all of Statistics.jl

export hist, hist!

function histrange{T<:AbstractFloat,N}(v::AbstractArray{T,N}, n::Integer)
    nv = length(v)
    if nv == 0 && n < 0
        throw(ArgumentError("number of bins must be ≥ 0 for an empty array, got $n"))
    elseif nv > 0 && n < 1
        throw(ArgumentError("number of bins must be ≥ 1 for a non-empty array, got $n"))
    end
    if nv == 0
        return 0.0:1.0:0.0
    end
    lo, hi = extrema(v)
    if hi == lo
        step = 1.0
    else
        bw = (hi - lo) / n
        e = 10.0^floor(log10(bw))
        r = bw / e
        if r <= 2
            step = 2*e
        elseif r <= 5
            step = 5*e
        else
            step = 10*e
        end
    end
    start = step*(ceil(lo/step)-1)
    nm1 = ceil(Int,(hi - start)/step)
    return start:step:(start + nm1*step)
end

function histrange{T<:Integer,N}(v::AbstractArray{T,N}, n::Integer)
    nv = length(v)
    if nv == 0 && n < 0
        throw(ArgumentError("number of bins must be ≥ 0 for an empty array, got $n"))
    elseif nv > 0 && n < 1
        throw(ArgumentError("number of bins must be ≥ 1 for a non-empty array, got $n"))
    end
    if nv == 0
        return 0:1:0
    end
    if n <= 0
        throw(ArgumentError("number of bins n = $n must be positive"))
    end
    lo, hi = extrema(v)
    if hi == lo
        step = 1
    else
        bw = (hi - lo) / n
        e = 10^max(0,floor(Int,log10(bw)))
        r = bw / e
        if r <= 1
            step = e
        elseif r <= 2
            step = 2*e
        elseif r <= 5
            step = 5*e
        else
            step = 10*e
        end
    end
    start = step*(ceil(lo/step)-1)
    nm1 = ceil(Int,(hi - start)/step)
    return start:step:(start + nm1*step)
end

# Sturges' formula
function sturges(n::Integer)
    n==0 && return one(n)
    return ceil(Int, log2(n)) + 1
end

function hist!{HT}(h::AbstractArray{HT}, v::AbstractVector, edg::AbstractVector)
    n = length(edg) - 1
    length(h) == n || throw(DimensionMismatch("length(histogram) must equal length(edges) - 1"))
    fill!(h, zero(HT))
    for x in v
        i = searchsortedfirst(edg, x)-1
        if 1 <= i <= n
            h[i] += 1
        end
    end
    return edg, h
end

hist(v::AbstractVector, edg::AbstractVector) = hist!(Array{Int}(length(edg)-1), v, edg)
hist(v::AbstractVector, n::Integer) = hist(v,histrange(v,n))
hist(v::AbstractVector) = hist(v,sturges(length(v)))

end
