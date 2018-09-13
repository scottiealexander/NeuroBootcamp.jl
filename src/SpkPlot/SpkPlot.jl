module SpkPlot

using PyPlot, Colors, ColorTypes

using ..Spk, ..Histogram

include("./plot_helpers.jl")

export plot_xcorr, plot_raster, plot_raster!, plot_cycle_mean, plot_cycle_mean!

export cla, default_figure, default_axes

# ============================================================================ #
cla() = close("all")
# ============================================================================ #
function plot_xcorr(t::AbstractVector, xc::Vector; bin_size=0.0005, tmax=0.015,
    color="black")

    h = qbar(t, xc, bin_size*.8, color)
    xlabel("Time lag (sec)")
    ylabel("# of spikes")
    tmax += bin_size
    xlim(-tmax, tmax)
    return h
end
# ---------------------------------------------------------------------------- #
function plot_xcorr(ts1::Vector{T}, ts2::Vector{T};
            bin_size=0.0005, tmax=0.02, color="black", shift=0, tf=4.0,
            shift_color="cyan") where {T<:AbstractFloat}

    xc, t = get_xcorr(ts1, ts2, bin_size, tmax)
    h = plot_xcorr(t, xc, bin_size=bin_size, tmax=tmax, color=color)
    if shift > 0
        sp = get_shift(ts1, ts2, bin_size, tmax, shift, tf)
        plot(t, sp, color=shift_color, linewidth=3)
    end
    return h
end
# ---------------------------------------------------------------------------- #
function plot_xcorr(ts1::Vector{T}; bin_size=0.0005, tmax=0.02, color="black",
    shift::Int=0, tf=4.0) where {T<:AbstractFloat}

    xc, t = get_xcorr(ts1, bin_size, tmax)
    h = plot_xcorr(t, xc, bin_size=bin_size, tmax=tmax, color=color)
    if shift > 0
        sp = get_shift(ts1, ts2, bin_size, tmax, shift, tf)
        plot(t, sp, color=shift_color, linewidth=3)
    end
    return h
end
# ============================================================================ #
function plot_raster!(ax, ts::Vector{T}, evt::AbstractArray{T,1}, dur::AbstractFloat,
    color::String="blue") where {T<:AbstractFloat}

    bin_size = 0.001
    nbin = floor(Int, dur / bin_size)
    p = psth(ts, evt, 0:nbin-1, bin_size)[1];

    for k = 1:size(p,2)
       idx = findall(p[:,k] .> 0)
       ax[:plot](idx * bin_size, fill(k, length(idx)), ".", color=color)
   end
   ax[:set_xlabel]("Time (sec)", fontsize=16)
   ax[:set_ylabel]("Trial #", fontsize=16)
   return ax
end
function plot_raster(ts::Vector{T}, evt::AbstractArray{T,1}, dur::AbstractFloat,
    color::String="blue") where {T<:AbstractFloat}

    h = default_figure()
    return plot_raster!(h[:axes][1], ts, evt, dur, color)
end
# ============================================================================ #
function plot_cycle_mean!(ax, ts::Vector{T}, evt::AbstractArray{T,1},
    dur::AbstractFloat, tf::AbstractFloat, color::String="blue") where {T<:AbstractFloat}

    bin_size = 0.001
    nbin = floor(Int, dur / bin_size)
    bpc = floor(Int, 1/(tf * bin_size))
    p = psth(ts, evt, 0:nbin-1, bin_size)[1];

    cm = cycle_mean(sum(p, dims=2), bpc)
    h = ax[:plot](range(0, stop=1/tf, length=length(cm)), cm, color=color)
    ax[:set_xlabel]("Time (sec)", fontsize=16)
    ax[:set_ylabel]("Firing rate (Hz)", fontsize=16)
    return h
end
# ---------------------------------------------------------------------------- #
function plot_cycle_mean(ts::Vector{T}, evt::AbstractArray{T,1},
    dur::AbstractFloat, tf::AbstractFloat, color::String="blue") where {T<:AbstractFloat}
    h = default_figure()
    return plot_cycle_mean!(h[:axes][1], ts, evt, dur, tf, color)
end
# ============================================================================ #
end
