using Plot, PyPlot

export plot_xcorr

# ============================================================================ #
function plot_xcorr(t::AbstractVector, xc::Vector; bin_size=0.0005, tmax=0.015, color="black")
    h = qbar(t, xc, bin_size*.8, color)
    xlabel("Time lag (sec)")
    ylabel("# of spikes")
    tmax += bin_size
    xlim(-tmax, tmax)
    return h
end
# ---------------------------------------------------------------------------- #
function plot_xcorr(ts1::Vector{T}, ts2::Vector{T};
            bin_size=0.0005, tmax=0.015, color="black", shift=0, tf=4.0,
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
function plot_xcorr(ts1::Vector{T}; bin_size=0.0005, tmax=0.015, color="black",
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
function plot_raster(ts::Vector{T}, evt::AbstractArray{T,1}, dur::AbstractFloat,
    color::String="blue") where {T<:AbstractFloat}

    bin_size = 0.001
    nbin = floor(Int, dur / bin_size)
    p = psth(ts, collect(evt), 0:nbin, bin_size)[1];
    h = default_figure()
    for k = 1:size(p,2)
       idx = find(p[:,k])
       plot(idx * bin_size, fill(k, length(idx)), ".", color=color)
   end
   xlabel("Time (sec)", fontsize=16)
   ylabel("Trial #", fontsize=16)
   return h
end
# ============================================================================ #
function plot_cycle_mean(ts::Vector{T}, evt::AbstractArray{T,1},
    dur::AbstractFloat, tf::AbstractFloat, color::String="blue") where {T<:AbstractFloat}

    bin_size = 0.001
    nbin = floor(Int, dur / bin_size)
    bpc = floor(Int, 1/(tf * bin_size))
    p = psth(ts, collect(evt), 0:nbin, bin_size)[1];
    h = default_figure()
    cm = cycle_mean(sum(p, 2), bpc)
    h = plot(linspace(0, 1/tf, length(cm)), cm, color=color)
    xlabel("Time (sec)", fontsize=16)
    ylabel("Firing rate (Hz)", fontsize=16)
    return h
end
# ============================================================================ #
