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
function plot_xcorr{T<:AbstractFloat}(ts1::Array{T}, ts2::Array{T};
            bin_size=0.0005, tmax=0.015, color="black", shift=0, tf=4.0,
            shift_color="cyan")

    xc, t = get_xcorr(ts1, ts2, bin_size, tmax)
    h = plot_xcorr(t, xc, bin_size=bin_size, tmax=tmax, color=color)
    if shift > 0
        sp = get_shift(ts1, ts2, bin_size, tmax, shift, tf)
        plot(t, sp, color=shift_color, linewidth=3)
    end
    return h
end
# ---------------------------------------------------------------------------- #
function plot_xcorr{T<:AbstractFloat}(ts1::Array{T}; bin_size=0.0005, tmax=0.015, color="black", shift::Int=0, tf=4.0)
    xc, t = get_xcorr(ts1, bin_size, tmax)
    h = plot_xcorr(t, xc, bin_size=bin_size, tmax=tmax, color=color)
    if shift > 0
        sp = get_shift(ts1, ts2, bin_size, tmax, shift, tf)
        plot(t, sp, color=shift_color, linewidth=3)
    end
    return h
end
# ============================================================================ #
