# ============================================================================ #
function plot_with_error(x::AbstractArray, y::Vector, yerr::Vector,
    col::AbstractString, ax=nothing; linewidth=4.0, args...)
    plot_with_error(x, y, yerr, parse(Colorant, col), ax, args...)
end
# ---------------------------------------------------------------------------- #
function plot_with_error(x::AbstractArray, y::Vector, yerr::Vector,
    col::ColorTypes.RGB, ax=nothing; linewidth=4.0, args...)
    if ax == nothing
        ax = default_axes()
    end
    col_array = [col.r, col.g, col.b]
    fcol, ecol = shading_color(col_array)
    ax[:fill_between](x, y.-yerr, y.+yerr, facecolor=fcol, edgecolor=ecol)
    ax[:plot](x, y, "-", color=col_array, linewidth=linewidth, args...)
end
# ---------------------------------------------------------------------------- #
function shading_color(col::Vector{T}) where {T<:Number}
    ferr = 8.0
    fedge = 0.25
    orig = convert(HSV, RGB(col...))
    hsv = HSV(orig.h, orig.s/ferr, 1.0 - (abs(1.0 - orig.s)^ferr/ferr))
    col_err = hsv2rgb(hsv)
    col_edge = (1.0 - fedge) * col_err + fedge * col
    return col_err, col_edge
end
# ---------------------------------------------------------------------------- #
function hsv2rgb(x::ColorTypes.HSV{T}) where {T<:Number}
    # xv = HSV(min([x.h, x.s, x.v], T(1.0))...)
    rgb = convert(RGB, x)
    return T[rgb.r, rgb.g, rgb.b]
end
# ============================================================================ #
plot_hist(x::AbstractVector) = _plot_hist(hist(x)...)
plot_hist(x::AbstractVector, n::Integer) = _plot_hist(hist(x, n)...)
plot_hist(x::AbstractVector, e::AbstractVector) = _plot_hist(hist(x, e)...)
# ============================================================================ #
edges2centers(edges::AbstractVector) = edges[1:end-1] .+ (step(edges)/2.0)
# ============================================================================ #
function _plot_hist(edges::AbstractVector, counts::AbstractVector)
    return qbar(edges2centers(edges), counts, step(edges))
end
# ============================================================================ #
function vline(x::Number; args...)
    plot([x, x], ylim(); args...)
end
# --------------------------------------------------------------------------- #
function vline(x::AbstractArray; args...)
    for item in x
        vline(item; args...)
    end
end
# ============================================================================ #
function hline(y::Number; args...)
    plot(xlim(), [y, y]; args...)
end
# --------------------------------------------------------------------------- #
function hline(y::AbstractArray; args...)
    for item in y
        hline(item; args...)
    end
end
# ============================================================================ #
function qplot(x, y)
    h = default_figure()
    p = plot(x, y, color="blue", linewidth=3.0)

    return h
end
# ============================================================================ #
function qplot(y)
    x = 1:length(y)
    h = qplot(x, y)
    return h
end
# ============================================================================ #
function qbar(x, y, width, col="black")
    h = default_figure()

    #width = absolute bar width
    # add error with yerr=...
    p = bar(x, y, width, color=col)

    return h
end
# ============================================================================ #
function qbar(y)
    x = 1:length(y)
    h = qbar(x-0.5, y, 1)
    return h
end
# ============================================================================ #
function closeall()
    plt[:close]()
end
# ============================================================================ #
default_figure() = default_figure(figure())
# ---------------------------------------------------------------------------- #
function default_figure(h::PyPlot.Figure)

    if !plt[:fignum_exists](h[:number])
        h = figure()
    end

    h[:set_facecolor]("white")
    ax = h[:get_axes]()
    if !isempty(ax)
        for x in ax
            delaxes(x)
        end
    end

    ax = h[:add_axes](default_axes())

    return h
end
# ============================================================================ #
function default_axes(ax=nothing)
    if ax == nothing
        ax = PyPlot.axes()
    end
    #set ticks to face out
    ax[:tick_params](direction="out", length=8.0, width=4.0)

    #turn off top and right axes
    ax[:spines]["right"][:set_visible](false)
    ax[:spines]["top"][:set_visible](false)

    #remove top and right tick marks
    tmp = ax[:get_xaxis]()
    tmp[:tick_bottom]()

    tmp = ax[:get_yaxis]()
    tmp[:tick_left]()

    ax[:spines]["left"][:set_linewidth](4.0)
    ax[:spines]["bottom"][:set_linewidth](4.0)

    return ax
end
# ============================================================================ #

# ============================================================================ #
# NOTES
# ============================================================================ #
#  #update figure
#  h[:canvas][:draw]()
#
#  #close all figures
#  plt[:close]()
