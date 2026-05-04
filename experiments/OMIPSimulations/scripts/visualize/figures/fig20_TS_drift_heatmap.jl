# Figure 20: T (row 1) and S (row 2) horizontal-mean drift as time × depth
# contours. Each per-case cell is a vertically-split panel: top half spans
# 0 → 1000 m (upper-ocean detail), bottom half spans 1000 m → bottom. The
# two sub-axes share the time axis and sit flush with no gap, so visually
# they read as a single panel with a piecewise-linear depth coordinate.
function fig20(caches, labels, cases)
    ncases = length(labels)
    temperature_drift_levels = range(-1.6, 1.6; length = 17)
    salinity_drift_levels    = range(-0.1, 0.1; length = 21)
    fig = Figure(size = (900 * ncases, 1200), fontsize = 14)

    upper = (-1000, 0)
    deep  = (-5500, -1000)

    # Build a vertically-split (top: 0-1000 m, bot: 1000 m-bottom) contour
    # panel at `fig[row, col]`. Returns the heatmap handle for the colorbar.
    function split_panel!(fig, row, col, t, z, data, levels, title_str)
        gl = fig[row, col] = GridLayout()
        ax_top = Axis(gl[1, 1]; title = title_str, ylabel = "Depth (m)")
        ax_bot = Axis(gl[2, 1]; xlabel = "Time (years)", ylabel = "Depth (m)")
        hm = contourf!(ax_top, t, z, data; levels = levels, colormap = :balance,
                        extendlow = :auto, extendhigh = :auto)
        contourf!(ax_bot, t, z, data; levels = levels, colormap = :balance,
                   extendlow = :auto, extendhigh = :auto)
        ylims!(ax_top, upper)
        ylims!(ax_bot, deep)
        linkxaxes!(ax_top, ax_bot)
        hidexdecorations!(ax_top; grid = false, ticks = false, minorticks = false)
        rowgap!(gl, 0)
        return hm
    end

    for (i, lab) in enumerate(labels)
        c  = caches[lab]
        z  = get_field(c, :depth)
        t  = get_field(c, :drift_time_in_years)
        ΔT = get_field(c, :temperature_drift)
        ΔS = get_field(c, :salinity_drift)

        hm_T = split_panel!(fig, 1, 2i-1, t, z, ΔT,
                             temperature_drift_levels, "$lab: ΔT (deg C)")
        Colorbar(fig[1, 2i], hm_T; label = "deg C")

        hm_S = split_panel!(fig, 2, 2i-1, t, z, ΔS,
                             salinity_drift_levels, "$lab: ΔS (PSU)")
        Colorbar(fig[2, 2i], hm_S; label = "PSU")
    end

    savefig(fig, "fig20_TS_drift_heatmap.png")
end
