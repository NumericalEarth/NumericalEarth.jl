# Figure 32: horizontal RMS of T − WOA (row 1) and S − WOA (row 2) per level, as time × depth contours — the RMS
# companion of fig21, with the same 0–1000 m / 1000 m–bottom split. The signed horizontal mean in fig21 lets a warm
# and a cold region at one depth cancel; the RMS keeps both.
function fig32(caches, labels, cases)
    ncases = length(labels)
    temperature_rms_levels = range(0, 3; length = 13)
    salinity_rms_levels    = range(0, 0.6; length = 13)
    fig = Figure(size = (900 * ncases, 1200), fontsize = 14)

    upper = (-1000, 0)
    deep  = (-5500, -1000)

    function split_panel!(fig, row, col, t, z, data, levels, title_str)
        gl = fig[row, col] = GridLayout()
        ax_top = Axis(gl[1, 1]; title = title_str, ylabel = "Depth (m)")
        ax_bot = Axis(gl[2, 1]; xlabel = "Time (years)", ylabel = "Depth (m)")
        hm = contourf!(ax_top, t, z, data; levels, colormap = :thermal, extendhigh = :auto)
        contourf!(ax_bot, t, z, data; levels, colormap = :thermal, extendhigh = :auto)
        ylims!(ax_top, upper)
        ylims!(ax_bot, deep)
        linkxaxes!(ax_top, ax_bot)
        hidexdecorations!(ax_top; grid = false, ticks = false, minorticks = false)
        rowgap!(gl, 0)
        return hm
    end

    for (i, lab) in enumerate(labels)
        c = caches[lab]
        z = get_field(c, :depth)
        t, temperature_rms, salinity_rms = get_field(c, :rms_drift_profiles)
        hm = split_panel!(fig, 1, 2i - 1, t, z, temperature_rms, temperature_rms_levels, "$lab: RMS(T − WOA)")
        Colorbar(fig[1, 2i], hm; label = "deg C")
        hm = split_panel!(fig, 2, 2i - 1, t, z, salinity_rms, salinity_rms_levels, "$lab: RMS(S − WOA)")
        Colorbar(fig[2, 2i], hm; label = "g/kg")
    end
    savefig(fig, "fig32_rms_drift_hovmoller.png")
end
