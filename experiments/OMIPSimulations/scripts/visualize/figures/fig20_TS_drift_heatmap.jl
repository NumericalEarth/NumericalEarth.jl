# Figure 20: T (row 1) and S (row 2) horizontal-mean drift as time × depth contours.
function fig20(caches, labels, cases)
    ncases = length(labels)
    temperature_drift_levels = range(-1.6, 1.6; length = 17)
    salinity_drift_levels    = range(-0.1, 0.1; length = 21)
    fig = Figure(size = (900 * ncases, 1200), fontsize = 14)

    for (i, lab) in enumerate(labels)
        c                   = caches[lab]
        depth               = get_field(c, :depth)
        time_in_years       = get_field(c, :drift_time_in_years)
        temperature_drift   = get_field(c, :temperature_drift)
        salinity_drift      = get_field(c, :salinity_drift)

        ax_temperature = Axis(fig[1, 2i-1]; xlabel = "Time (years)", ylabel = "Depth (m)",
                              title = "$lab: ΔT (deg C)")
        hm_temperature = contourf!(ax_temperature, time_in_years, depth, temperature_drift;
                                    levels = temperature_drift_levels,
                                    colormap = :balance, extendlow = :auto, extendhigh = :auto)
        ylims!(ax_temperature, (-5500, 0))
        Colorbar(fig[1, 2i], hm_temperature; label = "deg C")

        ax_salinity = Axis(fig[2, 2i-1]; xlabel = "Time (years)", ylabel = "Depth (m)",
                           title = "$lab: ΔS (PSU)")
        hm_salinity = contourf!(ax_salinity, time_in_years, depth, salinity_drift;
                                 levels = salinity_drift_levels,
                                 colormap = :balance, extendlow = :auto, extendhigh = :auto)
        ylims!(ax_salinity, (-5500, 0))
        Colorbar(fig[2, 2i], hm_salinity; label = "PSU")
    end

    savefig(fig, "fig20_TS_drift_heatmap.png")
end
