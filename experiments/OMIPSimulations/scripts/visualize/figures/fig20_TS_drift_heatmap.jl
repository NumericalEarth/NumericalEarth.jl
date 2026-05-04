# Figure 20: T (rows 1-2) and S (rows 3-4) horizontal-mean drift as
# time × depth contours, split into upper-ocean (0 → 1000 m) and
# deep-ocean (1000 m → bottom) bands so the surface signal is not
# compressed by the deep-ocean depth range.
function fig20(caches, labels, cases)
    ncases = length(labels)
    temperature_drift_levels = range(-1.6, 1.6; length = 17)
    salinity_drift_levels    = range(-0.1, 0.1; length = 21)
    fig = Figure(size = (900 * ncases, 2200), fontsize = 14)

    upper = (-1000, 0)
    deep  = (-5500, -1000)

    for (i, lab) in enumerate(labels)
        c     = caches[lab]
        z     = get_field(c, :depth)
        t     = get_field(c, :drift_time_in_years)
        ΔT    = get_field(c, :temperature_drift)
        ΔS    = get_field(c, :salinity_drift)

        # Temperature drift, upper 1000 m
        ax = Axis(fig[1, 2i-1]; xlabel = "Time (years)", ylabel = "Depth (m)",
                  title = "$lab: ΔT (0 → 1000 m, deg C)")
        hm = contourf!(ax, t, z, ΔT; levels = temperature_drift_levels,
                       colormap = :balance, extendlow = :auto, extendhigh = :auto)
        ylims!(ax, upper)
        Colorbar(fig[1, 2i], hm; label = "deg C")

        # Temperature drift, 1000 m → bottom
        ax = Axis(fig[2, 2i-1]; xlabel = "Time (years)", ylabel = "Depth (m)",
                  title = "$lab: ΔT (1000 m → bottom, deg C)")
        hm = contourf!(ax, t, z, ΔT; levels = temperature_drift_levels,
                       colormap = :balance, extendlow = :auto, extendhigh = :auto)
        ylims!(ax, deep)
        Colorbar(fig[2, 2i], hm; label = "deg C")

        # Salinity drift, upper 1000 m
        ax = Axis(fig[3, 2i-1]; xlabel = "Time (years)", ylabel = "Depth (m)",
                  title = "$lab: ΔS (0 → 1000 m, PSU)")
        hm = contourf!(ax, t, z, ΔS; levels = salinity_drift_levels,
                       colormap = :balance, extendlow = :auto, extendhigh = :auto)
        ylims!(ax, upper)
        Colorbar(fig[3, 2i], hm; label = "PSU")

        # Salinity drift, 1000 m → bottom
        ax = Axis(fig[4, 2i-1]; xlabel = "Time (years)", ylabel = "Depth (m)",
                  title = "$lab: ΔS (1000 m → bottom, PSU)")
        hm = contourf!(ax, t, z, ΔS; levels = salinity_drift_levels,
                       colormap = :balance, extendlow = :auto, extendhigh = :auto)
        ylims!(ax, deep)
        Colorbar(fig[4, 2i], hm; label = "PSU")
    end

    savefig(fig, "fig20_TS_drift_heatmap.png")
end
