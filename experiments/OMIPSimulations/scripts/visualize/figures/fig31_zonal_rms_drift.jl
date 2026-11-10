# Figure 31: zonal-mean RMS deviation from WOA, and the part of the error that cancels within a
# zonal ring. Companion to fig19, which plots the same comparison with signed biases.
#
# fig19 averages T − WOA around a latitude circle, so a warm western boundary and a cold interior at
# the same latitude and depth cancel and the ring reads as unbiased. Squaring before the average keeps
# them: `sqrt(<(T − Tᵂᴼᴬ)²>)` is the error the model actually carries at that latitude.
#
# The bottom rows are the difference between the two, `sqrt(<Δ²>) − |<Δ>|`, which is the zonal standard
# deviation of the bias and is ≥ 0 by construction. It is large exactly where fig19 looks clean because
# the errors offset — the compensating errors this figure exists to find. Read it against fig19: a band
# with a small signed bias and a large cancellation is a model that is wrong in both directions at once,
# not a model that is right.
function fig31(caches, labels, cases)
    latitude = zonal_latitude_centers()

    # RMS is non-negative, so a sequential map; the signed panels in fig19 keep `:balance`.
    temperature_rms_levels = range(0, 3; length = 13)
    salinity_rms_levels    = range(0, 0.75; length = 13)

    fig = Figure(size = (600 * length(labels), 1200), fontsize = 14)
    for (i, lab) in enumerate(labels)
        c = caches[lab]
        depth = get_field(c, :depth)

        ax = Axis(fig[1, 2i-1]; xlabel = "Latitude", ylabel = "Depth (m)",
                  title = "$lab: RMS T − WOA")
        hm = contourf!(ax, latitude, depth, get_field(c, :zonal_temperature_rms);
                       levels = temperature_rms_levels, colormap = :thermal,
                       extendlow = :auto, extendhigh = :auto)
        Colorbar(fig[1, 2i], hm; label = "deg C"); ylims!(ax, (-5500, 0))

        ax = Axis(fig[2, 2i-1]; xlabel = "Latitude", ylabel = "Depth (m)",
                  title = "$lab: RMS S − WOA")
        hm = contourf!(ax, latitude, depth, get_field(c, :zonal_salinity_rms);
                       levels = salinity_rms_levels, colormap = :haline,
                       extendlow = :auto, extendhigh = :auto)
        Colorbar(fig[2, 2i], hm; label = "PSU"); ylims!(ax, (-5500, 0))

        ax = Axis(fig[3, 2i-1]; xlabel = "Latitude", ylabel = "Depth (m)",
                  title = "$lab: T cancelled in the zonal mean")
        hm = contourf!(ax, latitude, depth, get_field(c, :zonal_temperature_cancellation);
                       levels = temperature_rms_levels, colormap = :thermal,
                       extendlow = :auto, extendhigh = :auto)
        Colorbar(fig[3, 2i], hm; label = "deg C"); ylims!(ax, (-5500, 0))

        ax = Axis(fig[4, 2i-1]; xlabel = "Latitude", ylabel = "Depth (m)",
                  title = "$lab: S cancelled in the zonal mean")
        hm = contourf!(ax, latitude, depth, get_field(c, :zonal_salinity_cancellation);
                       levels = salinity_rms_levels, colormap = :haline,
                       extendlow = :auto, extendhigh = :auto)
        Colorbar(fig[4, 2i], hm; label = "PSU"); ylims!(ax, (-5500, 0))
    end
    savefig(fig, "fig31_zonal_rms_drift.png")
end
