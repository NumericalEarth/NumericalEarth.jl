# Figure 10: Sea-ice area climatology (model + NSIDC), Arctic and Antarctic.
function fig10(caches, labels, cases)
    month_names = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    m2_to_million_km2 = 1e-12
    arctic_obs    = nsidc_arctic()
    antarctic_obs = nsidc_antarctic()

    fig = Figure(size = (600 + 200 * length(labels), 500), fontsize = 14)
    ax_arctic = Axis(fig[1, 1]; xlabel = "Month", ylabel = "SIA (Million km²)",
                     title = "Arctic SIA Climatology", xticks = (1:12, month_names))
    isnothing(arctic_obs) || lines!(ax_arctic, 1:12, arctic_obs.area_monthly;
        color = OBS_COLOR, linewidth = OBS_LINEWIDTH, linestyle = OBS_LINESTYLE, label = "NSIDC")
    for (i, lab) in enumerate(labels)
        lines!(ax_arctic, 1:12,
               get_field(caches[lab], :sea_ice_diagnostics).arctic_area_monthly .* m2_to_million_km2;
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
    end
    ax_antarctic = Axis(fig[1, 2]; xlabel = "Month", ylabel = "SIA (Million km²)",
                        title = "Antarctic SIA Climatology", xticks = (1:12, month_names))
    isnothing(antarctic_obs) || lines!(ax_antarctic, 1:12, antarctic_obs.area_monthly;
        color = OBS_COLOR, linewidth = OBS_LINEWIDTH, linestyle = OBS_LINESTYLE, label = "NSIDC")
    for (i, lab) in enumerate(labels)
        lines!(ax_antarctic, 1:12,
               get_field(caches[lab], :sea_ice_diagnostics).antarctic_area_monthly .* m2_to_million_km2;
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
    end
    Legend(fig[1, 3], ax_arctic)
    savefig(fig, "fig10_sia.png")
end
