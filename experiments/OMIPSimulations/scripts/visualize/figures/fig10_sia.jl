# Figure 10: Sea-ice area climatology (model + NSIDC), Arctic and Antarctic.
function fig10(caches, labels, cases)
    month_names = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    m2_to_million_km2 = 1e-12
    fig = Figure(size = (600 + 200 * length(labels), 500), fontsize = 14)
    ax_arctic = Axis(fig[1, 1]; xlabel = "Month", ylabel = "SIA (Million km²)",
                     title = "Arctic SIA Climatology", xticks = (1:12, month_names))
    lines!(ax_arctic, 1:12, nsidc_arctic().area_monthly;
           color = :black, linewidth = 2, label = "NSIDC")
    for (i, lab) in enumerate(labels)
        lines!(ax_arctic, 1:12,
               get_field(caches[lab], :sea_ice_diagnostics).arctic_area_monthly .* m2_to_million_km2;
               color = case_colors[i], label = lab)
    end
    ax_antarctic = Axis(fig[1, 2]; xlabel = "Month", ylabel = "SIA (Million km²)",
                        title = "Antarctic SIA Climatology", xticks = (1:12, month_names))
    lines!(ax_antarctic, 1:12, nsidc_antarctic().area_monthly;
           color = :black, linewidth = 2, label = "NSIDC")
    for (i, lab) in enumerate(labels)
        lines!(ax_antarctic, 1:12,
               get_field(caches[lab], :sea_ice_diagnostics).antarctic_area_monthly .* m2_to_million_km2;
               color = case_colors[i], label = lab)
    end
    Legend(fig[1, 3], ax_arctic)
    savefig(fig, "fig10_sia.png")
end
