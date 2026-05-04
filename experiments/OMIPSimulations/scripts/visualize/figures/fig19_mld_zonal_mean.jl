# Figure 19: Zonal-mean MLD (summer min and winter max) per case + dBM reference.
function fig19(caches, labels, cases)
    latitude = zonal_latitude_centers()

    fig = Figure(size = (1100 + 200 * length(labels), 550), fontsize = 14)
    ax_min = Axis(fig[1, 1]; xlabel = "Latitude", ylabel = "MLD (m)",
                  title = "Zonal-mean MLD (summer minimum)")
    ax_max = Axis(fig[1, 2]; xlabel = "Latitude", ylabel = "MLD (m)",
                  title = "Zonal-mean MLD (winter maximum)")
    for (i, lab) in enumerate(labels)
        c = caches[lab]
        lines!(ax_min, latitude, abs.(get_field(c, :zonal_mld_min));
               color = case_colors[i], label = lab, linewidth = 2)
        lines!(ax_max, latitude, abs.(get_field(c, :zonal_mld_max));
               color = case_colors[i], label = lab, linewidth = 2)
    end
    reference_label_index = findfirst(lab -> !isnothing(get_field(caches[lab], :zonal_mld_min_dbm)), labels)
    if !isnothing(reference_label_index)
        reference_cache = caches[labels[reference_label_index]]
        lines!(ax_min, latitude, abs.(get_field(reference_cache, :zonal_mld_min_dbm));
               color = :black, linewidth = 2, linestyle = :dash, label = "dBM")
        lines!(ax_max, latitude, abs.(get_field(reference_cache, :zonal_mld_max_dbm));
               color = :black, linewidth = 2, linestyle = :dash, label = "dBM")
    end
    Legend(fig[1, 3], ax_min)
    savefig(fig, "fig19_mld_zonal_mean.png")
end
