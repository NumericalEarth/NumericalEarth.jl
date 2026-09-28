# Figure 20: Zonal-mean MLD per case + dBM reference. Top row: per-cell extreme monthly means
# (summer min, winter max). Bottom row: Treguier et al. (2023) hemispheric seasonal means.
function fig20(caches, labels, cases)
    latitude = zonal_latitude_centers()

    fig = Figure(size = (1100 + 200 * length(labels), 1100), fontsize = 14)
    panels = [(1, 1, :zonal_mld_min,    "Zonal-mean MLD (summer minimum)"),
              (1, 2, :zonal_mld_max,    "Zonal-mean MLD (winter maximum)"),
              (2, 1, :zonal_mld_summer, "Zonal-mean MLD (summer: JAS N / JFM S)"),
              (2, 2, :zonal_mld_winter, "Zonal-mean MLD (winter: JFM N / JAS S)")]
    reference_label_index = findfirst(lab -> !isnothing(get_field(caches[lab], :zonal_mld_min_dbm)), labels)

    axes = map(panels) do (row, col, sym, title)
        ax = Axis(fig[row, col]; xlabel = "Latitude", ylabel = "MLD (m)", title)
        for (i, lab) in enumerate(labels)
            lines!(ax, latitude, abs.(get_field(caches[lab], sym));
                   color = case_colors[i], label = lab, linewidth = CASE_LINEWIDTH)
        end
        if !isnothing(reference_label_index)
            reference_cache = caches[labels[reference_label_index]]
            lines!(ax, latitude, abs.(get_field(reference_cache, Symbol(sym, :_dbm)));
                   color = OBS_COLOR, linewidth = OBS_LINEWIDTH, linestyle = OBS_LINESTYLE, label = "dBM")
        end
        ax
    end
    Legend(fig[1:2, 3], first(axes))
    savefig(fig, "fig20_mld_zonal_mean.png")
end
