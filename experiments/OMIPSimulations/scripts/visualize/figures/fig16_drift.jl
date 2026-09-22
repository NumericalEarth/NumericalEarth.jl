# Figure 15: Global-mean T and S drift time series.
function fig16(caches, labels, cases)
    fig = Figure(size = (600 + 200 * length(labels), 450), fontsize = 14)
    ax_temperature = Axis(fig[1, 1]; xlabel = "Time (years)", ylabel = "ΔT (deg C)",
                          title = "Global-mean temperature drift")
    n_temperature = 0
    for (i, lab) in enumerate(labels)
        temperature = get_field(caches[lab], :global_mean_temperature_timeseries)
        if isnothing(temperature)
            @info "  $lab: no global-mean temperature in its averages stream, skipped"
            continue
        end
        time_in_years = get_field(caches[lab], :time_in_years)
        lines!(ax_temperature, time_in_years, temperature .- temperature[1];
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
        n_temperature += 1
    end
    ax_salinity = Axis(fig[1, 2]; xlabel = "Time (years)", ylabel = "ΔS (PSU)",
                       title = "Global-mean salinity drift")
    n_salinity = 0
    for (i, lab) in enumerate(labels)
        salinity = get_field(caches[lab], :global_mean_salinity_timeseries)
        if isnothing(salinity)
            @info "  $lab: no global-mean salinity in its averages stream, skipped"
            continue
        end
        time_in_years = get_field(caches[lab], :time_in_years)
        lines!(ax_salinity, time_in_years, salinity .- salinity[1];
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
        n_salinity += 1
    end
    Legend(fig[1, 3], n_salinity > n_temperature ? ax_salinity : ax_temperature)
    savefig(fig, "fig16_drift.png")
end
