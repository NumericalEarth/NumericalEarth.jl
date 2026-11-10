# Figure 21: Strait transports (Bering, Drake, ITF) annual means per case.

# Observed transports compiled by Iovino et al. (2023, GMD 16, 6127, Fig. 16), in the sign convention of
# `strait_transports` (Bering northward, Drake eastward, ITF northward, so negative into the Indian Ocean).
const BERING_OBSERVED = (mean = 1.0, low = 0.8, high = 1.2)      # Woodgate and Peralta-Ferriz (2021), 1990–2019
const ITF_OBSERVED    = (mean = -15.0, low = -18.7, high = -10.7) # INSTANT 2004–2006, Sprintall et al. (2009)
const DRAKE_OBSERVED  = [(127.7,  8.1, :diamond),    # Chidichimo et al. (2014)
                         (129.0,  6.0, :dtriangle),  # Lumpkin and Speer (2007)
                         (134.0, 13.0, :circle),     # Whitworth and Peterson (1985)
                         (135.3, 10.2, :utriangle),  # Xu et al. (2020)
                         (141.0,  2.7, :rect)]       # DRAKE, Koenig et al. (2014)

function fig22(caches, labels, cases)
    # Bin a time series sampled at `t_seconds` into yearly means starting at year 0.
    function annual_means(t_seconds, values)
        years_full = floor.(Int, t_seconds ./ (365.25 * 86400))
        unique_years = sort(unique(years_full))
        centers = Float64[]
        means   = Float64[]
        for y in unique_years
            mask = years_full .== y
            any(mask) || continue
            push!(centers, y + 0.5)
            push!(means,   mean(values[mask]))
        end
        return centers, means
    end

    have_any = false
    for lab in labels
        st = get_field(caches[lab], :strait_transports)
        isnothing(st) || (have_any = true; break)
    end
    have_any || return

    fig = Figure(size = (1000 + 200 * length(labels), 500), fontsize = 14)
    ax_b = Axis(fig[1, 1]; xlabel = "Time (years)", ylabel = "Transport (Sv)", title = "Bering Strait")
    ax_d = Axis(fig[1, 2]; xlabel = "Time (years)", ylabel = "Transport (Sv)", title = "Drake Passage")
    ax_i = Axis(fig[1, 3]; xlabel = "Time (years)", ylabel = "Transport (Sv)", title = "Indonesian Throughflow")
    for (ax, observed) in ((ax_b, BERING_OBSERVED), (ax_i, ITF_OBSERVED))
        hspan!(ax, observed.low, observed.high; color = (OBS_COLOR, 0.2))
        hlines!(ax, observed.mean; color = OBS_COLOR, linestyle = OBS_LINESTYLE, linewidth = OBS_LINEWIDTH,
                label = "observations")
    end
    final_year = 0.0
    for (i, lab) in enumerate(labels)
        st = get_field(caches[lab], :strait_transports)
        isnothing(st) && continue
        final_year = max(final_year, last(st.time) / (365.25 * 86400))
        tb, yb = annual_means(st.time, st.bering)
        td, yd = annual_means(st.time, st.drake)
        ti, yi = annual_means(st.time, st.itf)
        lines!(ax_b, tb, yb; color = case_colors[i], label = lab, linewidth = CASE_LINEWIDTH)
        lines!(ax_d, td, yd; color = case_colors[i], label = lab, linewidth = CASE_LINEWIDTH)
        lines!(ax_i, ti, yi; color = case_colors[i], label = lab, linewidth = CASE_LINEWIDTH)
    end
    observed_years = final_year .* (1.04 .+ 0.025 .* (0:length(DRAKE_OBSERVED)-1))
    errorbars!(ax_d, observed_years, first.(DRAKE_OBSERVED), getindex.(DRAKE_OBSERVED, 2);
               color = OBS_COLOR, whiskerwidth = 6)
    scatter!(ax_d, observed_years, first.(DRAKE_OBSERVED); marker = last.(DRAKE_OBSERVED),
             color = OBS_COLOR, markersize = 11)
    Legend(fig[1, 4], ax_b)
    savefig(fig, "fig22_strait_transports.png")
end
