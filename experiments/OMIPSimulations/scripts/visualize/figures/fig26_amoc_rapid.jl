# Figure 26: AMOC at 26.5°N — comparison with the RAPID-MOCHA array.
#
# Left  : time-mean streamfunction profile ψ(z) at the j-row closest to
#         26.5°N. ψ on the x-axis (Sv) and depth on the y-axis with the
#         ocean bottom at the bottom of the plot (ylims = (-5500, 0)).
#         Model lines are one per case; the dashed grey line is the
#         RAPID daily climatology over 2004–present (mean profile,
#         bracketed by two thin ±1σ lines).
#
# Right : AMOC index time series — ψ_max at 26.5°N averaged over 6-month
#         bins per case, against the RAPID index `moc_mar_hc10` averaged
#         over the same bins (band: ±1σ of its monthly means). The model
#         time axis is decimal calendar year assuming a 1958-01-01
#         simulation start (the OMIP/JRA55-do convention used by
#         `compute_monthly_means`).
function six_month_means(year, values)
    bins = floor.(2 .* year) ./ 2
    centers = sort!(unique(bins))
    groups = [filter(isfinite, values[bins .== b]) for b in centers]
    keep = .!isempty.(groups)
    return (year = centers[keep] .+ 0.25, mean = mean.(groups[keep]),
            std = [length(g) > 1 ? std(g) : 0.0 for g in groups[keep]])
end

function fig26(caches, labels, cases)
    rapid_profile = rapid_amoc_profile()
    rapid_index   = rapid_amoc_timeseries()

    fig = Figure(size = (1100, 500), fontsize = 14)

    j_lat = get_field(caches[first(labels)], :amoc_26n_j)
    lat   = let lats = get_field(caches[first(labels)], :amoc_latitudes)
        ((lats[1:end-1] .+ lats[2:end]) ./ 2)[j_lat]
    end

    ax_profile = Axis(fig[1, 1];
        xlabel = "ψ (Sv)", ylabel = "Depth (m)",
        title  = "AMOC ψ(z) at $(round(lat; digits = 1))°N")

    if !isnothing(rapid_profile)
        z_rapid = -rapid_profile.depth
        ψ̄       = rapid_profile.psi_mean
        σ       = rapid_profile.psi_std
        good    = findall(isfinite, ψ̄)
        zg      = z_rapid[good]
        lo      = Point2f.(ψ̄[good] .- σ[good], zg)
        hi      = Point2f.(ψ̄[good] .+ σ[good], zg)
        band!(ax_profile, lo, hi; color = (OBS_COLOR, 0.25))
        lines!(ax_profile, ψ̄[good], zg;
               color = OBS_COLOR, linestyle = OBS_LINESTYLE,
               linewidth = OBS_LINEWIDTH, label = "RAPID (mean ± 1σ)")
    end
    for (i, lab) in enumerate(labels)
        c = caches[lab]
        lines!(ax_profile, get_field(c, :amoc_26n_profile), get_field(c, :depth);
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
    end
    ylims!(ax_profile, (-5500, 0))

    ax_index = Axis(fig[1, 2];
        xlabel = "Year", ylabel = "ψ_max (Sv)",
        title  = "AMOC index at $(round(lat; digits = 1))°N, 6-month means")

    if !isnothing(rapid_index)
        r = six_month_means(rapid_index.year, rapid_index.psi_max)
        band!(ax_index, r.year, r.mean .- r.std, r.mean .+ r.std; color = (OBS_COLOR, 0.25))
        lines!(ax_index, r.year, r.mean;
               color = OBS_COLOR, linestyle = OBS_LINESTYLE,
               linewidth = OBS_LINEWIDTH, label = "RAPID")
    end
    for (i, lab) in enumerate(labels)
        ts = get_field(caches[lab], :amoc_max_timeseries)
        m = six_month_means(ts.year, ts.psi_max)
        lines!(ax_index, m.year, m.mean;
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
    end

    Legend(fig[1, 3], ax_profile)
    savefig(fig, "fig26_amoc_rapid.png")
end
