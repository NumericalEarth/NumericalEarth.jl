# Cross-suite comparison of every calibration configuration from the saved 20-day validation runs
# and calibration files: domain-median trajectories, a per-window scorecard, the calibrated
# parameter maps side by side, and the held-out RMS maps.
#
#   julia --project=docs render_suite_comparison.jl   (from the run directory)

using CairoMakie
using JLD2
using Statistics: median, std, cor
using Printf

load(name) = jldopen(f -> Dict(k => f[k] for k in keys(f)), name)
validation(stem) = isfile("validation_of_$(stem).jld2") ? load("validation_of_$(stem).jld2") : nothing   # suites still running are skipped
series(key) = isnothing(V[key]) ? nothing : V[key]["snapshots"][:θ]

V = Dict(
    :free_joint6  => validation("map_joint_r1_gpu"),
    :depth        => validation("map_iterations_r1_gpu_two_sided"),
    :free_k       => validation("map_logK_r1_gpu"),
    :free_joint12 => validation("map_joint_r1_gpu_12d"),
    :vm           => validation("map_variance_matched_r1_gpu"),
    :darcy_k      => validation("map_logK_r1_gpu_darcy_12d"),
    :darcy_joint  => validation("map_joint_r1_gpu_darcy_12d"),
    :darcy_kl     => validation("map_hydrology_K0_exchange_r1_gpu_darcy_12d"),
    :darcy_kln    => validation("map_hydrology_K0_exchange_retention_r1_gpu_darcy_12d"),
    :darcy_mean   => validation("map_logK_r1_gpu_darcy_12d_meanhead"),
    :darcy_fc     => validation("map_hydrology_K0_deephead_r1_gpu_darcy_fc_12d"),
    :darcy_t0     => validation("map_logK_r1_gpu_darcy_12d_initialhead"),
    :darcy_init   => validation("map_hydrology_K0_deephead_r1_gpu_darcy_init_12d"),
    :darcy_s2     => validation("map_logK_r1_gpu_darcy_12d_smooth2d"),
    :darcy_s5     => validation("map_logK_r1_gpu_darcy_12d_smooth5d"),
    :store        => validation("map_logK_r1_gpu_darcy_12d_store"),
    :store_ptf    => validation("map_logK_r1_gpu_darcy_12d_store_ptf"),
    :store_cal3   => validation("map_hydrology_K0_exchange_thickness_r1_gpu_darcy_store_12d"),
    :store_cal4   => validation("map_hydrology_K0_exchange_thickness_deepK0_r1_gpu_darcy_store4_12d"),
    :store_cal4b  => validation("map_hydrology_K0_exchange_thickness_deepK0_r1_gpu_darcy_store4b_12d"),
    :store_cal4dl => validation("map_hydrology_K0_exchange_thickness_deepK0_r1_gpu_darcy_store4dl_12d"),
    :store_wt     => validation("map_hydrology_K0_exchange_thickness_watertable_r1_gpu_darcy_store_wt_12d"),
    :store_wteq   => validation("map_hydrology_K0_exchange_thickness_watertable_r1_gpu_darcy_store_wteq_12d"))
era5 = jldopen(f -> f["data"], "surface_cache/era5_land_r1.jld2")

θ_obs = V[:free_joint6]["θ_obs"]
land = V[:free_joint6]["weight"] .> 0
cells = findall(land)
λ, φ = V[:free_joint6]["longitude"], V[:free_joint6]["latitude"]
Nh = size(θ_obs, 1)
days = (0:Nh-1) ./ 24
windows = ("days 0–6.25" => 1:151, "days 6.25–12.25" => 151:295, "days 12.25–20" => 295:Nh)

# (label, θ series, family, fitting-window end in days)
suites = [
    ("pedotransfer, free drainage",              V[:free_joint6]["snapshots_uncalibrated"][:θ], :free,  0.0),
    ("depth only (6.25 d)",                      series(:depth),                    :free,  6.25),
    ("K₀ only (6.25 d)",                         series(:free_k),                   :free,  6.25),
    ("joint (h, K₀) (6.25 d)",                   series(:free_joint6),              :free,  6.25),
    ("joint (h, K₀) (12.25 d)",                  series(:free_joint12),             :free,  12.25),
    ("joint, variance-matched loss (12.25 d)",   series(:vm),                       :free,  12.25),
    ("pedotransfer, Darcy exchange",             V[:darcy_k]["snapshots_uncalibrated"][:θ],     :darcy, 0.0),
    ("Darcy: K₀ (12.25 d)",                      series(:darcy_k),                  :darcy, 12.25),
    ("Darcy: joint (h, K₀) (12.25 d)",           series(:darcy_joint),              :darcy, 12.25),
    ("Darcy: K₀ + exchange length (12.25 d)",    series(:darcy_kl),                 :darcy, 12.25),
    ("Darcy: K₀ + ℓ + retention n (12.25 d)",    series(:darcy_kln),                :darcy, 12.25),
    ("Darcy: suite-8 K₀, deep head frozen to its mean", series(:darcy_mean),         :fair,  12.25),
    ("Darcy, no deep data: K₀ + calibrated constant head (12.25 d)", series(:darcy_fc), :fair, 12.25),
    ("Darcy: suite-8 K₀, deep head fixed at ERA5-Land's t = 0 value", series(:darcy_t0), :fair, 12.25),
    ("Darcy: K₀ + constant head started from the t = 0 value (12.25 d)", series(:darcy_init), :fair, 12.25),
    ("Darcy: suite-8 K₀, deep head smoothed over 2 d",  series(:darcy_s2),               :darcy, 12.25),
    ("Darcy: suite-8 K₀, deep head smoothed over 5 d",  series(:darcy_s5),               :darcy, 12.25),
    ("deep store: suite-8 K₀, store drains at the slab's K₀",     series(:store),      :store, 12.25),
    ("deep store: suite-8 K₀, store drains at pedotransfer K₀",   series(:store_ptf),  :store, 12.25),
    ("deep store: K₀ + ℓ + hᵈ calibrated (12.25 d)",              series(:store_cal3), :store, 12.25),
    ("deep store: K₀ + ℓ + hᵈ + K₀ᵈ calibrated (12.25 d)",        series(:store_cal4), :store, 12.25),
    ("deep store: K₀ + ℓ + hᵈ + K₀ᵈ, deep layer in the loss",     series(:store_cal4dl), :store, 12.25),
    ("deep store on a water table (2.5 m start): K₀ + ℓ + hᵈ + depth", series(:store_wt), :store, 12.25),
    ("deep store: K₀ + ℓ + hᵈ + K₀ᵈ, 16 iterations (control)",    series(:store_cal4b), :store, 12.25),
    ("deep store on a water table (equilibrium start): K₀ + ℓ + hᵈ + depth", series(:store_wteq), :store, 12.25),
]
filter!(s -> !isnothing(s[2]), suites)

function scores(θm, hours)
    r = Float64[]; σ = Float64[]; mse = Float64[]
    for c in cells
        m, o = θm[hours, c], θ_obs[hours, c]
        push!(mse, sum(abs2, m .- o) / length(hours)); push!(σ, std(m) / std(o))
        std(m) > 0 && push!(r, cor(m, o))
    end
    return (; rms = sqrt(sum(mse) / length(mse)), r = median(r), σ = median(σ))
end
table = [(label, [scores(θm, hours) for (_, hours) in windows]) for (label, θm, _, _) in suites]
@printf("%-44s", "configuration"); foreach(w -> @printf("%30s", first(w)), windows); println()
for (label, s) in table
    @printf("%-44s", label); foreach(x -> @printf("   %.4f · %.2f · %.2f   ", x.rms, x.r, x.σ), s); println()
end
jldsave("suite_scorecard_r1.jld2"; labels = first.(table), windows = first.(windows),
        rms = [table[i][2][w].rms for i in eachindex(table), w in eachindex(windows)],
        r = [table[i][2][w].r for i in eachindex(table), w in eachindex(windows)],
        σ = [table[i][2][w].σ for i in eachindex(table), w in eachindex(windows)])

med(x) = [median(x[k, :, :][land]) for k in 1:Nh]
mask(a) = ifelse.(land, a, NaN)
free_colors  = Makie.wong_colors()[1:6]
darcy_colors = (:gray40, :firebrick, :darkorange, :seagreen, :mediumpurple, :deepskyblue, :goldenrod)
fair_colors  = (:sienna, :navy, :teal, :crimson)
store_colors = (:gray40, :darkorange, :seagreen, :firebrick, :indianred, :navy, :teal, :purple)

# ## Trajectories

fig = Figure(size = (2100, 1900), fontsize = 15)
Label(fig[0, 1:2], "Domain-median soil water, every calibration suite, against ERA5-Land over 20 days"; fontsize = 19)
for (row, family, title, colors) in ((1, :free, "free drainage (the original bottom boundary)", free_colors),
                                     (2, :darcy, "Darcy exchange to ERA5-Land's 28–100 cm head", darcy_colors),
                                     (3, :fair, "Darcy exchange to a constant head (no reanalysis after t = 0)", fair_colors),
                                     (4, :store, "Darcy exchange to a prognostic deep store (no reanalysis after t = 0)", store_colors))
    ax = Axis(fig[row, 1]; title, xlabel = "day", ylabel = "θ (m³ m⁻³)")
    vlines!(ax, [6.25, 12.25]; color = :gray60, linestyle = :dash)
    lines!(ax, days, med(θ_obs); color = :black, linewidth = 2.6, label = "ERA5-Land")
    for (i, (label, θm, fam, _)) in enumerate(filter(s -> s[3] == family, suites))
        occursin("pedotransfer K₀", label) && continue   # the saturating store stays in the scorecard only
        lines!(ax, days, med(θm); color = colors[i], linewidth = 1.8, label)
    end
    Legend(fig[row, 2], ax; labelsize = 12)
end
colsize!(fig.layout, 2, Auto(false))
save("suite_trajectories_r1.png", fig)

# ## The deep store against the reanalysis layer it never sees

store_keys = filter(k -> !isnothing(V[k]), [:store, :store_cal3, :store_cal4, :store_cal4dl, :store_wt, :store_cal4b, :store_wteq])
store_labels = [label for (label, _, fam, _) in suites if fam == :store && !occursin("pedotransfer", label)]
store_color = Dict(:store => 1, :store_cal3 => 3, :store_cal4 => 4, :store_cal4dl => 5, :store_wt => 6, :store_cal4b => 7, :store_wteq => 8)
fig = Figure(size = (1500, 1000), fontsize = 15)
Label(fig[0, 1], "The prognostic deep store (28–100 cm) against ERA5-Land's deep layer, which only set its t = 0 state"; fontsize = 18)
ax1 = Axis(fig[1, 1]; title = "domain-median water content of the deep layer", xlabel = "day", ylabel = "θᵈ (m³ m⁻³)")
vlines!(ax1, [6.25, 12.25]; color = :gray60, linestyle = :dash)
lines!(ax1, days, [median(era5.layer_3[k, :, :][land]) for k in 1:Nh]; color = :black, linewidth = 2.6, label = "ERA5-Land")
for (i, key) in enumerate(store_keys)
    lines!(ax1, days, med(V[key]["snapshots"][:θᵈ]); color = store_colors[store_color[key]], linewidth = 1.8, label = store_labels[i])
end
ax2 = Axis(fig[2, 1]; title = "and the slab above it (0–28 cm)", xlabel = "day", ylabel = "θ (m³ m⁻³)")
vlines!(ax2, [6.25, 12.25]; color = :gray60, linestyle = :dash)
lines!(ax2, days, med(θ_obs); color = :black, linewidth = 2.6, label = "ERA5-Land")
for (i, key) in enumerate(store_keys)
    lines!(ax2, days, med(V[key]["snapshots"][:θ]); color = store_colors[store_color[key]], linewidth = 1.8, label = store_labels[i])
end
Legend(fig[3, 1], ax2; orientation = :horizontal, nbanks = 2, labelsize = 13, tellwidth = false)
save("suite_store_r1.png", fig)

# ## Scorecard

fig = Figure(size = (1800, 950), fontsize = 14)
Label(fig[0, 1:3], "Per-window scores of every suite (RMS in m³ m⁻³, median per-cell correlation, median amplitude ratio)"; fontsize = 18)
labels = first.(table)
xs = 1:length(labels)
window_colors = (:steelblue, :darkorange, :firebrick)
for (col, (metric, name, ref)) in enumerate(((:rms, "RMS", nothing), (:r, "correlation r", nothing), (:σ, "σ_model / σ_obs", 1.0)))
    ax = Axis(fig[1, col]; title = name, xticks = (xs, labels), xticklabelrotation = π / 3, xticklabelsize = 11,
              yscale = metric == :rms ? log10 : identity)
    isnothing(ref) || hlines!(ax, [ref]; color = :gray50, linestyle = :dash)
    for (w, (wname, _)) in enumerate(windows)
        ys = [getfield(s[w], metric) for (_, s) in table]
        scatterlines!(ax, xs, ys; color = window_colors[w], markersize = 10, label = wname)
    end
    col == 1 && axislegend(ax; position = :rt, labelsize = 11)
end
save("suite_scorecard_r1.png", fig)

# ## Parameter maps

C = Dict(:free_k => load("map_logK_r1_gpu.jld2"), :free_joint6 => load("map_joint_r1_gpu.jld2"),
         :free_joint12 => load("map_joint_r1_gpu_12d.jld2"), :vm => load("map_variance_matched_r1_gpu.jld2"),
         :darcy_k => load("map_logK_r1_gpu_darcy_12d.jld2"), :darcy_joint => load("map_joint_r1_gpu_darcy_12d.jld2"),
         :darcy_kl => load("map_hydrology_K0_exchange_r1_gpu_darcy_12d.jld2"),
         :darcy_kln => load("map_hydrology_K0_exchange_retention_r1_gpu_darcy_12d.jld2"),
         :darcy_fc => load("map_hydrology_K0_deephead_r1_gpu_darcy_fc_12d.jld2"),
         :darcy_init => load("map_hydrology_K0_deephead_r1_gpu_darcy_init_12d.jld2"),
         :store_cal4 => load("map_hydrology_K0_exchange_thickness_deepK0_r1_gpu_darcy_store4_12d.jld2"),
         :depth => load("map_iterations_r1_gpu_two_sided.jld2"))
q₀ = C[:free_k]["q_pedotransfer"]
ψ_initial = exp.(C[:darcy_init]["history"][1][3][:deephead])

fig = Figure(size = (1900, 2150), fontsize = 14)
Label(fig[0, 1:8], "Calibrated parameter fields across the suites"; fontsize = 19)
function panel!(pos, data, title, label; colormap, colorrange, scale = identity)
    ax = Axis(fig[pos...]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, data; colormap, colorrange, colorscale = scale)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return ax
end
qlim = (0, 2.6)
panel!((1, 1), mask(C[:free_k]["q"] .- q₀), "Δlog₁₀K₀ — K₀ only, free drainage", "decades"; colormap = :viridis, colorrange = qlim)
panel!((1, 3), mask(C[:free_joint12]["q"] .- q₀), "Δlog₁₀K₀ — joint, free drainage (12.25 d)", "decades"; colormap = :viridis, colorrange = qlim)
panel!((1, 5), mask(C[:darcy_k]["q"] .- q₀), "Δlog₁₀K₀ — Darcy: K₀", "decades"; colormap = :viridis, colorrange = qlim)
panel!((1, 7), mask(C[:darcy_kl]["q"] .- q₀), "Δlog₁₀K₀ — Darcy: K₀ + ℓ", "decades"; colormap = :viridis, colorrange = qlim)
hlim = (0.02, 5.0)
panel!((2, 1), mask(C[:depth]["depths"]), "slab depth — depth only (6.25 d)", "m"; colormap = :viridis, colorrange = hlim, scale = log10)
panel!((2, 3), mask(C[:free_joint12]["depths"]), "slab depth — joint, free drainage (12.25 d)", "m"; colormap = :viridis, colorrange = hlim, scale = log10)
panel!((2, 5), mask(C[:vm]["depths"]), "slab depth — variance-matched loss", "m"; colormap = :viridis, colorrange = hlim, scale = log10)
panel!((2, 7), mask(C[:darcy_joint]["depths"]), "slab depth — Darcy: joint (h, K₀)", "m"; colormap = :viridis, colorrange = hlim, scale = log10)
panel!((3, 1), mask(exp.(C[:darcy_kl]["log_exchange_length"])), "exchange length ℓ — Darcy: K₀ + ℓ", "m"; colormap = :viridis, colorrange = (0.1, 1.4))
panel!((3, 3), mask(1 .+ exp.(C[:darcy_kln]["log_n_minus_1"])), "retention exponent n — Darcy: K₀ + ℓ + n", "–"; colormap = :viridis, colorrange = (1.10, 1.20))
panel!((3, 5), mask(q₀), "pedotransfer log₁₀K₀ (prior)", "log₁₀ m s⁻¹"; colormap = :viridis, colorrange = extrema(q₀[land]))
panel!((3, 7), mask(1 .+ exp.(C[:darcy_kln]["ν_pedotransfer"])), "pedotransfer n (prior)", "–"; colormap = :viridis, colorrange = (1.10, 1.20))
ψlim = (0.3, 10.0)
panel!((4, 1), mask(ψ_initial), "deep suction ψᵈ — ERA5-Land at t = 0 (the start)", "m"; colormap = :viridis, colorrange = ψlim, scale = log10)
panel!((4, 3), mask(exp.(C[:darcy_init]["log_deep_suction"])), "deep suction ψᵈ — calibrated from the t = 0 value", "m"; colormap = :viridis, colorrange = ψlim, scale = log10)
panel!((4, 5), mask(exp.(C[:darcy_fc]["log_deep_suction"])), "deep suction ψᵈ — calibrated from 1 m", "m"; colormap = :viridis, colorrange = ψlim, scale = log10)
panel!((4, 7), mask(C[:darcy_init]["q"] .- q₀), "Δlog₁₀K₀ — K₀ + head from the t = 0 value", "decades"; colormap = :viridis, colorrange = qlim)
panel!((5, 1), mask(exp.(C[:store_cal4]["log_thickness"])), "store thickness hᵈ — deep store, 4 fields", "m"; colormap = :viridis, colorrange = (0.1, 3.0), scale = log10)
panel!((5, 3), mask(C[:store_cal4]["q_deep"] .- q₀), "Δlog₁₀K₀ᵈ (store drainage) — deep store, 4 fields", "decades"; colormap = :viridis, colorrange = (-2, 2.6))
panel!((5, 5), mask(exp.(C[:store_cal4]["log_exchange_length"])), "exchange length ℓ — deep store, 4 fields", "m"; colormap = :viridis, colorrange = (0.1, 1.4))
panel!((5, 7), mask(C[:store_cal4]["q"] .- q₀), "Δlog₁₀K₀ (slab) — deep store, 4 fields", "decades"; colormap = :viridis, colorrange = qlim)
save("suite_parameters_r1.png", fig)

# ## Held-out RMS maps

held = last(windows)[2]
cell_rms(θm) = [sqrt(sum(abs2, θm[held, i, j] .- θ_obs[held, i, j]) / length(held)) for i in axes(θm, 2), j in axes(θm, 3)]
fig = Figure(size = (1900, 1000), fontsize = 14)
Label(fig[0, 1:8], "Held-out RMS (days 12.25–20, unseen by every calibration), shared scale"; fontsize = 19)
picks = filter(≤(length(suites)), [1, 5, 8, 10, 13, 15, 23, 25])
for (k, idx) in enumerate(picks)
    label, θm, _, _ = suites[idx]
    row, col = divrem(k - 1, 4)
    ax = Axis(fig[row + 1, 2col + 1]; title = label, titlesize = 12, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, mask(cell_rms(θm)); colormap = :amp, colorrange = (0, 0.05))
    Colorbar(fig[row + 1, 2col + 2], hm; label = "m³ m⁻³")
end
save("suite_heldout_rms_r1.png", fig)
@info "saved suite_trajectories_r1.png, suite_scorecard_r1.png, suite_parameters_r1.png, suite_heldout_rms_r1.png"
