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
validation(stem) = load("validation_of_$(stem).jld2")

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
    :darcy_fc     => validation("map_hydrology_K0_deephead_r1_gpu_darcy_fc_12d"))

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
    ("depth only (6.25 d)",                      V[:depth]["snapshots"][:θ],                    :free,  6.25),
    ("K₀ only (6.25 d)",                         V[:free_k]["snapshots"][:θ],                   :free,  6.25),
    ("joint (h, K₀) (6.25 d)",                   V[:free_joint6]["snapshots"][:θ],              :free,  6.25),
    ("joint (h, K₀) (12.25 d)",                  V[:free_joint12]["snapshots"][:θ],             :free,  12.25),
    ("joint, variance-matched loss (12.25 d)",   V[:vm]["snapshots"][:θ],                       :free,  12.25),
    ("pedotransfer, Darcy exchange",             V[:darcy_k]["snapshots_uncalibrated"][:θ],     :darcy, 0.0),
    ("Darcy: K₀ (12.25 d)",                      V[:darcy_k]["snapshots"][:θ],                  :darcy, 12.25),
    ("Darcy: joint (h, K₀) (12.25 d)",           V[:darcy_joint]["snapshots"][:θ],              :darcy, 12.25),
    ("Darcy: K₀ + exchange length (12.25 d)",    V[:darcy_kl]["snapshots"][:θ],                 :darcy, 12.25),
    ("Darcy: K₀ + ℓ + retention n (12.25 d)",    V[:darcy_kln]["snapshots"][:θ],                :darcy, 12.25),
    ("Darcy: suite-8 K₀, deep head frozen to its mean", V[:darcy_mean]["snapshots"][:θ],         :fair,  12.25),
    ("Darcy, no deep data: K₀ + calibrated constant head (12.25 d)", V[:darcy_fc]["snapshots"][:θ], :fair, 12.25),
]

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
darcy_colors = (:gray40, :firebrick, :darkorange, :seagreen, :mediumpurple)
fair_colors  = (:sienna, :navy)

# ## Trajectories

fig = Figure(size = (1700, 1450), fontsize = 15)
Label(fig[0, 1], "Domain-median soil water, every calibration suite, against ERA5-Land over 20 days"; fontsize = 19)
for (row, family, title, colors) in ((1, :free, "free drainage (the original bottom boundary)", free_colors),
                                     (2, :darcy, "Darcy exchange to ERA5-Land's 28–100 cm head", darcy_colors),
                                     (3, :fair, "Darcy exchange without the reanalysis deep layer", fair_colors))
    ax = Axis(fig[row, 1]; title, xlabel = "day", ylabel = "θ (m³ m⁻³)")
    vlines!(ax, [6.25, 12.25]; color = :gray60, linestyle = :dash)
    lines!(ax, days, med(θ_obs); color = :black, linewidth = 2.6, label = "ERA5-Land")
    for (i, (label, θm, fam, _)) in enumerate(filter(s -> s[3] == family, suites))
        lines!(ax, days, med(θm); color = colors[i], linewidth = 1.8, label)
    end
    axislegend(ax; position = :rt, nbanks = 2, labelsize = 12)
end
save("suite_trajectories_r1.png", fig)

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
         :depth => load("map_iterations_r1_gpu_two_sided.jld2"))
q₀ = C[:free_k]["q_pedotransfer"]

fig = Figure(size = (1900, 1350), fontsize = 14)
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
save("suite_parameters_r1.png", fig)

# ## Held-out RMS maps

held = last(windows)[2]
cell_rms(θm) = [sqrt(sum(abs2, θm[held, i, j] .- θ_obs[held, i, j]) / length(held)) for i in axes(θm, 2), j in axes(θm, 3)]
fig = Figure(size = (1900, 1000), fontsize = 14)
Label(fig[0, 1:8], "Held-out RMS (days 12.25–20, unseen by every calibration), shared scale"; fontsize = 19)
picks = [1, 3, 5, 7, 8, 10, 12, 13]
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
