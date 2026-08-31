# Why the joint calibration's r is lower: per-cell MSE decomposition
#   MSE = bias² + (σm − r σo)² + (1 − r²) σo²
# and lag-correlations of the hourly series, from the saved snapshots.
using JLD2, CairoMakie, Statistics, Printf

run_dir = "/shared/home/xklee/numericalearth_land_scratch_plans/borneo_calibration"
logK  = jldopen(f -> Dict(k => f[k] for k in keys(f)), joinpath(run_dir, "map_logK_r1_gpu.jld2"))
joint = jldopen(f -> Dict(k => f[k] for k in keys(f)), joinpath(run_dir, "map_joint_r1_gpu.jld2"))
depth = jldopen(f -> Dict(k => f[k] for k in keys(f)), joinpath(run_dir, "map_iterations_r1_gpu_two_sided.jld2"))
θ_obs = logK["θ_obs"]
land = logK["weight"] .> 0
cells = findall(land)
Nh = size(θ_obs, 1)

configs = [("uncalibrated",      logK["snapshots_initial"][:θ], :gray55),
           ("depth-only",        depth["snapshots"][:θ],        :darkorange),
           ("K₀-only, h=0.28 m", logK["snapshots"][:θ],         :steelblue),
           ("joint (h, K₀)",     joint["snapshots"][:θ],        :firebrick)]

sd(x) = std(x; corrected = false)
stats(θm) = map(cells) do c
    m, o = θm[:, c], θ_obs[:, c]
    σm, σo, r = sd(m), sd(o), cor(m, o)
    b = mean(m) - mean(o)
    (; r, ratio = σm / σo, bias2 = b^2, amp2 = (σm - r * σo)^2, decor2 = (1 - r^2) * σo^2)
end

@printf("%-18s %8s %8s | median MSE terms ×10⁶: %8s %8s %8s\n", "config", "med r", "med σ/σ", "bias²", "(σm−rσo)²", "(1−r²)σo²")
for (name, θm, _) in configs
    st = stats(θm)
    @printf("%-18s %8.3f %8.2f | %26.2f %8.2f %8.2f\n", name,
            median([s.r for s in st]), median([s.ratio for s in st]),
            1e6 * median([s.bias2 for s in st]), 1e6 * median([s.amp2 for s in st]), 1e6 * median([s.decor2 for s in st]))
end

lagcor(m, o, ℓ) = ℓ >= 0 ? cor(m[1+ℓ:end], o[1:end-ℓ]) : cor(m[1:end+ℓ], o[1-ℓ:end])
lags = -12:24
lagcurve(θm) = [median([lagcor(θm[:, c], θ_obs[:, c], ℓ) for c in cells]) for ℓ in lags]

fig = Figure(size = (1500, 620), fontsize = 16)
ax1 = Axis(fig[1, 1]; title = "per-cell correlation vs amplitude ratio", xlabel = "r (model, obs)",
           ylabel = "σ_model / σ_obs", yscale = log10)
rr = 0.05:0.01:1.0
lines!(ax1, rr, rr; color = :black, linestyle = :dash, label = "σm = r·σo (L2-optimal amplitude)")
for (name, θm, color) in configs
    st = stats(θm)
    scatter!(ax1, [s.r for s in st], [s.ratio for s in st]; color, markersize = 5, alpha = 0.55, label = name)
end
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1, 2]; title = "median correlation vs lag (model delayed →)", xlabel = "lag (hours)", ylabel = "r")
vlines!(ax2, [0.0]; color = (:black, 0.3))
for (name, θm, color) in configs
    lines!(ax2, lags, lagcurve(θm); color, linewidth = 2.5, label = name)
end
axislegend(ax2; position = :rt)

save(joinpath(run_dir, "r_tradeoff_r1.png"), fig)
@info "saved r_tradeoff_r1.png"
