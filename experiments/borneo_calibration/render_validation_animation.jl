# The 20-day comparison of the two joint calibrations — fitted on the wet first 6.25 days
# and on the first 12.25 days (which include a dry-down) — from the saved validation runs:
# hourly maps of ERA5-Land, the uncalibrated slab and both calibrated slabs, the two
# mismatch maps, and the domain-median trajectories with the fitting windows marked. Also
# prints the scores of every configuration on identical windows.
#
#   julia --project=plotting_env render_validation_animation.jl

using CairoMakie
using JLD2
using Statistics: median, std, cor, quantile
using Printf
using Dates: DateTime, Hour

start_date = DateTime(2020, 4, 1)
short = jldopen(f -> Dict(k => f[k] for k in keys(f)), "validation_of_map_joint_r1_gpu.jld2")
long  = jldopen(f -> Dict(k => f[k] for k in keys(f)), "validation_of_map_joint_r1_gpu_12d.jld2")
θ_obs = short["θ_obs"]
θ_unc = short["snapshots_uncalibrated"][:θ]
θ_6d, θ_12d = short["snapshots"][:θ], long["snapshots"][:θ]
rain = short["snapshots"][:rain]
λ, φ = short["longitude"], short["latitude"]
land = short["weight"] .> 0
cells = findall(land)
mask(a) = ifelse.(land, a, NaN)
Nh = size(θ_obs, 1)
days = (0:Nh-1) ./ 24
split_6d, split_12d = short["calibration_steps"] ÷ 6 + 1, long["calibration_steps"] ÷ 6 + 1

# ## Scores on identical windows

windows = ("days 0–6.25" => 1:split_6d, "days 6.25–12.25" => split_6d:split_12d, "days 12.25–20" => split_12d:Nh)
configs = ("uncalibrated" => θ_unc, "calibrated on 6.25 d" => θ_6d, "calibrated on 12.25 d" => θ_12d)
function scores(θm, hours)
    r = Float64[]; σ = Float64[]; mse = Float64[]
    for c in cells
        m, o = θm[hours, c], θ_obs[hours, c]
        push!(mse, sum(abs2, m .- o) / length(hours)); push!(σ, std(m) / std(o))
        std(m) > 0 && push!(r, cor(m, o))
    end
    return (; rms = sqrt(sum(mse) / length(mse)), r = median(r), σ = median(σ))
end
@printf("%-24s", "")
foreach(w -> @printf("%34s", first(w)), windows); println()
for (name, θm) in configs
    @printf("%-24s", name)
    for (_, hours) in windows
        s = scores(θm, hours)
        @printf("   RMS %.4f  r %.2f  σ %.2f    ", s.rms, s.r, s.σ)
    end
    println()
end

# ## Animation

θlim = extrema(filter(isfinite, vcat([vec(mask(x[k, :, :])) for x in (θ_obs, θ_unc, θ_6d, θ_12d), k in 1:24:Nh]...)))
mlim = (0, quantile(filter(isfinite, vec(abs.(θ_unc .- θ_obs))), 0.9))
rainlim = (0, max(quantile(filter(isfinite, vec(rain)) .* 3600, 0.99), 0.5))
med(x) = [median(x[k, :, :][land]) for k in 1:Nh]

n = Observable(1)
fig = Figure(size = (1800, 950), fontsize = 14)
title = @lift @sprintf("Central Borneo, calibrated slab land vs ERA5-Land — day %.1f (%s)", ($n - 1) / 24, string(start_date + Hour($n - 1)))
Label(fig[0, 1:10], title; fontsize = 19)

function panel!(pos, data, ttl, label; colormap, colorrange)
    ax = Axis(fig[pos...]; title = ttl, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return ax
end
panel!((1, 1), @lift(mask(θ_obs[$n, :, :])), "ERA5-Land θ (0–28 cm)", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 3), @lift(mask(θ_unc[$n, :, :])), "uncalibrated", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 5), @lift(mask(θ_6d[$n, :, :])), "calibrated on days 0–6.25", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 7), @lift(mask(θ_12d[$n, :, :])), "calibrated on days 0–12.25", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 9), @lift(mask(rain[$n, :, :]) .* 3600), "ERA5 rain", "mm h⁻¹"; colormap = :dense, colorrange = rainlim)
panel!((2, 1), @lift(mask(abs.(θ_6d[$n, :, :] .- θ_obs[$n, :, :]))), "|mismatch|, 6.25-day calibration", "m³ m⁻³"; colormap = :amp, colorrange = mlim)
panel!((2, 3), @lift(mask(abs.(θ_12d[$n, :, :] .- θ_obs[$n, :, :]))), "|mismatch|, 12.25-day calibration", "m³ m⁻³"; colormap = :amp, colorrange = mlim)

ax = Axis(fig[2, 5:10]; title = "domain-median θ(t)", xlabel = "day", ylabel = "θ (m³ m⁻³)")
vspan!(ax, 0, (split_6d - 1) / 24; color = (:firebrick, 0.07))
vspan!(ax, 0, (split_12d - 1) / 24; color = (:darkorange, 0.07))
lines!(ax, days, med(θ_obs); color = :black, linewidth = 2, label = "ERA5-Land")
lines!(ax, days, med(θ_unc); color = :steelblue, linewidth = 2, label = "uncalibrated")
lines!(ax, days, med(θ_6d); color = :firebrick, linewidth = 2, label = "calibrated on days 0–6.25")
lines!(ax, days, med(θ_12d); color = :darkorange, linewidth = 2, label = "calibrated on days 0–12.25")
vlines!(ax, [(split_6d - 1) / 24, (split_12d - 1) / 24]; color = :gray40, linestyle = :dash)
vlines!(ax, @lift([($n - 1) / 24]); color = :gray20)
axislegend(ax; position = :rt)

stride = parse(Int, get(ENV, "FRAME_STRIDE", "1"))   # hours per frame; 2 keeps the file small enough to embed
name = stride == 1 ? "validation_comparison_r1.mp4" : "validation_comparison_r1_$(stride)h.mp4"
CairoMakie.record(fig, name, 1:stride:Nh; framerate = 12 ÷ stride, compression = 28) do k
    n[] = k
end
@info "saved $name ($(round(filesize(name) / 1e6, digits = 1)) MB)"
