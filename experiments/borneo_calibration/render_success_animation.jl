# Animation of the joint-calibration result saved by `map_calibrate_joint.jl`: hourly
# ERA5-Land θ beside the uncalibrated and calibrated slabs, their instantaneous mismatch,
# and the domain-median trajectories with a time cursor.
#
#   julia --project=docs render_success_animation.jl

include(joinpath(@__DIR__, "borneo_config.jl"))
using CairoMakie
using Statistics: median, quantile
using Printf
using Dates: Hour

tag = get(ENV, "TAG", "map_joint_r$(refinement)_gpu")
run = jldopen(f -> Dict(k => f[k] for k in keys(f)), "$(tag).jld2")
θ_cal, θ_unc, θ_obs = run["snapshots"][:θ], run["snapshots_initial"][:θ], run["θ_obs"]
rain = run["snapshots"][:rain]
λ, φ = run["longitude"], run["latitude"]
land = run["weight"] .> 0
mask(a) = ifelse.(land, a, NaN)
Nframes = size(θ_obs, 1)
t = 0:Nframes-1

θlim = extrema(filter(isfinite, vcat([vec(mask(x[k, :, :])) for x in (θ_obs, θ_unc, θ_cal), k in (1, Nframes ÷ 2, Nframes)]...)))
mlim = (0, quantile(filter(isfinite, vec(abs.(θ_unc .- θ_obs))), 0.98))
rainlim = (0, max(quantile(filter(isfinite, vec(rain)) .* 3600, 0.99), 0.5))
med(x) = [median(x[k, :, :][land]) for k in 1:Nframes]

n = Observable(1)
fig = Figure(size = (1600, 900), fontsize = 14)
title = @lift @sprintf("Joint (h, K₀) calibration against ERA5-Land, Central Borneo — t = %d h (%s)",
                       $n - 1, string(start_date + Hour($n - 1)))
Label(fig[0, 1:8], title; fontsize = 19)

function panel!(pos, data, ttl, label; colormap, colorrange)
    ax = Axis(fig[pos...]; title = ttl, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return ax
end

panel!((1, 1), @lift(mask(θ_obs[$n, :, :])), "ERA5-Land θ (0–28 cm)", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 3), @lift(mask(θ_unc[$n, :, :])), "slab θ, uncalibrated", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 5), @lift(mask(θ_cal[$n, :, :])), "slab θ, calibrated (h, K₀)", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 7), @lift(mask(rain[$n, :, :]) .* 3600), "ERA5 rain", "mm h⁻¹"; colormap = :dense, colorrange = rainlim)
panel!((2, 1), @lift(mask(abs.(θ_unc[$n, :, :] .- θ_obs[$n, :, :]))), "|mismatch|, uncalibrated", "m³ m⁻³"; colormap = :amp, colorrange = mlim)
panel!((2, 3), @lift(mask(abs.(θ_cal[$n, :, :] .- θ_obs[$n, :, :]))), "|mismatch|, calibrated", "m³ m⁻³"; colormap = :amp, colorrange = mlim)

ax = Axis(fig[2, 5:8]; title = "domain-median θ(t)", xlabel = "hour", ylabel = "θ (m³ m⁻³)")
lines!(ax, t, med(θ_obs); color = :black, linewidth = 2, label = "ERA5-Land")
lines!(ax, t, med(θ_unc); color = :steelblue, linewidth = 2, label = "uncalibrated")
lines!(ax, t, med(θ_cal); color = :firebrick, linewidth = 2, label = "calibrated")
vlines!(ax, @lift([$n - 1.0]); color = :gray, linestyle = :dash)
axislegend(ax; position = :rc)

CairoMakie.record(fig, "$(tag).mp4", 1:Nframes; framerate = 10, compression = 25) do k
    n[] = k
end
@info "saved $(tag).mp4 ($(round(filesize("$(tag).mp4") / 1e6, digits = 1)) MB)"
