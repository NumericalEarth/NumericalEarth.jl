# Animation of the eager forward run saved by `map_calibration.jl`: slab soil water against
# ERA5-Land hour by hour, with the turbulent fluxes and the land surface temperature.
#
#   TAG=map_calibration_r1_cpu julia --project=docs render_map_animation.jl

include(joinpath(@__DIR__, "borneo_config.jl"))
using CairoMakie
using Statistics: mean, quantile
using Printf

tag = get(ENV, "TAG", "map_calibration_r$(refinement)_cpu")
run = jldopen(f -> Dict(k => f[k] for k in keys(f)), "$(tag).jld2")
era5_land = load_cache("era5_land")
snapshots = run["snapshots"]
λ, φ = run["longitude"], run["latitude"]
land = run["weight"] .> 0
mask(a) = ifelse.(land, a, NaN)
Nframes = size(snapshots[:θ], 1)

n = Observable(1)
frame(name) = @lift mask(snapshots[name][$n, :, :])
θ_obs = @lift mask(era5_land_soil_water(era5_land, $n))
θlim = extrema(filter(isfinite, mask(snapshots[:θ][end, :, :])))
θlim = (min(θlim[1], minimum(filter(isfinite, mask(era5_land_soil_water(era5_land, Nframes))))), max(θlim[2], maximum(filter(isfinite, mask(era5_land_soil_water(era5_land, Nframes))))))
LElim = (0, max(quantile(filter(isfinite, vec(snapshots[:LE])), 0.99), 10))
Hlim = (-100, max(quantile(filter(isfinite, vec(snapshots[:H])), 0.99), 10))
Tlim = extrema(filter(isfinite, vec(snapshots[:LST])))
rainlim = (0, max(quantile(filter(isfinite, vec(snapshots[:rain])) .* 3600, 0.99), 0.5))

fig = Figure(size = (1900, 1000), fontsize = 15)
title = @lift @sprintf("Central Borneo slab-canopy land, h = %.2f m, t = %d h (%s)", run["h₀"], $n - 1, string(start_date + Hour($n - 1)))
Label(fig[0, 1:8], title; fontsize = 20)
function panel!(pos, data, ttl, label; colormap, colorrange)
    ax = Axis(fig[pos...]; title = ttl, aspect = DataAspect())
    hm = heatmap!(ax, λ, φ, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return ax
end
panel!((1, 1), frame(:θ), "slab soil water θ", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 3), θ_obs, "ERA5-Land θ (0–28 cm)", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 5), @lift(mask(snapshots[:rain][$n, :, :]) .* 3600), "ERA5 rain", "mm h⁻¹"; colormap = :dense, colorrange = rainlim)
panel!((1, 7), frame(:Wᶜ), "canopy water Wᶜ", "kg m⁻²"; colormap = :YlGn, colorrange = (0, 1.5))
panel!((2, 1), frame(:LE), "latent heat LE", "W m⁻²"; colormap = :viridis, colorrange = LElim)
panel!((2, 3), frame(:LEᶜ), "LE canopy (transpiration + wet canopy)", "W m⁻²"; colormap = :viridis, colorrange = LElim)
panel!((2, 5), frame(:H), "sensible heat H", "W m⁻²"; colormap = :balance, colorrange = Hlim)
panel!((2, 7), frame(:LST), "land surface temperature", "K"; colormap = :thermal, colorrange = Tlim)

CairoMakie.record(fig, "$(tag).mp4", 1:Nframes; framerate = 8) do k
    n[] = k
end
@info "saved $(tag).mp4"

# Domain-mean time series against ERA5-Land.
t = 0:Nframes-1
θ_model_mean = [mean(snapshots[:θ][k, :, :][land]) for k in 1:Nframes]
θ_obs_mean = [mean(era5_land_soil_water(era5_land, k)[land]) for k in 1:Nframes]
fig2 = Figure(size = (1400, 800), fontsize = 15)
ax = Axis(fig2[1, 1]; title = "Land-mean soil water", xlabel = "t (h)", ylabel = "θ (m³ m⁻³)")
lines!(ax, t, θ_obs_mean; color = :black, linewidth = 2, label = "ERA5-Land 0–28 cm")
lines!(ax, t, θ_model_mean; color = :firebrick, linewidth = 2, label = @sprintf("slab, h = %.2f m", run["h₀"]))
axislegend(ax; position = :lt)
ax = Axis(fig2[1, 2]; title = "Land-mean fluxes", xlabel = "t (h)", ylabel = "W m⁻²")
lines!(ax, t, [mean(snapshots[:LE][k, :, :][land]) for k in 1:Nframes]; color = :navy, label = "LE")
lines!(ax, t, [mean(snapshots[:LEᶜ][k, :, :][land]) for k in 1:Nframes]; color = :seagreen, label = "LE canopy")
lines!(ax, t, [mean(snapshots[:LEᵍ][k, :, :][land]) for k in 1:Nframes]; color = :sienna, label = "LE soil")
lines!(ax, t, [mean(snapshots[:H][k, :, :][land]) for k in 1:Nframes]; color = :orange, label = "H")
axislegend(ax; position = :lt)
ax = Axis(fig2[2, 1]; title = "Land-mean rain and evaporation", xlabel = "t (h)", ylabel = "mm h⁻¹")
lines!(ax, t, [mean(snapshots[:rain][k, :, :][land]) for k in 1:Nframes] .* 3600; color = :steelblue, label = "rain")
lines!(ax, t, [mean(snapshots[:E][k, :, :][land]) for k in 1:Nframes] .* 3600; color = :navy, label = "evaporation")
axislegend(ax; position = :lt)
ax = Axis(fig2[2, 2]; title = "Land-mean temperatures", xlabel = "t (h)", ylabel = "K")
lines!(ax, t, [mean(snapshots[:T][k, :, :][land]) for k in 1:Nframes]; color = :firebrick, label = "slab Tˡᵃ")
lines!(ax, t, [mean(snapshots[:LST][k, :, :][land]) for k in 1:Nframes]; color = :black, label = "LST")
axislegend(ax; position = :lt)
save("$(tag)_timeseries.png", fig2)
@info "saved $(tag)_timeseries.png"
