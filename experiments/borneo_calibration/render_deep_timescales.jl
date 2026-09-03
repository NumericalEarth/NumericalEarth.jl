# Timescales of ERA5-Land's deep (28–100 cm) layer against the surface (0–28 cm) layer over the
# 20-day window: the deep head as the Darcy runs see it (hourly series), its 2- and 5-day running
# means, its time mean and its t = 0 value, with a slow/fast variance partition per layer.
#
#   julia --project=docs render_deep_timescales.jl   (from the run directory)

using CairoMakie
using JLD2
using Statistics: mean, median, std, cor
using Printf

era5 = jldopen(f -> f["data"], "surface_cache/era5_land_r1.jld2")
static = jldopen(f -> Dict(k => f[k] for k in keys(f)), "surface_cache/static_r1.jld2")
land = .!static["water"]
cells = findall(land)
Nt = length(era5.times)
days = (0:Nt-1) ./ 24
θ_surface = 0.25 .* era5.layer_1 .+ 0.75 .* era5.layer_2
θ_deep = era5.layer_3

head(θ, α, n, ν, θʳ) = (𝒮 = clamp((θ - θʳ) / (ν - θʳ), 1e-6, 1); m = 1 - 1 / n; 𝒮 ≥ 1 ? 0.0 : -(𝒮^(-1 / m) - 1)^(1 / n) / α)
Π_deep = [head(θ_deep[t, i, j], static["inverse_air_entry_head"][i, j], static["pore_size_uniformity"][i, j],
               static["porosity"][i, j], static["residual_liquid_fraction"][i, j]) for t in 1:Nt, i in axes(θ_deep, 2), j in axes(θ_deep, 3)]

running_mean(x, w) = [mean(x[max(1, t - w ÷ 2):min(length(x), t + w ÷ 2)]) for t in eachindex(x)]
med(x) = [median(x[t, :, :][land]) for t in 1:Nt]
function efolding(x)
    a = x .- mean(x)
    k = findfirst(k -> sum(a[1:end-k] .* a[1+k:end]) / sum(a .* a) < exp(-1), 1:Nt-1)
    return isnothing(k) ? Inf : k / 24
end
partition(series, w) = (median([std(running_mean(series[:, c], w)) for c in cells]), median([std(series[:, c] .- running_mean(series[:, c], w)) for c in cells]))

fig = Figure(size = (1500, 1150), fontsize = 15)
Label(fig[0, 1:2], "ERA5-Land over the Borneo box: does the deep layer vary on the surface layer's timescale?"; fontsize = 19)

ax = Axis(fig[1, 1:2]; title = "domain-median soil water", xlabel = "day", ylabel = "θ (m³ m⁻³)")
vlines!(ax, [6.25, 12.25]; color = :gray60, linestyle = :dash)
lines!(ax, days, med(θ_surface); color = :black, linewidth = 2.4, label = "0–28 cm (the calibration target)")
lines!(ax, days, med(θ_deep); color = :firebrick, linewidth = 2.4, label = "28–100 cm (the deep reservoir)")
axislegend(ax; position = :rt)

ax = Axis(fig[2, 1:2]; title = "domain-median deep pressure head Πᵈ, as the Darcy boundary sees it", xlabel = "day", ylabel = "Πᵈ (m)")
vlines!(ax, [6.25, 12.25]; color = :gray60, linestyle = :dash)
lines!(ax, days, med(Π_deep); color = :firebrick, linewidth = 2.4, label = "hourly series (suite 8)")
lines!(ax, days, med(mapslices(x -> running_mean(x, 48), Π_deep; dims = 1)); color = :deepskyblue, linewidth = 2, label = "2-day running mean")
lines!(ax, days, med(mapslices(x -> running_mean(x, 120), Π_deep; dims = 1)); color = :goldenrod, linewidth = 2, label = "5-day running mean")
hlines!(ax, [median(mean(Π_deep; dims = 1)[1, :, :][land])]; color = :sienna, linewidth = 2, linestyle = :dot, label = "time mean")
hlines!(ax, [median(Π_deep[1, :, :][land])]; color = :teal, linewidth = 2, linestyle = :dash, label = "t = 0 value (initial condition only)")
axislegend(ax; position = :rb, nbanks = 2)

ax = Axis(fig[3, 1]; title = "per-cell 20-day range of the deep head vs surface θ", xlabel = "surface θ range (m³ m⁻³)", ylabel = "deep suction range (m)")
scatter!(ax, [maximum(θ_surface[:, c]) - minimum(θ_surface[:, c]) for c in cells], [maximum(Π_deep[:, c]) - minimum(Π_deep[:, c]) for c in cells];
         color = :firebrick, markersize = 7, alpha = 0.7)

rows = [("surface θ", θ_surface), ("deep θ", θ_deep), ("deep head Πᵈ", Π_deep)]
text = join([@sprintf("%-14s e-folding %.1f d   fast (< 2 d) share %2.0f %%", name,
                      median([efolding(s[:, c]) for c in cells]),
                      100 * partition(s, 48)[2]^2 / sum(abs2, partition(s, 48))) for (name, s) in rows], "\n")
r_fast = median([cor(Π_deep[:, c] .- running_mean(Π_deep[:, c], 48), θ_surface[:, c] .- running_mean(θ_surface[:, c], 48)) for c in cells])
r_slow = median([cor(running_mean(Π_deep[:, c], 48), running_mean(θ_surface[:, c], 48)) for c in cells])
Label(fig[3, 2], text * @sprintf("\n\ncorrelation of deep head with surface θ (median per cell):\n  slow (> 2 d) components r = %.2f;  fast components r = %.2f", r_slow, r_fast);
      font = "DejaVu Sans Mono", fontsize = 14, justification = :left, tellwidth = false)
save("deep_timescales_r1.png", fig)
@info "saved deep_timescales_r1.png"
