# Land-cover sensitivity: the WorldCover-driven and MODIS-driven runs side by side.
#   julia --project=<docs> compare_landcover.jl [TAG_WORLDCOVER] [TAG_MODIS]
using Oceananigans
using NumericalEarth
using CairoMakie
using JLD2
using Printf
using Statistics: mean, quantile

tag_a = length(ARGS) ≥ 1 ? ARGS[1] : "conus12km_v2"
tag_b = length(ARGS) ≥ 2 ? ARGS[2] : "conus12km_v2_modis"

series(tag, name) = FieldTimeSeries("$(tag)_land.jld2", name; backend = OnDisk())
LE_a = series(tag_a, "LE"); LE_b = series(tag_b, "LE")
H_a  = series(tag_a, "H");  H_b  = series(tag_b, "H")
LST_a = series(tag_a, "LST"); LST_b = series(tag_b, "LST")
Jʳⁿ_a = series(tag_a, "Jʳⁿ"); Jʳⁿ_b = series(tag_b, "Jʳⁿ")

times = LE_a.times
Nt = min(length(times), length(LE_b.times))
grid = LE_a.grid
λ, φ, _ = nodes(grid, Center(), Center(), Center())
static_a = jldopen("$(tag_a)_static.jld2"); static_b = jldopen("$(tag_b)_static.jld2")
water = static_a["water"] .| static_b["water"]
land = .!water
mask(a) = ifelse.(water, NaN, a)
lmean(fts, n) = mean(interior(fts[n], :, :, 1)[land])

n13 = argmin(abs.(times .- (3 * 86400 + 19 * 3600)))
field(fts, n) = interior(fts[n], :, :, 1)

fveg_a = static_a["vegetation_fraction"]; fveg_b = static_b["vegetation_fraction"]
urban_a = static_a["urban_cover"]; urban_b = static_b["urban_cover"]
ΔLE = field(LE_a, n13) .- field(LE_b, n13)
ΔH = field(H_a, n13) .- field(H_b, n13)
ΔLST = field(LST_a, n13) .- field(LST_b, n13)

@printf("land cells %d; vegetated fraction: WorldCover mean %.3f, MODIS mean %.3f; built-up: %.4f vs %.4f\n",
        count(land), mean(fveg_a[land]), mean(fveg_b[land]), mean(urban_a[land]), mean(urban_b[land]))
@printf("1300 CST case day, land mean: LE %.1f vs %.1f, H %.1f vs %.1f, LST %.2f vs %.2f K\n",
        mean(field(LE_a, n13)[land]), mean(field(LE_b, n13)[land]), mean(field(H_a, n13)[land]), mean(field(H_b, n13)[land]),
        mean(field(LST_a, n13)[land]), mean(field(LST_b, n13)[land]))
@printf("ΔLE (WorldCover − MODIS) over land: mean %.1f, 5%% %.1f, 95%% %.1f W/m²; |Δf_veg| > 0.2 in %d cells\n",
        mean(ΔLE[land]), quantile(vec(ΔLE[land]), (0.05, 0.95))..., count(abs.(fveg_a .- fveg_b)[land] .> 0.2))
big = land .& (abs.(fveg_a .- fveg_b) .> 0.2)
@printf("where |Δf_veg| > 0.2: Δf_veg mean %.2f, ΔLE mean %.1f, ΔH mean %.1f, ΔLST mean %.2f K\n",
        mean((fveg_a .- fveg_b)[big]), mean(ΔLE[big]), mean(ΔH[big]), mean(ΔLST[big]))

fig = Figure(size = (1800, 1300), fontsize = 15)
panels = (("vegetated fraction — WorldCover", fveg_a, :speed, (0, 1)),
          ("vegetated fraction — MODIS IGBP", fveg_b, :speed, (0, 1)),
          ("Δ vegetated fraction", fveg_a .- fveg_b, :balance, (-0.5, 0.5)),
          ("Δ LE at 1300 CST (W m⁻²)", ΔLE, :balance, (-150, 150)),
          ("Δ H at 1300 CST (W m⁻²)", ΔH, :balance, (-150, 150)),
          ("Δ LST at 1300 CST (K)", ΔLST, :balance, (-3, 3)))
for (k, (title, data, colormap, colorrange)) in enumerate(panels)
    row, column = fldmod1(k, 3)
    ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, mask(data); colormap, colorrange, nan_color = :lightsteelblue1)
    Colorbar(fig[row, 2column], hm)
end
hours = times[1:Nt] ./ 3600
ax = Axis(fig[3, 1:6]; title = "land-mean fluxes: WorldCover (solid) vs MODIS (dashed) land cover", xlabel = "hours since 17 May 00 UTC", ylabel = "W m⁻²")
lines!(ax, hours, [lmean(LE_a, n) for n in 1:Nt]; color = :navy, label = "LE")
lines!(ax, hours, [lmean(LE_b, n) for n in 1:Nt]; color = :navy, linestyle = :dash)
lines!(ax, hours, [lmean(H_a, n) for n in 1:Nt]; color = :orangered, label = "H")
lines!(ax, hours, [lmean(H_b, n) for n in 1:Nt]; color = :orangered, linestyle = :dash)
lines!(ax, hours, [3600 * 100 * lmean(Jʳⁿ_a, n) for n in 1:Nt]; color = :steelblue, label = "rain × 100 (mm hr⁻¹)")
lines!(ax, hours, [3600 * 100 * lmean(Jʳⁿ_b, n) for n in 1:Nt]; color = :steelblue, linestyle = :dash)
axislegend(ax; position = :lt)
Label(fig[0, 1:6], "Land-cover sensitivity: ESA WorldCover vs MODIS IGBP fractions, 20 May 2011 1300 CST", fontsize = 18)
save("$(tag_a)_vs_$(tag_b).png", fig)
@info "Saved $(tag_a)_vs_$(tag_b).png"
close(static_a); close(static_b)
