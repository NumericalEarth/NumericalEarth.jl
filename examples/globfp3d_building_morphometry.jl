# # 3D-GloBFP building morphometry over Manhattan
#
# 3D-GloBFP is a global set of ~1.3 billion building footprints, each carrying an estimated
# height. [`GlobalBuildingFootprints3D`](@ref) ingests it by rasterizing those heights onto a
# fine (3 m) grid, and [`building_morphometry`](@ref) reduces that raster onto a coarser target
# grid: mean height `H`, height standard deviation `σH`, maximum height `Hmax`, built-up fraction
# `λp`, frontal-area index `λf`, and the gross building lift `λp·H`.
#
# `σH`, `Hmax` and `λf` are the height-heterogeneity inputs an urban aerodynamic-roughness closure
# (Kanda et al. 2013) is designed for, in place of the assumed ratios `σH = 0.4 H`, `Hmax = 2.5 H`,
# `λf ≈ λp` that a mean-height-only product forces. Manhattan makes the point: supertalls in the
# Financial District and Midtown beside low-rise blocks, with Central Park and the rivers as voids.
# The heights are machine-learning estimates (RMSE 1.9–14.6 m) and biased low.

using NumericalEarth
using Oceananigans
using ArchGDAL                          # the OGR read + rasterize
using CairoMakie
using Statistics: quantile, median

region = BoundingBox(longitude = (-74.02, -73.93), latitude = (40.70, 40.82))
dataset = GlobalBuildingFootprints3D(resolution = 3)   # rasterize the footprints at 3 m

# The fine building-height raster (downloads the tile + rasterizes on first use).
building_height = Field(Metadatum(:building_height; dataset, region), CPU())

# ## The fine 3 m building-height raster
#
# Each footprint fills the cells it covers, so Central Park and the rivers read as zeros.
# The raster is ~3000 cells across, and a heatmap minifies by nearest-neighbor sampling —
# below one output pixel per cell it drops the thin gaps between buildings and fuses them
# into streaks. Saving at ≥ 1 pixel per cell keeps every building and street.
fig1 = Figure(size = (760, 900))
ax1 = Axis(fig1[1, 1]; title = "3D-GloBFP building height (rasterized, 3 m) — Manhattan",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm1 = heatmap!(ax1, building_height; colormap = :viridis, colorrange = (0, 120))
Colorbar(fig1[1, 2], hm1; label = "building height (m)")
save("globfp3d_building_height.png", fig1; px_per_unit = 6)
fig1

# ## Morphometry reduced onto a ~100 m grid
target_grid = LatitudeLongitudeGrid(CPU(), Float64; size = (102, 136),
                                    longitude = region.longitude, latitude = region.latitude,
                                    topology = (Bounded, Bounded, Flat))
m = building_morphometry(target_grid; dataset, region)

robust_range(field) = (v = filter(>(0), interior(field, :, :, 1)); isempty(v) ? (0, 1) : (0, quantile(v, 0.98)))

# Sharing one color range between the 3 m raster and the height panels shows how much the 100 m
# aggregation smooths the towers.
height_range = (0, max(robust_range(m.mean_building_height)[2], robust_range(m.maximum_building_height)[2]))

function panel!(layout, i, j, field, title, units; colormap = :viridis, colorrange = robust_range(field))
    ax = Axis(layout[i, 2j - 1]; title, xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
    hm = heatmap!(ax, field; colormap, colorrange)
    Colorbar(layout[i, 2j], hm; label = units)
    return ax
end

fig2 = Figure(size = (2050, 950))
Label(fig2[0, 1:2], "3D-GloBFP building morphometry — Manhattan (3 m raster → 100 m)", fontsize = 20)

## Left: the fine raster, max-pooled to ~9 m by `building_morphometry` itself so this small
## panel stays free of the nearest-neighbor minification streaks.
overview_grid = LatitudeLongitudeGrid(CPU(), Float64; size = size(building_height)[1:2] .÷ 3,
                                      longitude = region.longitude, latitude = region.latitude,
                                      topology = (Bounded, Bounded, Flat))
building_height_overview = building_morphometry(overview_grid; dataset, region).maximum_building_height

left = fig2[1, 1] = GridLayout()
ax_bh = Axis(left[1, 1]; title = "building height (9 m max of the 3 m raster)",
             xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_bh = heatmap!(ax_bh, building_height_overview; colormap = :viridis, colorrange = height_range)
Colorbar(left[1, 2], hm_bh; label = "m")

## Right: the six morphometry fields at 100 m; H and Hmax share the raster's color range.
right = fig2[1, 2] = GridLayout()
panel!(right, 1, 1, m.mean_building_height,    "mean height H",         "m"; colorrange = height_range)
panel!(right, 1, 2, m.maximum_building_height, "maximum height Hmax",   "m"; colorrange = height_range)
panel!(right, 1, 3, m.building_height_std,     "height std σH",         "m"; colormap = :magma)
panel!(right, 2, 1, m.built_up_fraction,       "built-up fraction λp",  "–"; colormap = :turbo)
panel!(right, 2, 2, m.frontal_area_index,      "frontal-area index λf", "–"; colormap = :turbo)
panel!(right, 2, 3, m.gross_building_height,   "gross building lift",   "m")
colsize!(fig2.layout, 1, Relative(0.32))
save("globfp3d_morphometry.png", fig2)
fig2

# `σH` is bright where towers sit among low-rise blocks — the height heterogeneity a
# mean-height-only product cannot express.

# ## Where the assumed Kanda ratios are wrong
#
# The ratios are field operations, so unbuilt cells come out as `0/0 = NaN` and drop out (gray).
Hmax_over_H = compute!(Field(m.maximum_building_height / m.mean_building_height))
σH_over_H   = compute!(Field(m.building_height_std / m.mean_building_height))

ratios = ((Hmax_over_H, "Hmax / H  (assumed 2.5)", 2.5),
          (σH_over_H,   "σH / H  (assumed 0.4)",   0.4))

fig3 = Figure(size = (1250, 520))
for (j, (ratio, title, assumed)) in enumerate(ratios)
    vals = filter(isfinite, interior(ratio, :, :, 1))
    ax = Axis(fig3[1, 2j - 1]; title, xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
    hm = heatmap!(ax, ratio; colormap = :balance, nan_color = :gray90, colorrange = (0, 2assumed))
    Colorbar(fig3[1, 2j], hm)
    @info "$title:  median = $(round(median(vals), digits=2)) (assumed $assumed);  " *
          "fraction above assumed = $(round(100 * count(>(assumed), vals) / length(vals)))%"
end
Label(fig3[0, :], "Real height ratios vs the assumed Kanda constants", fontsize = 18)
save("globfp3d_assumed_ratios.png", fig3)
fig3

# ## λf departs from the λf ≈ λp assumption
λp = interior(m.built_up_fraction, :, :, 1)
λf = interior(m.frontal_area_index, :, :, 1)
keep = (λp .> 0) .& isfinite.(λf)

fig4 = Figure(size = (620, 560))
ax4 = Axis(fig4[1, 1]; title = "frontal-area index vs plan-area fraction",
           xlabel = "λp (plan-area fraction)", ylabel = "λf (frontal-area index)")
scatter!(ax4, λp[keep], λf[keep]; markersize = 3, color = (:steelblue, 0.25))
lines!(ax4, [0, 1], [0, 1]; color = :black, linestyle = :dash, label = "λf = λp (the assumption)")
axislegend(ax4; position = :lt)
ylims!(ax4, 0, quantile(λf[keep], 0.99))
save("globfp3d_frontal_vs_plan.png", fig4)
fig4

# ## The roughness closure fed the measured morphometry
#
# [`urban_roughness`](@ref) evaluates the Macdonald–Kanda closure per cell. Fed only
# `(H, λp)` — all a mean-height product supplies — it regresses `σH`, `Hmax` and `λf` from
# them; fed all five fields, the measured height heterogeneity replaces the regressions and
# only the drag-partition physics and the Kanda height-spread corrections remain.
regressed_roughness_length, regressed_displacement_height =
    urban_roughness(m.mean_building_height, m.built_up_fraction)

roughness_length, displacement_height =
    urban_roughness(m.mean_building_height, m.built_up_fraction, m.building_height_std,
                    m.maximum_building_height, m.frontal_area_index)

roughness_ratio = compute!(Field(roughness_length / regressed_roughness_length))
displacement_shift = compute!(Field(displacement_height - regressed_displacement_height))

fig5 = Figure(size = (1900, 1150))
Label(fig5[0, 1:6], "Aerodynamic roughness of Manhattan: measured σH, Hmax, λf vs the regressions", fontsize = 20)
panel!(fig5, 1, 1, roughness_length,               "ℓᵐ, measured morphometry",  "m")
panel!(fig5, 1, 2, regressed_roughness_length,     "ℓᵐ, regressed from (H, λp)", "m";
       colorrange = robust_range(roughness_length))
panel!(fig5, 2, 1, displacement_height,            "d, measured morphometry",   "m")
panel!(fig5, 2, 2, regressed_displacement_height,  "d, regressed from (H, λp)", "m";
       colorrange = robust_range(displacement_height))

## The comparison panels: a log₂-scaled ratio (centered on 1) and the displacement shift.
ax5 = Axis(fig5[1, 5]; title = "ℓᵐ ratio (measured / regressed)",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm5 = heatmap!(ax5, roughness_ratio; colormap = :balance, colorscale = log2, colorrange = (1/8, 8))
Colorbar(fig5[1, 6], hm5; label = "–")
panel!(fig5, 2, 3, displacement_shift,             "d shift (measured − regressed)", "m";
       colormap = :balance, colorrange = (-100, 100))
save("globfp3d_urban_roughness.png", fig5)
fig5

# Over the built cells, the medians quantify what the measured inputs change.
built = interior(m.built_up_fraction, :, :, 1) .> 0.01
for (field, name, units) in ((roughness_ratio, "ℓᵐ measured/regressed", ""),
                             (displacement_shift, "d measured − regressed", " m"))
    vals = filter(isfinite, interior(field, :, :, 1)[built])
    @info "$name: median = $(round(median(vals), digits = 2))$units, " *
          "IQR = $(round.(quantile(vals, (0.25, 0.75)), digits = 2))"
end

# The regressions were fitted to 1 km Tokyo/Nagoya districts, so on a 100 m grid they hand
# every cell with 30 m mean height a ~160 m tallest building — the median regressed `ℓᵐ` runs
# ~4× the measured-input value and `d` ~2×. The measured statistics are per-cell facts, valid
# at any resolution; only where supertalls actually stand do `σH` and `Hmax` stay this large.
