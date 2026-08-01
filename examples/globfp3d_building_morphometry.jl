# # 3D-GloBFP building morphometry over Manhattan
#
# 3D-GloBFP is a global set of ~1.3 billion building footprints, each carrying an estimated
# height. [`GlobalBuildingFootprints3D`](@ref) ingests it by rasterizing those heights onto a
# fine (3 m) grid, and [`building_morphometry`](@ref) reduces that raster onto a coarser target
# grid: mean height `h`, height standard deviation `σʰ`, maximum height `hᵐᵃˣ`, plan-area index
# `λᵖ`, frontal-area index `λᶠ`, and the gross building lift `λᵖ·h`.
#
# `σʰ`, `hᵐᵃˣ` and `λᶠ` are the height-heterogeneity inputs an urban aerodynamic-roughness closure
# (Kanda et al. 2013) is designed for, in place of the assumed ratios `σʰ = 0.4 h`, `hᵐᵃˣ = 2.5 h`,
# `λᶠ ≈ λᵖ` that a mean-height-only product forces. Manhattan makes the point: supertalls in the
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
morphometry = building_morphometry(target_grid; dataset, region)

robust_range(field) = (v = filter(>(0), interior(field, :, :, 1)); isempty(v) ? (0, 1) : (0, quantile(v, 0.98)))

# Sharing one color range between the 3 m raster and the height panels shows how much the 100 m
# aggregation smooths the towers.
height_range = (0, max(robust_range(morphometry.mean_building_height)[2],
                       robust_range(morphometry.maximum_building_height)[2]))

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

## Right: the six morphometry fields at 100 m; h and hᵐᵃˣ share the raster's color range.
right = fig2[1, 2] = GridLayout()
panel!(right, 1, 1, morphometry.mean_building_height,      "mean height h",         "m"; colorrange = height_range)
panel!(right, 1, 2, morphometry.maximum_building_height,   "maximum height hᵐᵃˣ",   "m"; colorrange = height_range)
panel!(right, 1, 3, morphometry.building_height_deviation, "height deviation σʰ",   "m"; colormap = :magma)
panel!(right, 2, 1, morphometry.plan_area_index,           "plan-area index λᵖ",    "–"; colormap = :turbo)
panel!(right, 2, 2, morphometry.frontal_area_index,        "frontal-area index λᶠ", "–"; colormap = :turbo)
panel!(right, 2, 3, morphometry.gross_building_height,     "gross building lift",   "m")
colsize!(fig2.layout, 1, Relative(0.32))
save("globfp3d_morphometry.png", fig2)
fig2

# `σʰ` is bright where towers sit among low-rise blocks — the height heterogeneity a
# mean-height-only product cannot express.

# ## Where the assumed Kanda ratios are wrong
#
# The ratios are field operations, so unbuilt cells come out as `0/0 = NaN` and drop out (gray).
maximum_to_mean_height = compute!(Field(morphometry.maximum_building_height / morphometry.mean_building_height))
spread_to_mean_height  = compute!(Field(morphometry.building_height_deviation / morphometry.mean_building_height))

ratios = ((maximum_to_mean_height, "hᵐᵃˣ / h  (assumed 2.5)", 2.5),
          (spread_to_mean_height,  "σʰ / h  (assumed 0.4)",   0.4))

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

# ## λᶠ departs from the λᶠ ≈ λᵖ assumption
λᵖ = interior(morphometry.plan_area_index, :, :, 1)
λᶠ = interior(morphometry.frontal_area_index, :, :, 1)
keep = (λᵖ .> 0) .& isfinite.(λᶠ)

fig4 = Figure(size = (620, 560))
ax4 = Axis(fig4[1, 1]; title = "frontal-area index vs plan-area index",
           xlabel = "λᵖ (plan-area index)", ylabel = "λᶠ (frontal-area index)")
scatter!(ax4, λᵖ[keep], λᶠ[keep]; markersize = 3, color = (:steelblue, 0.25))
lines!(ax4, [0, 1], [0, 1]; color = :black, linestyle = :dash, label = "λᶠ = λᵖ (the assumption)")
axislegend(ax4; position = :lt)
ylims!(ax4, 0, quantile(λᶠ[keep], 0.99))
save("globfp3d_frontal_vs_plan.png", fig4)
fig4

# ## The roughness closure fed the measured morphometry
#
# [`urban_roughness`](@ref) evaluates the Macdonald–Kanda closure per cell. Fed only
# `(h, λᵖ)` — all a mean-height product supplies — it regresses `σʰ`, `hᵐᵃˣ` and `λᶠ` from
# them; fed all five fields, the measured height heterogeneity replaces the regressions and
# only the drag-partition physics and the Kanda height-spread corrections remain.
regressed_roughness_length, regressed_displacement_height =
    urban_roughness(morphometry.mean_building_height, morphometry.plan_area_index)

roughness_length, displacement_height =
    urban_roughness(morphometry.mean_building_height, morphometry.plan_area_index,
                    morphometry.building_height_deviation, morphometry.maximum_building_height,
                    morphometry.frontal_area_index)

roughness_ratio = compute!(Field(roughness_length / regressed_roughness_length))
displacement_shift = compute!(Field(displacement_height - regressed_displacement_height))

fig5 = Figure(size = (1900, 1150))
Label(fig5[0, 1:6], "Aerodynamic roughness of Manhattan: measured σʰ, hᵐᵃˣ, λᶠ vs the regressions", fontsize = 20)
panel!(fig5, 1, 1, roughness_length,               "ℓᵐ, measured morphometry",  "m")
panel!(fig5, 1, 2, regressed_roughness_length,     "ℓᵐ, regressed from (h, λᵖ)", "m";
       colorrange = robust_range(roughness_length))
panel!(fig5, 2, 1, displacement_height,            "d, measured morphometry",   "m")
panel!(fig5, 2, 2, regressed_displacement_height,  "d, regressed from (h, λᵖ)", "m";
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
built = interior(morphometry.plan_area_index, :, :, 1) .> 0.01
for (field, name, units) in ((roughness_ratio, "ℓᵐ measured/regressed", ""),
                             (displacement_shift, "d measured − regressed", " m"))
    vals = filter(isfinite, interior(field, :, :, 1)[built])
    @info "$name: median = $(round(median(vals), digits = 2))$units, " *
          "IQR = $(round.(quantile(vals, (0.25, 0.75)), digits = 2))"
end

# The regressions were fitted to 1 km Tokyo/Nagoya districts, so on a 100 m grid they hand
# every cell with 30 m mean height a ~160 m tallest building — the median regressed `ℓᵐ` runs
# ~4× the measured-input value and `d` ~2×. The measured statistics are per-cell facts, valid
# at any resolution; only where supertalls actually stand do `σʰ` and `hᵐᵃˣ` stay this large.

# ## The same comparison at the regressions' fitted 1 km scale
#
# If that reading is right, aggregating the morphometry to the ~1 km districts the regressions
# were fitted on should close much of the gap: a 1 km Manhattan cell really does mix towers
# with low-rise blocks.
kilometer_grid = LatitudeLongitudeGrid(CPU(), Float64; size = (8, 13),
                                       longitude = region.longitude, latitude = region.latitude,
                                       topology = (Bounded, Bounded, Flat))
kilometer_morphometry = building_morphometry(kilometer_grid; dataset, region)

kilometer_regressed_roughness, kilometer_regressed_displacement =
    urban_roughness(kilometer_morphometry.mean_building_height, kilometer_morphometry.plan_area_index)

kilometer_measured_roughness, kilometer_measured_displacement =
    urban_roughness(kilometer_morphometry.mean_building_height, kilometer_morphometry.plan_area_index,
                    kilometer_morphometry.building_height_deviation, kilometer_morphometry.maximum_building_height,
                    kilometer_morphometry.frontal_area_index)

kilometer_built = interior(kilometer_morphometry.plan_area_index, :, :, 1) .> 0.01
cell_values(field, mask) = interior(field, :, :, 1)[mask]

fig6 = Figure(size = (1250, 640))
Label(fig6[0, 1:2], "Measured vs regressed closure inputs, at 100 m and at the fitted 1 km scale", fontsize = 18)
panels = (("roughness length ℓᵐ (m)", regressed_roughness_length, roughness_length,
           kilometer_regressed_roughness, kilometer_measured_roughness),
          ("displacement height d (m)", regressed_displacement_height, displacement_height,
           kilometer_regressed_displacement, kilometer_measured_displacement))
for (j, (name, regressed, measured, kilometer_regressed, kilometer_measured)) in enumerate(panels)
    ax = Axis(fig6[1, j]; xscale = log10, yscale = log10, title = name, aspect = 1,
              xlabel = "regressed from (h, λᵖ)", ylabel = "measured σʰ, hᵐᵃˣ, λᶠ")
    scatter!(ax, cell_values(regressed, built), cell_values(measured, built);
             markersize = 4, color = (:steelblue, 0.25), label = "100 m cells")
    scatter!(ax, cell_values(kilometer_regressed, kilometer_built), cell_values(kilometer_measured, kilometer_built);
             markersize = 11, color = :orangered, strokecolor = :black, strokewidth = 0.5, label = "1 km cells")
    low, high = extrema(vcat(cell_values(regressed, built), cell_values(measured, built)))
    lines!(ax, [low, high], [low, high]; color = :black, linestyle = :dash)
    axislegend(ax; position = :lt)
end
save("globfp3d_scale_dependence.png", fig6)
fig6

kilometer_roughness_ratio = cell_values(kilometer_measured_roughness, kilometer_built) ./
                            cell_values(kilometer_regressed_roughness, kilometer_built)
kilometer_height_spread_ratio = cell_values(kilometer_morphometry.building_height_deviation, kilometer_built) ./
                                cell_values(kilometer_morphometry.mean_building_height, kilometer_built)
@info "1 km ℓᵐ measured/regressed: median = $(round(median(kilometer_roughness_ratio), digits = 2)), " *
      "IQR = $(round.(quantile(kilometer_roughness_ratio, (0.25, 0.75)), digits = 2))"
@info "1 km measured σʰ/h: median = $(round(median(kilometer_height_spread_ratio), digits = 2))"

# It does — about half of it. From 100 m to 1 km the measured `σʰ/h` median rises from 0.09
# to ~0.34, `hᵐᵃˣ/h` from 1.11 to ~2.05 (toward the assumed 2.5), and the `ℓᵐ` ratio recovers
# from ~0.27 to ~0.53, with the 1 km cells straddling the 1:1 line. The factor ~2 that remains
# even at the fitted scale is a city mismatch: Kanda's `σʰ = 1.05h − 3.7` prescribes
# `σʰ/h ≈ 0.95` at Manhattan's 1 km mean heights, three times the heterogeneity its uniform
# mid- and high-rise districts actually have. The measured morphometry carries neither the
# fitted scale nor the fitted city.
