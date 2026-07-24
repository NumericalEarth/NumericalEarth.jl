# # 3D-GloBFP building morphometry over Manhattan
#
# 3D-GloBFP is a global set of ~1.3 billion individual building footprints, each carrying an
# estimated height. [`BuildingFootprints3D`](@ref) ingests it by **rasterizing** the footprint
# heights onto a fine (3 m) grid — a `:building_height` field, the height of the building
# covering each cell — which is the accurate common source for building morphometry.
#
# [`building_morphometry`](@ref) then reduces that fine raster onto any coarser target grid,
# computing per cell — each with the estimator appropriate to it — the mean height `H`, the
# height standard deviation `σH`, the maximum height `Hmax`, the built-up fraction `λp`, the
# frontal-area index `λf`, and the gross building lift (`= λp·H`, the correction for a digital
# surface model). `σH`, `Hmax` and `λf` are the real height-heterogeneity inputs an urban
# aerodynamic-roughness closure (Kanda et al. 2013) is designed for, in place of the assumed
# ratios (`σH = 0.4 H`, `Hmax = 2.5 H`, `λf ≈ λp`) a mean-height-only product forces.
#
# Manhattan is chosen for extreme height heterogeneity: supertalls in the Financial District
# and Midtown beside low-rise blocks, with Central Park and the rivers as voids. The footprint
# heights are ML-estimated (RMSE 1.9–14.6 m) and biased low.

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum
using NumericalEarth.DataWrangling.GloBFP3D: BuildingFootprints3D, building_morphometry
using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.Grids: λnodes, φnodes
using ArchGDAL                          # activates NumericalEarthArchGDALExt (the OGR read + rasterize)
using CairoMakie                        # loads Makie → NumericalEarthMakieExt
using Statistics: quantile, median

region = BoundingBox(longitude = (-74.02, -73.93), latitude = (40.70, 40.82))
dataset = BuildingFootprints3D(resolution = 3)   # rasterize the footprints at 3 m

# The fine building-height raster (downloads the tile + rasterizes on first use).
building_height = Field(Metadatum(:building_height; dataset, region), CPU())
λf3, φf3 = λnodes(building_height.grid, Center()), φnodes(building_height.grid, Center())

# ## The fine 3 m building-height raster
#
# Each footprint fills the cells it covers, so this is a crisp building map — the common source
# for every morphometric statistic below (Central Park and the rivers read as zeros).
fig1 = Figure(size = (760, 900))
ax1 = Axis(fig1[1, 1]; title = "3D-GloBFP building height (rasterized, 3 m) — Manhattan",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm1 = heatmap!(ax1, building_height; colormap = :viridis, colorrange = (0, 120))
Colorbar(fig1[1, 2], hm1; label = "building height (m)")
save("globfp3d_building_height.png", fig1)
fig1

# ## Morphometry reduced onto a ~100 m grid
#
# `building_morphometry` aggregates the 3 m raster onto the target grid — a mean over built
# cells for `H`, the coverage fraction for `λp`, a max for `Hmax`, height gradients for `λf`.
target_grid = LatitudeLongitudeGrid(CPU(), Float64; size = (102, 136),
                                    longitude = region.longitude, latitude = region.latitude,
                                    topology = (Bounded, Bounded, Flat))
m = building_morphometry(target_grid; dataset, region)

λ = range(region.longitude...; length = size(target_grid, 1))
φ = range(region.latitude...;  length = size(target_grid, 2))
robust_range(field) = (v = filter(>(0), vec(interior(field))); isempty(v) ? (0, 1) : (0, quantile(v, 0.98)))

# The fine 3 m raster (left) shares a colour range with the mean and maximum height panels,
# so the 100 m aggregation's smoothing of the towers is an apples-to-apples comparison.
height_range = (0, max(robust_range(m.mean_building_height)[2], robust_range(m.maximum_building_height)[2]))

function panel!(layout, i, j, field, title, units; colormap = :viridis, colorrange = robust_range(field))
    ax = Axis(layout[i, 2j - 1]; title, xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
    hm = heatmap!(ax, field; colormap, colorrange)
    Colorbar(layout[i, 2j], hm; label = units)
    return ax
end

fig2 = Figure(size = (2050, 950))
Label(fig2[0, 1:2], "3D-GloBFP building morphometry — Manhattan (3 m raster → 100 m)", fontsize = 20)

## Left: the fine 3 m building-height raster, spanning both rows (4× a morphometry panel).
left = fig2[1, 1] = GridLayout()
ax_bh = Axis(left[1, 1]; title = "building height (rasterized, 3 m)",
             xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_bh = heatmap!(ax_bh, λf3, φf3, interior(building_height)[:, :, 1]; colormap = :viridis, colorrange = height_range)
Colorbar(left[1, 2], hm_bh; label = "m")

## Right: the six morphometry fields at 100 m; H and Hmax share the raster's colour range.
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

# Now `σH` is a real field (bright where towers sit among low-rise), not the near-zero result
# that centroid-binning gives when a 100 m cell catches only one large NYC building.

# ## Where the assumed Kanda ratios are wrong
Hm = interior(m.mean_building_height)[:, :, 1]
built = Hm .> 1
ratio(field) = (r = fill(NaN, size(built)); r[built] .= interior(field)[:, :, 1][built] ./ Hm[built]; r)

fig3 = Figure(size = (1250, 520))
for (j, (r, title, assumed)) in enumerate(((ratio(m.maximum_building_height), "Hmax / H  (assumed 2.5)", 2.5),
                                           (ratio(m.building_height_std),     "σH / H  (assumed 0.4)",   0.4)))
    vals = filter(isfinite, vec(r))
    ax = Axis(fig3[1, 2j - 1]; title, xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
    hm = heatmap!(ax, λ, φ, r; colormap = :balance, nan_color = :gray90, colorrange = (0, 2assumed))
    Colorbar(fig3[1, 2j], hm)
    @info "$title:  median = $(round(median(vals), digits=2)) (assumed $assumed);  " *
          "fraction above assumed = $(round(100 * count(>(assumed), vals) / length(vals)))%"
end
Label(fig3[0, :], "Real height ratios vs the assumed Kanda constants", fontsize = 18)
save("globfp3d_assumed_ratios.png", fig3)
fig3

# ## λf departs from the λf ≈ λp assumption
λp_v = vec(interior(m.built_up_fraction)[:, :, 1])
λf_v = vec(interior(m.frontal_area_index)[:, :, 1])
keep = (λp_v .> 0) .& isfinite.(λf_v)

fig4 = Figure(size = (620, 560))
ax4 = Axis(fig4[1, 1]; title = "frontal-area index vs plan-area fraction",
           xlabel = "λp (plan-area fraction)", ylabel = "λf (frontal-area index)")
scatter!(ax4, λp_v[keep], λf_v[keep]; markersize = 3, color = (:steelblue, 0.25))
lines!(ax4, [0, 1], [0, 1]; color = :black, linestyle = :dash, label = "λf = λp (the discarded assumption)")
axislegend(ax4; position = :lt)
ylims!(ax4, 0, quantile(λf_v[keep], 0.99))
save("globfp3d_frontal_vs_plan.png", fig4)
fig4
