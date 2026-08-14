# # ESA WorldCover fractional land cover
#
# This example ingests [ESA WorldCover](https://esa-worldcover.org) 10 m land-cover
# tiles over a mixed forest/urban/water/farmland region — the Veluwe and the
# Randmeren lakes in the central Netherlands — and turns them into the continuous
# fields a fractional-cover land surface consumes:
#
#   * `vegetation_fraction` — the mosaic weight `f_veg` (a `TiledLandInterface`'s
#     area weight between its vegetated and non-vegetated tiles);
#   * one `<class>_fraction` per land-cover class (tree cover, cropland, water, …);
#   * `landcover_class` — the majority class code, for per-tile biome priors.
#
# The 10 m `Map` band is *categorical*: its byte value is a class code, not a
# quantity, so it is never averaged. The ingest counts codes over each aggregated
# cell to form area fractions. Onto the model grid the fractions are regridded
# conservatively (area-weighted, so they still sum to one), and the categorical
# majority class is the argmax of those fractions — the class covering the most
# area of each cell, never a blend, so no intermediate code is ever invented.

# ## Load packages
using NumericalEarth
using Oceananigans
using ArchGDAL                       # activates the anonymous-COG read path
using CairoMakie
using Statistics                     # mean

# ## Region, dataset, and model grid
#
# The window spans the Veluwe forest, the cities of Apeldoorn and Harderwijk, the
# Randmeren lakes, and the surrounding polders and farmland. The default
# `aggregation_factor = 12` reduces the 10 m raster to ~110 m cells — comparable
# to a regional-LES grid, so each aggregated cell still samples ~144 sub-pixels.

region  = BoundingBox(longitude = (5.45, 5.95), latitude = (52.05, 52.45))
dataset = ESAWorldCover()

grid = LatitudeLongitudeGrid(CPU();
                             size = (150, 120),
                             longitude = region.longitude,
                             latitude  = region.latitude,
                             topology = (Bounded, Bounded, Flat))

# The ESA WorldCover legend: verbose name, class code, and the official palette
# colour used for the categorical map below.
legend = (tree_cover              = (10,  "#006400"),
          shrubland               = (20,  "#ffbb22"),
          grassland               = (30,  "#ffff4c"),
          cropland                = (40,  "#f096ff"),
          built_up                = (50,  "#fa0000"),
          bare_sparse_vegetation  = (60,  "#b4b4b4"),
          snow_and_ice            = (70,  "#f0f0f0"),
          permanent_water_bodies  = (80,  "#0064c8"),
          herbaceous_wetland      = (90,  "#0096a0"),
          mangroves               = (95,  "#00cf75"),
          moss_and_lichen         = (100, "#fae6a0"))

class_names = keys(legend)
class_codes = map(first, values(legend))
class_colors = map(last, values(legend))

# ## Load the fields
#
# `f_veg` and the majority class on both the native aggregated grid and the model
# grid, plus every per-class fraction on the model grid. Each is one
# `Field(Metadatum(...), grid)` call — the first materializes the regional NetCDF
# from the anonymous S3 tiles, the rest read cached bands. The model-grid fractions
# ride the conservative regrid; the model-grid class rides nearest-neighbor.

fveg_native = Field(Metadatum(:vegetation_fraction; dataset, region), CPU())
fveg_model  = Field(Metadatum(:vegetation_fraction; dataset, region), grid)
class_native = Field(Metadatum(:landcover_class; dataset, region), CPU())
class_model  = Field(Metadatum(:landcover_class; dataset, region), grid)

fraction_fields = NamedTuple(name => Field(Metadatum(Symbol(name, :_fraction); dataset, region), grid)
                             for name in class_names)

# Sum of the eleven per-class fractions — a wiring check that must be ≈ 1 over
# every valid cell.
fraction_sum = sum(interior(f) for f in fraction_fields)

# Grid-node coordinates for plotting.
λ_native, φ_native = λnodes(class_native.grid, Center()), φnodes(class_native.grid, Center())
λ_model,  φ_model  = λnodes(fveg_model.grid, Center()),   φnodes(fveg_model.grid, Center())

array(field) = Array(interior(field))[:, :, 1]

# ## Physical checks
#
# Before plotting, confirm the pipeline is physically consistent on the model grid.
# The categorical class must stay on exact legend codes (nearest-neighbor never
# blends); the conservative regrid must keep the eleven fractions summing to one
# and inside `[0, 1]`; and `f_veg` must equal the sum of the vegetated-class
# fractions (the vegetation definition ESA WorldCover's tree/shrub/grass/crop/
# wetland/mangrove classes imply).

model_codes = filter(!isnan, unique(round.(Int, vec(array(class_model)))))
@info "model-grid majority-class codes ⊆ legend (no invented codes): $(issubset(model_codes, class_codes))"

fraction_sum_model = sum(array(fraction_fields[name]) for name in class_names)
@info "Σ class fractions (model grid): extrema = $(extrema(filter(!isnan, fraction_sum_model)))"
@info "f_veg (model grid): extrema = $(extrema(filter(!isnan, array(fveg_model)))), any NaN = $(any(isnan, array(fveg_model)))"

vegetated_codes = (10, 20, 30, 40, 90, 95)   # tree, shrub, grass, crop, herbaceous wetland, mangrove
vegetated_names = [name for name in class_names if legend[name][1] in vegetated_codes]
fveg_from_classes = sum(array(fraction_fields[name]) for name in vegetated_names)
@info "max |f_veg − Σ vegetated fractions| (model grid): $(maximum(abs.(array(fveg_model) .- fveg_from_classes)))"

# ## Majority land-cover class: native vs model grid
#
# On the native grid the majority class is an exact legend code. On the model grid
# it is the argmax of the conservatively regridded fractions — the class covering
# the most area of each cell — so it stays on exact legend codes and agrees with
# the fraction fields. We map each code to its legend index so the categorical
# palette and legend line up.

to_class_index(field) = map(array(field)) do code
    isnan(code) && return NaN                        # no-data cells (ocean / outside coverage)
    i = findfirst(==(round(Int, code)), class_codes)
    isnothing(i) ? NaN : Float64(i)
end

class_index_native = to_class_index(class_native)
class_index_model  = to_class_index(class_model)

fig = Figure(size = (1180, 640), fontsize = 15)
for (col, panel) in enumerate((("native ~110 m", λ_native, φ_native, class_index_native),
                               ("model grid (area majority)", λ_model, φ_model, class_index_model)))
    title, λ, φ, class_index = panel
    ax = Axis(fig[1, col]; title = "Majority class ($title)", xlabel = "longitude", ylabel = "latitude")
    heatmap!(ax, λ, φ, class_index;
             colormap = cgrad(collect(class_colors), categorical = true),
             colorrange = (0.5, length(class_codes) + 0.5))
end
present = sort(unique(filter(!isnan, vcat(vec(class_index_native), vec(class_index_model)))))
Legend(fig[1, 3],
       [PolyElement(color = class_colors[Int(i)]) for i in present],
       [replace(string(class_names[Int(i)]), "_" => " ") for i in present],
       "class"; framevisible = false)
save("esa_worldcover_landcover_class.png", fig)
fig

# ## Per-class area fractions
#
# One panel per class that occupies at least ~1% of the domain, each on the model
# grid with a shared 0–1 colour scale. Vegetated classes carry most of the area;
# built-up and water appear cleanly where the cities and lakes are.

shown = [name for name in class_names if mean(array(fraction_fields[name])) > 0.01]
ncols = 3
nrows = cld(length(shown), ncols)
fig = Figure(size = (360 * ncols + 90, 300 * nrows), fontsize = 14)
for (k, name) in enumerate(shown)
    i, j = fldmod1(k, ncols)
    axis = Axis(fig[i, j]; title = replace(string(name), "_" => " "),
                xlabel = "longitude", ylabel = "latitude")
    heatmap!(axis, λ_model, φ_model, array(fraction_fields[name]);
             colormap = :viridis, colorrange = (0, 1))
end
Colorbar(fig[:, ncols + 1]; colorrange = (0, 1), colormap = :viridis, label = "area fraction")
save("esa_worldcover_class_fractions.png", fig)
fig

# ## Vegetation fraction `f_veg` and its distribution
#
# `f_veg` is high over the Veluwe forest and the farmland, and drops toward zero
# over the cities and the lakes.

fig = Figure(size = (1120, 460), fontsize = 15)
ax = Axis(fig[1, 1]; title = "Vegetation fraction f_veg (model grid)",
          xlabel = "longitude", ylabel = "latitude")
hm = heatmap!(ax, λ_model, φ_model, array(fveg_model); colormap = :YlGn, colorrange = (0, 1))
Colorbar(fig[1, 2], hm; label = "f_veg")
ax2 = Axis(fig[1, 3]; title = "distribution of f_veg", xlabel = "f_veg", ylabel = "cells")
hist!(ax2, vec(array(fveg_model)); bins = 30, color = (:seagreen, 0.8))
save("esa_worldcover_f_veg.png", fig)
fig

# ## Sum of fractions (wiring check) and native-vs-model comparison
#
# The eleven per-class fractions sum to ≈ 1 over every valid cell (left). The
# aggregated pattern is preserved from the native ~110 m grid to the model grid
# (right two panels): the conservative regrid area-averages but does not move the
# forest, cities, or lakes.

@info "sum-of-fractions over the domain: extrema = $(extrema(filter(!isnan, fraction_sum)))"

fig = Figure(size = (1500, 460), fontsize = 15)
ax = Axis(fig[1, 1]; title = "Σ class fractions (≈ 1)", xlabel = "longitude", ylabel = "latitude")
hm = heatmap!(ax, λ_model, φ_model, fraction_sum[:, :, 1]; colormap = :balance, colorrange = (0.95, 1.05))
Colorbar(fig[1, 2], hm)

ax = Axis(fig[1, 3]; title = "f_veg (native ~110 m)", xlabel = "longitude", ylabel = "latitude")
heatmap!(ax, λ_native, φ_native, array(fveg_native); colormap = :YlGn, colorrange = (0, 1))
ax = Axis(fig[1, 4]; title = "f_veg (model grid)", xlabel = "longitude", ylabel = "latitude")
hm = heatmap!(ax, λ_model, φ_model, array(fveg_model); colormap = :YlGn, colorrange = (0, 1))
Colorbar(fig[1, 5], hm; label = "f_veg")
save("esa_worldcover_fveg_check.png", fig)
fig
