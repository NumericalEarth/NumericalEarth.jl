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
# quantity, so it is never averaged. Instead, the ingest counts the codes in each
# aggregated cell to form area fractions. The fractions are regridded onto the model
# grid conservatively (area-weighted, so they still sum to the mapped share), and the
# majority class is the argmax of those fractions: the class that covers the most
# area of each cell. A class is never blended, so no intermediate code is invented.

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
# `aggregation_factor = 12` reduces the 10 m raster to ~110 m cells, comparable
# to a regional LES grid; each aggregated cell still samples ~144 sub-pixels.

region  = BoundingBox(longitude = (5.45, 5.95), latitude = (52.05, 52.45))
dataset = ESAWorldCover()

grid = LatitudeLongitudeGrid(CPU();
                             size = (150, 120),
                             longitude = region.longitude,
                             latitude  = region.latitude,
                             topology = (Bounded, Bounded, Flat))

# The ESA WorldCover legend: the verbose name, the class code, and the official palette
# color used for the categorical map below.
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
# We load the vegetation fraction `f_veg` and the majority class on both the native
# aggregated grid and the model grid, plus every per-class fraction on the model grid.
# Each is one `Field(Metadatum(...), grid)` call: the first materializes the regional
# NetCDF from the anonymous S3 tiles, and the rest read cached bands. The fractions reach
# the model grid by conservative regridding; the class reaches it by nearest neighbor.

native_vegetation_fraction = Field(Metadatum(:vegetation_fraction; dataset, region), CPU())
vegetation_fraction        = Field(Metadatum(:vegetation_fraction; dataset, region), grid)
native_landcover_class     = Field(Metadatum(:landcover_class; dataset, region), CPU())
landcover_class            = Field(Metadatum(:landcover_class; dataset, region), grid)

fraction_fields = NamedTuple(name => Field(Metadatum(Symbol(name, :_fraction); dataset, region), grid)
                             for name in class_names)

# The sum of the eleven per-class fractions is a wiring check: it must be ≈ 1 over every
# valid land cell.

fraction_sum = Field(sum(fraction_fields))

# ## Physical checks
#
# Before plotting, we confirm that the pipeline is physically consistent on the model grid.
# The categorical class must stay on exact legend codes (nearest neighbor never blends);
# the conservative regrid must keep the eleven fractions inside `[0, 1]` and summing to one;
# and `f_veg` must equal the sum of the vegetated-class fractions (tree cover, shrubland,
# grassland, cropland, herbaceous wetland, and mangroves).

model_codes = unique(round.(Int, filter(!isnan, interior(landcover_class))))
@info "model-grid majority-class codes ⊆ legend (no invented codes): $(issubset(model_codes, class_codes))"

@info "Σ class fractions (model grid): extrema = $(extrema(filter(!isnan, interior(fraction_sum))))"
@info "f_veg (model grid): extrema = $(extrema(filter(!isnan, interior(vegetation_fraction)))), any NaN = $(any(isnan, interior(vegetation_fraction)))"

vegetated_codes = (10, 20, 30, 40, 90, 95) ## tree, shrub, grass, crop, herbaceous wetland, mangrove
vegetated_names = [name for name in class_names if legend[name][1] in vegetated_codes]
vegetated_fraction_sum = sum(fraction_fields[name] for name in vegetated_names)
@info "max |f_veg − Σ vegetated fractions| (model grid): $(maximum(abs, vegetation_fraction - vegetated_fraction_sum))"

# ## Majority land-cover class: native vs model grid
#
# On the native grid, the majority class is an exact legend code. On the model grid, it
# is the argmax of the conservatively regridded fractions (the class covering the most
# area of each cell), so it too stays on exact legend codes and agrees with the fraction
# fields. We map each code to its legend index so that the categorical palette and the
# legend line up.

function class_index(landcover_class)
    indices = map(interior(landcover_class)) do code
        isnan(code) && return NaN ## no-data cells (ocean or outside the coverage)
        i = findfirst(==(round(Int, code)), class_codes)
        isnothing(i) ? NaN : Float64(i)
    end
    index = CenterField(landcover_class.grid)
    return set!(index, indices)
end

native_class_index = class_index(native_landcover_class)
model_class_index  = class_index(landcover_class)

fig = Figure(size = (1180, 640), fontsize = 15)
for (col, (title, index)) in enumerate((("native ~110 m", native_class_index),
                                        ("model grid (area majority)", model_class_index)))
    ax = Axis(fig[1, col]; title = "Majority class ($title)", xlabel = "longitude", ylabel = "latitude")
    heatmap!(ax, index;
             colormap = cgrad(collect(class_colors), categorical = true),
             colorrange = (0.5, length(class_codes) + 0.5))
end
present = sort(unique(filter(!isnan, vcat(vec(interior(native_class_index)), vec(interior(model_class_index))))))
Legend(fig[1, 3],
       [PolyElement(color = class_colors[Int(i)]) for i in present],
       [replace(string(class_names[Int(i)]), "_" => " ") for i in present],
       "class"; framevisible = false)
save("esa_worldcover_landcover_class.png", fig)
fig

# ## Per-class area fractions
#
# We draw one panel per class that covers at least ~1% of the domain, each on the model
# grid with a shared 0–1 color scale. The vegetated classes cover most of the area;
# built-up land and water appear cleanly where the cities and the lakes are.

shown = [name for name in class_names if mean(fraction_fields[name]) > 0.01]
ncols = 3
nrows = cld(length(shown), ncols)
fig = Figure(size = (360 * ncols + 90, 300 * nrows), fontsize = 14)
for (k, name) in enumerate(shown)
    i, j = fldmod1(k, ncols)
    axis = Axis(fig[i, j]; title = replace(string(name), "_" => " "),
                xlabel = "longitude", ylabel = "latitude")
    heatmap!(axis, fraction_fields[name]; colormap = :viridis, colorrange = (0, 1))
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
hm = heatmap!(ax, vegetation_fraction; colormap = :YlGn, colorrange = (0, 1))
Colorbar(fig[1, 2], hm; label = "f_veg")
ax2 = Axis(fig[1, 3]; title = "distribution of f_veg", xlabel = "f_veg", ylabel = "cells")
hist!(ax2, vec(interior(vegetation_fraction)); bins = 30, color = (:seagreen, 0.8))
save("esa_worldcover_f_veg.png", fig)
fig

# ## Sum of fractions (wiring check) and native-vs-model comparison
#
# The eleven per-class fractions sum to ≈ 1 over land cells and to the mapped share
# along the coast (left). The aggregated pattern carries over from the native ~110 m
# grid to the model grid (right two panels): the conservative regrid area-averages
# but does not move the forest, the cities, or the lakes.

fig = Figure(size = (1500, 460), fontsize = 15)
ax = Axis(fig[1, 1]; title = "Σ class fractions (≈ 1)", xlabel = "longitude", ylabel = "latitude")
hm = heatmap!(ax, fraction_sum; colormap = :balance, colorrange = (0.95, 1.05))
Colorbar(fig[1, 2], hm)

ax = Axis(fig[1, 3]; title = "f_veg (native ~110 m)", xlabel = "longitude", ylabel = "latitude")
heatmap!(ax, native_vegetation_fraction; colormap = :YlGn, colorrange = (0, 1))
ax = Axis(fig[1, 4]; title = "f_veg (model grid)", xlabel = "longitude", ylabel = "latitude")
hm = heatmap!(ax, vegetation_fraction; colormap = :YlGn, colorrange = (0, 1))
Colorbar(fig[1, 5], hm; label = "f_veg")
save("esa_worldcover_fveg_check.png", fig)
fig
