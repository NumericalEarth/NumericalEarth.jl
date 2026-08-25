module WorldCover

export ESAWorldCover, WorldCoverVersion, WorldCoverV100, WorldCoverV200

using Downloads: Downloads
using KernelAbstractions: @kernel, @index
using Oceananigans: Center, location
using Oceananigans.Architectures: architecture, child_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: @root
using Oceananigans.Fields: Field, regrid!
using Oceananigans.Utils: launch!

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, BoundingBox, Dataset

import Oceananigans

"""
    ESA_WORLDCOVER_CLASS_NAMES

The ESA WorldCover legend: verbose class name → LCCS class code, in ascending
code order. [`ESAWorldCover`](@ref) tabulates the 11 classes.

The step between codes is not uniform (…90, 95, 100), so the codes are
enumerated rather than derived from a stride. `0` is the no-data code and is not
a valid class.
"""
const ESA_WORLDCOVER_CLASS_NAMES = (tree_cover              = 10,
                                    shrubland               = 20,
                                    grassland               = 30,
                                    cropland                = 40,
                                    built_up                = 50,
                                    bare_sparse_vegetation  = 60,
                                    snow_and_ice            = 70,
                                    permanent_water_bodies  = 80,
                                    herbaceous_wetland      = 90,
                                    mangroves               = 95,
                                    moss_and_lichen         = 100)

const ESA_WORLDCOVER_CLASS_CODES = values(ESA_WORLDCOVER_CLASS_NAMES)

"""
    ESA_WORLDCOVER_VEGETATED_CLASSES

The classes summed into the `:vegetation_fraction` variable.

Which classes count as vegetated is a modeling choice rather than a property of
the product. Bare/sparse vegetation (60) and moss/lichen (100) are excluded here
because a land model driven by this fraction treats them as bare ground. A
different partition needs no new download: every class also has its own
`:<class>_fraction` variable, so any vegetated set is the sum of the
corresponding fields.
"""
const ESA_WORLDCOVER_VEGETATED_CLASSES = (10, 20, 30, 40, 90, 95)

# 10 m ≈ 1°/12000. The integer pixel-per-degree count is the primitive: window
# snapping multiplies coordinates by it (exact) rather than dividing by the
# rounded native step (two roundings, unstable at exact pixel boundaries).
const ESA_WORLDCOVER_PIXELS_PER_DEGREE = 12000
const ESA_WORLDCOVER_NATIVE_STEP = 1 / ESA_WORLDCOVER_PIXELS_PER_DEGREE   # degrees

download_ESAWorldCover_cache::String = ""
function __init__()
    global download_ESAWorldCover_cache = DataWrangling.download_cache("ESAWorldCover")
end

"""
    WorldCoverVersion

The published ESA WorldCover releases: `WorldCoverV100` (2020) and
`WorldCoverV200` (2021).

The two use different classification algorithms and ESA warns they are **not**
comparable for change detection.
"""
@enum WorldCoverVersion WorldCoverV100 WorldCoverV200

# The release year and the token appearing in S3 keys and cache filenames.
function release_tokens(version::WorldCoverVersion)
    version === WorldCoverV100 && return (year = 2020, tag = "v100")
    version === WorldCoverV200 && return (year = 2021, tag = "v200")
    throw(ArgumentError("unhandled ESA WorldCover release $version"))
end

"""
    ESAWorldCover(; version = WorldCoverV200, aggregation_factor = 12)

ESA WorldCover global 10 m land-cover classification.

`version` selects the release, either `WorldCoverV200` (2021, the default) or
`WorldCoverV100` (2020); see [`WorldCoverVersion`](@ref).

`aggregation_factor` is the integer number of 10 m pixels reduced per side into
one aggregated cell (`12` → ~110 m, the default, so each aggregated cell still
samples ~144 sub-pixels at ~100 m model resolution; `120` → ~1 km, cheaper for a
large region). Class codes are never averaged: the fine raster is counted
block-wise into a majority class field, a vegetation-fraction field, and one
per-class area-fraction field. The integer factor keeps the aggregated cell
boundaries on fine-pixel boundaries, so the categorical field is never
reprojected.

The `Map` band is a `UInt8` class code and `0` is no-data. The legend
([`ESA_WORLDCOVER_CLASS_NAMES`](@ref)) is

| Code | Class                    | Code | Class                    |
|:-----|:-------------------------|:-----|:-------------------------|
| 10   | `tree_cover`             | 70   | `snow_and_ice`           |
| 20   | `shrubland`              | 80   | `permanent_water_bodies` |
| 30   | `grassland`              | 90   | `herbaceous_wetland`     |
| 40   | `cropland`               | 95   | `mangroves`              |
| 50   | `built_up`               | 100  | `moss_and_lichen`        |
| 60   | `bare_sparse_vegetation` |      |                          |

Because the source is a 10 m categorical raster, it is read in regional windows
only: build the `Metadatum` with a longitude/latitude [`BoundingBox`](@ref).
Available variables are `:landcover_class` (the majority class), a per-class
`:<class>_fraction` for each name above (e.g. `:cropland_fraction`), and
`:vegetation_fraction`, the mosaic weight `f_veg` summing tree cover,
shrubland, grassland, cropland, herbaceous wetland, and mangroves — see
[`ESA_WORLDCOVER_VEGETATED_CLASSES`](@ref) for why that set is a modeling
choice.

Reading the anonymous Cloud-Optimized GeoTIFF tiles from the public
`s3://esa-worldcover/` bucket requires the `ArchGDAL` package to be loaded
(`using ArchGDAL`).

```jldoctest
using NumericalEarth

ESAWorldCover()

# output
ESAWorldCover(version = WorldCoverV200, aggregation_factor = 12)
```

Data source: https://esa-worldcover.org/en/data-access ;
DOI v200 `10.5281/zenodo.7254221` (Zanaga et al. 2022), license CC-BY 4.0.
"""
struct ESAWorldCover <: AbstractStaticDataset
    version :: WorldCoverVersion
    aggregation_factor :: Int
end

function ESAWorldCover(; version = WorldCoverV200, aggregation_factor = 12)
    aggregation_factor ≥ 1 ||
        throw(ArgumentError("ESAWorldCover aggregation_factor must be a positive number of 10 m " *
                            "pixels per aggregated cell side, got $aggregation_factor"))
    return ESAWorldCover(version, aggregation_factor)
end

Base.summary(dataset::ESAWorldCover) =
    string("ESAWorldCover(version = ", dataset.version,
           ", aggregation_factor = ", dataset.aggregation_factor, ")")

Base.show(io::IO, dataset::ESAWorldCover) = print(io, summary(dataset))

version_year(dataset::ESAWorldCover) = release_tokens(dataset.version).year
version_string(dataset::ESAWorldCover) = release_tokens(dataset.version).tag

# The aggregated cell size (degrees) for this dataset.
aggregated_step(dataset::ESAWorldCover) = ESA_WORLDCOVER_NATIVE_STEP * dataset.aggregation_factor

#####
##### Variables — all derived from the raw `Map` byte band; they differ only in
##### the post-processing the ingest applies. Each variable name equals the band
##### name written into the materialized NetCDF, so `retrieve_data` reads it back
##### directly.
#####

"""
    class_fraction_variable_name(class_name)

The per-class area-fraction variable name for `class_name`
(e.g. `:cropland` → `:cropland_fraction`).
"""
class_fraction_variable_name(class_name::Symbol) = Symbol(class_name, :_fraction)

# The 11 per-class area-fraction variable names, one per class.
const ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES =
    map(class_fraction_variable_name, keys(ESA_WORLDCOVER_CLASS_NAMES))

function build_worldcover_variable_names()
    names = Dict{Symbol, String}(:landcover_class => "Map", :vegetation_fraction => "Map")
    for name in ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES
        names[name] = "Map"
    end
    return names
end

const ESAWorldCover_dataset_variable_names = build_worldcover_variable_names()

const ESAWorldCoverMetadatum = Metadatum{<:ESAWorldCover}

#####
##### Categorical aggregation. These operate on plain arrays of raw class codes
##### with no IO: `pixels` is the block of 10 m raster pixels covering one
##### aggregated cell — the data itself, not the legend. Every product comes
##### from one histogram over the legend, so no-data (`0`) and any code outside
##### the legend are excluded from all of them.
#####

"""
    class_counts(pixels)

Return the number of pixels of each legend class in `pixels`, as a `NamedTuple`
keyed by the verbose class names. Codes outside the legend — including the
no-data code `0` — are counted by no class, so the sum of the counts is the
number of valid pixels.

Every aggregated product is derived from these counts.
"""
class_counts(pixels::AbstractArray) =
    NamedTuple{keys(ESA_WORLDCOVER_CLASS_NAMES)}(map(c -> count(==(c), pixels),
                                                     ESA_WORLDCOVER_CLASS_CODES))

"""
    majority_class(pixels)

Return the class code covering the most of `pixels`, ignoring no-data. Ties break
toward the smaller code, and an all-no-data block returns `0`.

This is the aggregation used for the categorical `:landcover_class` product —
class codes are counted, never averaged.
"""
majority_class(pixels::AbstractArray) = majority_class(class_counts(pixels))

function majority_class(counts::NamedTuple)
    largest = 0
    class = 0
    for (code, n) in zip(ESA_WORLDCOVER_CLASS_CODES, values(counts))
        if n > largest
            largest = n
            class = code
        end
    end
    return class
end

"""
    class_fractions(pixels)

Return a `NamedTuple` of per-class area fractions (each in `[0, 1]`) over
`pixels`, keyed by the verbose class names. The fractions sum to 1 over valid
pixels, and to 0 when every pixel is no-data.
"""
class_fractions(pixels::AbstractArray) = class_fractions(class_counts(pixels))

function class_fractions(counts::NamedTuple)
    valid = sum(counts)
    fractions = map(n -> valid == 0 ? 0.0 : n / valid, values(counts))
    return NamedTuple{keys(ESA_WORLDCOVER_CLASS_NAMES)}(fractions)
end

"""
    vegetation_fraction(pixels; vegetated_classes = ESA_WORLDCOVER_VEGETATED_CLASSES)

Return the area fraction of `pixels` belonging to `vegetated_classes` — the
subgrid `f_veg`. See [`ESA_WORLDCOVER_VEGETATED_CLASSES`](@ref) for the default
set and why it is a modeling choice.
"""
vegetation_fraction(pixels::AbstractArray; kw...) = vegetation_fraction(class_counts(pixels); kw...)

function vegetation_fraction(counts::NamedTuple; vegetated_classes = ESA_WORLDCOVER_VEGETATED_CLASSES)
    valid = sum(counts)
    valid == 0 && return 0.0
    vegetated = sum(map((code, n) -> ifelse(code in vegetated_classes, n, 0),
                        ESA_WORLDCOVER_CLASS_CODES, values(counts)))
    return vegetated / valid
end

"""
    aggregate_blockwise(pixels, factor, reduction)

Reduce the fine 2-D `pixels` raster onto a coarse grid by an integer `factor`,
applying `reduction` (e.g. [`class_counts`](@ref) or [`majority_class`](@ref)) to
each non-overlapping `factor × factor` block. Integer-factor aggregation keeps
the coarse-cell boundaries aligned with fine-pixel boundaries — no reprojection
of the categorical field.

`size(pixels)` must be divisible by `factor` in both dimensions.
"""
function aggregate_blockwise(pixels::AbstractMatrix, factor::Integer, reduction)
    Nx, Ny = size(pixels)
    (Nx % factor == 0 && Ny % factor == 0) ||
        throw(ArgumentError("array size $(size(pixels)) is not divisible by the integer factor $factor"))

    nx, ny = Nx ÷ factor, Ny ÷ factor
    coarse = Array{typeof(reduction(view(pixels, 1:factor, 1:factor)))}(undef, nx, ny)
    for j in 1:ny, i in 1:nx
        block = view(pixels, (i - 1) * factor + 1 : i * factor,
                            (j - 1) * factor + 1 : j * factor)
        coarse[i, j] = reduction(block)
    end
    return coarse
end

"""
    aggregate_landcover(pixels, factor; vegetated_classes = ESA_WORLDCOVER_VEGETATED_CLASSES)

Reduce the fine `pixels` raster onto the coarse grid by an INTEGER `factor`,
returning `(; landcover_class, vegetation_fraction, class_fractions)`: the
majority class code, the vegetated area fraction `f_veg`, and a `NamedTuple` of
per-class area fractions keyed by the verbose class names. Each block is counted
once, with every product derived from the same [`class_counts`](@ref).
"""
function aggregate_landcover(pixels::AbstractMatrix, factor::Integer;
                             vegetated_classes = ESA_WORLDCOVER_VEGETATED_CLASSES)
    counts = aggregate_blockwise(pixels, factor, class_counts)
    fractions = map(class_fractions, counts)

    landcover_class = map(majority_class, counts)
    vegetation = map(c -> vegetation_fraction(c; vegetated_classes), counts)
    fraction_arrays = map(name -> map(f -> f[name], fractions),
                          keys(ESA_WORLDCOVER_CLASS_NAMES))
    class_fraction_fields = NamedTuple{keys(ESA_WORLDCOVER_CLASS_NAMES)}(fraction_arrays)

    return (; landcover_class, vegetation_fraction = vegetation,
              class_fractions = class_fraction_fields)
end

"""
    worldcover_window(longitude_bounds, latitude_bounds, factor)

Global native-pixel index bounds `(i₁, i₂, j₁, j₂)` of the read window covering a
region at aggregation `factor`. The window snaps to the global aggregated-cell
lattice and is padded by one aggregated cell on every side, guaranteeing it is a
strict superset of the native grid that `construct_native_grid` builds (whose
center-bracketing can extend up to one aggregated cell past the region edges).
The read-back maps grid cells to file cells by an integer offset with no
interpolation, so a narrower window would misregister the field by a whole cell.
"""
function worldcover_window(longitude_bounds, latitude_bounds, factor)
    pixels = ESA_WORLDCOVER_PIXELS_PER_DEGREE
    λ₁, λ₂ = longitude_bounds
    φ₁, φ₂ = latitude_bounds
    i₁ = factor * fld(floor(Int, λ₁ * pixels), factor) - factor
    i₂ = factor * cld(ceil( Int, λ₂ * pixels), factor) + factor
    j₁ = factor * fld(floor(Int, φ₁ * pixels), factor) - factor
    j₂ = factor * cld(ceil( Int, φ₂ * pixels), factor) + factor
    return i₁, i₂, j₁, j₂
end

#####
##### DataWrangling interface
#####

DataWrangling.available_variables(::ESAWorldCover) = ESAWorldCover_dataset_variable_names
DataWrangling.default_download_directory(dataset::ESAWorldCover) = download_ESAWorldCover_cache
DataWrangling.longitude_interfaces(::ESAWorldCover) = (-180, 180)
# WorldCover covers all land except Antarctica; northern limit ≈ 84°N.
DataWrangling.latitude_interfaces(::ESAWorldCover)  = (-60, 84)

# Global size at the aggregated resolution. A regional `BoundingBox` sub-windows
# this in `construct_native_grid`; only the window is ever materialized.
function Base.size(dataset::ESAWorldCover, variable)
    λ₁, λ₂ = DataWrangling.longitude_interfaces(dataset)
    φ₁, φ₂ = DataWrangling.latitude_interfaces(dataset)
    Δ = aggregated_step(dataset)
    Nx = round(Int, (λ₂ - λ₁) / Δ)
    Ny = round(Int, (φ₂ - φ₁) / Δ)
    return (Nx, Ny, 1)
end

# One materialization writes every band (majority class, vegetation fraction, and
# each per-class fraction) into a single NetCDF, so the filename is
# variable-independent: all variables of a region/factor share one cached file and
# it is fetched and aggregated only once.
function DataWrangling.metadata_filename(dataset::ESAWorldCover, name, date, region)
    return string("ESA_WorldCover_", version_string(dataset),
                  "_f", dataset.aggregation_factor, "_",
                  region_suffix(region), ".nc")
end

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::ESAWorldCoverMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("ESAWorldCover() must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:vegetation_fraction; dataset = ESAWorldCover(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

# Metadatum-level traits
DataWrangling.is_three_dimensional(::ESAWorldCoverMetadatum) = false
DataWrangling.dataset_variable_name(data::ESAWorldCoverMetadatum) =
    ESAWorldCover_dataset_variable_names[data.name]
DataWrangling.longitude_name(::ESAWorldCoverMetadatum) = "lon"
DataWrangling.latitude_name(::ESAWorldCoverMetadatum)  = "lat"
DataWrangling.default_inpainting(::ESAWorldCoverMetadatum) = nothing

# Float64 rather than the Float32 default: cell area on a latitude–longitude grid
# carries a `sin φ₂ - sin φ₁` factor, and at ~110 m spacing that subtraction loses
# most of Float32's precision, leaving the conservative regrid ~1e-4 short of
# conserving area (fractions drift off one and stray outside [0, 1]).
Base.eltype(::ESAWorldCoverMetadatum) = Float64

# `0` is the no-data code for the categorical `:landcover_class` product and is
# correctly masked to NaN on load. For the derived fraction products, `0.0` is a
# *legitimate* value (a water cell has zero vegetation fraction), so there is no
# in-band missing sentinel — use `NaN`, which never equals a real value and
# therefore masks nothing.
DataWrangling.missing_value(data::ESAWorldCoverMetadatum) =
    data.name === :landcover_class ? 0 : NaN

Oceananigans.Fields.location(::ESAWorldCoverMetadatum) = (Center, Center, Center)

#####
##### Regridding onto a model grid
#####
##### The fraction products ride Oceananigans' conservative (area-weighted)
##### `regrid!`, so each target cell carries the true area fraction of every class
##### and the fractions still sum to one. The categorical `:landcover_class` is the argmax of
##### those fractions — the class covering the most area of the target cell.
##### Bilinear interpolation of the codes would instead invent intermediate,
##### non-legend classes. Both extend the shared `interpolate_physical!` regrid
##### hook, dispatched on the metadatum.
#####

DataWrangling.default_regrid(::ESAWorldCoverMetadatum) = false

function DataWrangling.interpolate_physical!(target, native, metadata::ESAWorldCoverMetadatum)
    if metadata.name === :landcover_class
        majority_class_regrid!(target, metadata)
    else
        regrid!(target, native)
    end
    return target
end

# Majority class of each target cell: the class holding the largest conservatively
# regridded area fraction. This keeps the class consistent with the fraction
# fields and confined to the legend. Refining rather than coarsening needs no
# special case — the conservative regrid then reduces to the containing native
# cell, whose majority class the target inherits. Ties resolve to the lower code,
# matching `majority_class`, because the classes accumulate in ascending order.
function majority_class_regrid!(target, metadata)
    grid = target.grid
    arch = child_architecture(architecture(grid))
    LX, LY, LZ = location(metadata)

    fraction = Field{LX, LY, LZ}(grid)
    largest_fraction = Field{LX, LY, LZ}(grid)
    total_fraction = Field{LX, LY, LZ}(grid)

    fill!(target, 0)
    fill!(largest_fraction, 0)
    fill!(total_fraction, 0)

    for (name, code) in zip(ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES, ESA_WORLDCOVER_CLASS_CODES)
        fraction_metadatum = Metadatum(name; dataset = metadata.dataset,
                                       region = metadata.region, dir = metadata.dir)
        regrid!(fraction, Field(fraction_metadatum, arch))
        launch!(arch, grid, :xyz, _accumulate_majority_class!,
                target, largest_fraction, total_fraction, fraction, code)
    end

    launch!(arch, grid, :xyz, _mask_uncovered_class!, target, total_fraction)
    fill_halo_regions!(target)
    return target
end

@kernel function _accumulate_majority_class!(target, largest_fraction, total_fraction, fraction, code)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        f = fraction[i, j, k]
        larger = f > largest_fraction[i, j, k]
        largest_fraction[i, j, k] = ifelse(larger, f, largest_fraction[i, j, k])
        target[i, j, k] = ifelse(larger, convert(eltype(target), code), target[i, j, k])
        total_fraction[i, j, k] += f
    end
end

# Cells outside the product's coverage have no valid pixels, so every fraction is
# zero and no class holds a majority.
@kernel function _mask_uncovered_class!(target, total_fraction)
    i, j, k = @index(Global, NTuple)
    FT = eltype(target)
    @inbounds covered = total_fraction[i, j, k] > convert(FT, 1//2)
    @inbounds target[i, j, k] = ifelse(covered, target[i, j, k], convert(FT, NaN))
end

#####
##### Download / materialization
#####
##### The real fetch lives in ext/NumericalEarthArchGDALExt.jl: it reads the
##### anonymous COG tiles windowed to the bbox and aggregates them (via the pure
##### helpers above) onto the integer-factor lat/lon grid, writing a regional
##### NetCDF whose bands are named exactly as the metadatum variables. The module
##### entry point below fires a clear fallback error when ArchGDAL is not loaded.
#####

function Downloads.download(metadatum::ESAWorldCoverMetadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        worldcover_cog_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
worldcover_cog_to_netcdf(metadatum, nc_path) =
    error("Reading ESA WorldCover COG tiles requires the ArchGDAL package. " *
          "Load it with `using ArchGDAL`.")

# The materialized NetCDF stores every post-processed band under its variable
# name, so each variable reads back through the shared regrid path.
function DataWrangling.retrieve_data(metadata::ESAWorldCoverMetadatum)
    path = metadata_path(metadata)
    name = String(metadata.name)
    ds = Dataset(path)
    data = ds[name][:, :, 1]
    close(ds)
    return data
end

end # module WorldCover
