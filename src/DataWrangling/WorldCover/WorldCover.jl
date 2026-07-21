module WorldCover

export ESAWorldCover

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, BoundingBox, Dataset

import Oceananigans

#####
##### Class legend (LCCS codes). Steps are NON-uniform (…90, 95, 100), so the
##### codes are enumerated explicitly rather than assuming a regular stride.
##### No-data is 0 (0 is not a valid class; valid classes start at 10).
#####

"""
    ESA_WORLDCOVER_MISSING_VALUE

The no-data code used by ESA WorldCover. `0` is not a valid land-cover class,
so it flags pixels outside coverage and is masked before any aggregation.
"""
const ESA_WORLDCOVER_MISSING_VALUE = 0

"""
    ESA_WORLDCOVER_CLASS_NAMES

`NamedTuple` mapping each verbose class name to its ESA WorldCover code. The
step near the top of the legend is non-uniform (…90, 95, 100).
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

"""
    ESA_WORLDCOVER_CLASS_CODES

The 11 ESA WorldCover land-cover class codes, in ascending order.
"""
const ESA_WORLDCOVER_CLASS_CODES = values(ESA_WORLDCOVER_CLASS_NAMES)

"""
    ESA_WORLDCOVER_VEGETATED_CLASSES

The class codes counted as vegetated when forming `vegetation_fraction`:
tree cover (10), shrubland (20), grassland (30), cropland (40), herbaceous
wetland (90), and mangroves (95).

Whether bare/sparse vegetation (60) and moss/lichen (100) count as vegetated is
a modeling choice; they are excluded here. This set is exposed deliberately so a
model can override the vegetation definition through the `vegetated_classes`
argument of [`vegetation_fraction`](@ref).
"""
const ESA_WORLDCOVER_VEGETATED_CLASSES = (10, 20, 30, 40, 90, 95)

#####
##### Native / aggregated grid geometry
#####
##### WorldCover is a 10 m (1°/12000) EPSG:4326 raster. Aggregating class codes
##### bilinearly is meaningless, so the ingest reduces the fine raster onto a
##### coarser lat/lon grid by an INTEGER factor (keeping pixel boundaries
##### aligned — no reprojection of the categorical field). That aggregated grid
##### is the dataset's native grid for the DataWrangling machinery, and the
##### derived continuous fraction fields ride the normal bilinear regrid onto a
##### model grid. `aggregation_factor` sets the aggregated resolution; the
##### default of 12 (~110 m) matches ~100 m regional runs so each aggregated
##### cell still samples ~144 sub-pixels.
#####

const ESA_WORLDCOVER_NATIVE_STEP = 1 / 12000   # 10 m in degrees

# WorldCover covers all land except Antarctica; northern limit ≈ 84°N.
const ESA_WORLDCOVER_LONGITUDE_INTERFACES = (-180, 180)
const ESA_WORLDCOVER_LATITUDE_INTERFACES  = (-60, 84)

download_ESAWorldCover_cache::String = ""
function __init__()
    global download_ESAWorldCover_cache = DataWrangling.download_cache("ESAWorldCover")
end

"""
    ESAWorldCover(; version = :v200, aggregation_factor = 12)

ESA WorldCover global 10 m land-cover classification.

`version` selects the release: `:v200` (2021, the default) or `:v100` (2020).
The two versions use different algorithms and ESA warns they are **not**
comparable for change detection — pick one.

`aggregation_factor` is the integer number of 10 m pixels averaged per side into
one aggregated cell (`12` → ~110 m, the default; `120` → ~1 km, cheaper for a
large region). Class codes are never averaged: the fine raster is reduced
block-wise into a majority class field, a vegetation-fraction field, and one
per-class area-fraction field.

The `Map` band is a `UInt8` class code (see [`ESA_WORLDCOVER_CLASS_NAMES`](@ref));
no-data is `0`. Because the source is a 10 m categorical raster, it is read in
regional windows only: build the `Metadatum` with a longitude/latitude
[`BoundingBox`](@ref). Available variables are `:landcover_class` (majority),
`:vegetation_fraction` (the mosaic weight `f_veg`), and a per-class
`:<class>_fraction` for each name in [`ESA_WORLDCOVER_CLASS_NAMES`](@ref)
(e.g. `:cropland_fraction`).

Reading the anonymous Cloud-Optimized GeoTIFF tiles from the public
`s3://esa-worldcover/` bucket requires the `ArchGDAL` package to be loaded
(`using ArchGDAL`).

```jldoctest
using NumericalEarth

ESAWorldCover()

# output
ESAWorldCover(version = :v200, aggregation_factor = 12)
```

Data source: https://esa-worldcover.org/en/data-access ;
DOI v200 `10.5281/zenodo.7254221` (Zanaga et al. 2022), license CC-BY 4.0.
"""
struct ESAWorldCover <: AbstractStaticDataset
    version :: Symbol
    aggregation_factor :: Int
end

ESAWorldCover(; version = :v200, aggregation_factor = 12) =
    ESAWorldCover(version, aggregation_factor)

Base.show(io::IO, dataset::ESAWorldCover) =
    print(io, "ESAWorldCover(version = :", dataset.version,
          ", aggregation_factor = ", dataset.aggregation_factor, ")")

# Release-specific tokens used to build S3 keys and cache filenames.
version_year(dataset::ESAWorldCover) = dataset.version === :v100 ? 2020 : 2021
version_string(dataset::ESAWorldCover) = String(dataset.version)

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

"""
    ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES

The 11 per-class area-fraction variable names, one per entry of
[`ESA_WORLDCOVER_CLASS_NAMES`](@ref).
"""
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
##### Pure aggregation helpers (the main, unit-testable deliverable).
##### These operate on plain arrays of `UInt8` (or `Integer`) class codes with
##### NO IO. `codes` is a block of fine 10 m pixels covering one coarse cell.
##### No-data (0) pixels are excluded from every count.
#####

"""
    mode_aggregate(codes)

Return the majority (mode) class code over `codes`, ignoring no-data (`0`)
pixels. Ties break toward the smaller code (deterministic). Returns `0` when
every pixel is no-data.

This is the aggregation used for the categorical `:landcover_class` product —
class codes are counted, never averaged.
"""
function mode_aggregate(codes)
    best_code = 0
    best_count = 0
    for c in ESA_WORLDCOVER_CLASS_CODES
        n = count(==(c), codes)
        if n > best_count
            best_count = n
            best_code = c
        end
    end
    return best_code
end

"""
    class_fraction(codes, c)

Return the area fraction of class `c` over `codes`, the count of pixels equal to
`c` divided by the count of valid (non-`0`) pixels. Returns `0.0` when there are
no valid pixels.
"""
function class_fraction(codes, c)
    valid = count(!=(ESA_WORLDCOVER_MISSING_VALUE), codes)
    valid == 0 && return 0.0
    return count(==(c), codes) / valid
end

"""
    class_fractions(codes; class_names = ESA_WORLDCOVER_CLASS_NAMES)

Return a `NamedTuple` of per-class area fractions (each in `[0, 1]`) over
`codes`, keyed by the verbose class names. Over valid (non-`0`) pixels the
fractions sum to 1 (they sum to 0 when every pixel is no-data).
"""
function class_fractions(codes; class_names = ESA_WORLDCOVER_CLASS_NAMES)
    fractions = map(c -> class_fraction(codes, c), values(class_names))
    return NamedTuple{keys(class_names)}(fractions)
end

"""
    vegetation_fraction(codes; vegetated_classes = ESA_WORLDCOVER_VEGETATED_CLASSES)

Return the fraction of valid (non-`0`) pixels in `codes` belonging to the
`vegetated_classes` set — the subgrid `f_veg`. The vegetated-class set is a
modeling choice and is passed as an argument so it can be overridden.
"""
function vegetation_fraction(codes; vegetated_classes = ESA_WORLDCOVER_VEGETATED_CLASSES)
    valid = count(!=(ESA_WORLDCOVER_MISSING_VALUE), codes)
    valid == 0 && return 0.0
    vegetated = count(c -> c in vegetated_classes, codes)
    return vegetated / valid
end

"""
    aggregate_blockwise(codes, factor, reduction)

Reduce the fine 2-D `codes` raster onto a coarse grid by an INTEGER `factor`,
applying `reduction` (e.g. [`mode_aggregate`](@ref) or a
`block -> class_fraction(block, c)` closure) to each non-overlapping
`factor × factor` block. Integer-factor aggregation keeps the coarse-cell
boundaries aligned with fine-pixel boundaries — no reprojection of the
categorical field.

`size(codes)` must be divisible by `factor` in both dimensions.
"""
function aggregate_blockwise(codes::AbstractMatrix, factor::Integer, reduction)
    Nx, Ny = size(codes)
    (Nx % factor == 0 && Ny % factor == 0) ||
        throw(ArgumentError("array size $(size(codes)) is not divisible by the integer factor $factor"))

    nx, ny = Nx ÷ factor, Ny ÷ factor
    coarse = Array{typeof(reduction(view(codes, 1:factor, 1:factor)))}(undef, nx, ny)
    for j in 1:ny, i in 1:nx
        block = view(codes, (i - 1) * factor + 1 : i * factor,
                            (j - 1) * factor + 1 : j * factor)
        coarse[i, j] = reduction(block)
    end
    return coarse
end

#####
##### DataWrangling interface
#####

DataWrangling.available_variables(::ESAWorldCover) = ESAWorldCover_dataset_variable_names
DataWrangling.default_download_directory(dataset::ESAWorldCover) = download_ESAWorldCover_cache
DataWrangling.longitude_interfaces(::ESAWorldCover) = ESA_WORLDCOVER_LONGITUDE_INTERFACES
DataWrangling.latitude_interfaces(::ESAWorldCover)  = ESA_WORLDCOVER_LATITUDE_INTERFACES

# Global size at the aggregated resolution. A regional `BoundingBox` sub-windows
# this in `construct_native_grid`; only the window is ever materialized.
function Base.size(dataset::ESAWorldCover, variable)
    λ₁, λ₂ = ESA_WORLDCOVER_LONGITUDE_INTERFACES
    φ₁, φ₂ = ESA_WORLDCOVER_LATITUDE_INTERFACES
    Δ = aggregated_step(dataset)
    Nx = round(Int, (λ₂ - λ₁) / Δ)
    Ny = round(Int, (φ₂ - φ₁) / Δ)
    return (Nx, Ny, 1)
end

function DataWrangling.metadata_filename(dataset::ESAWorldCover, name, date, region)
    return string("ESA_WorldCover_", version_string(dataset),
                  "_f", dataset.aggregation_factor, "_",
                  name, "_", region_suffix(region), ".nc")
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

# `0` is the no-data code for the categorical `:landcover_class` product and is
# correctly masked to NaN on load. For the derived fraction products, `0.0` is a
# *legitimate* value (a water cell has zero vegetation fraction), so there is no
# in-band missing sentinel — use `NaN`, which never equals a real value and
# therefore masks nothing.
DataWrangling.missing_value(data::ESAWorldCoverMetadatum) =
    data.name === :landcover_class ? ESA_WORLDCOVER_MISSING_VALUE : NaN

Oceananigans.Fields.location(::ESAWorldCoverMetadatum) = (Center, Center, Center)

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
