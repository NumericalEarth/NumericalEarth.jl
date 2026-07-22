module GHSL

export GHSBuiltH, GHSBuiltS

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, BoundingBox

import Oceananigans

download_GHSL_cache::String = ""
function __init__()
    global download_GHSL_cache = DataWrangling.download_cache("GHSL")
end

#####
##### Dataset types
#####

"""
    AbstractGHSLDataset

Supertype for the Global Human Settlement Layer (GHSL) R2023A built-up rasters
(European Commission JRC). All are distributed in the World Mollweide projection
(ESRI:54009) and so require a reprojection to EPSG:4326 in the read path — the one
ingestion difference from the geographic land rasters. Open access, no
authentication.

Data source: https://human-settlement.emergency.copernicus.eu
"""
abstract type AbstractGHSLDataset <: AbstractStaticDataset end

"""
    GHSBuiltH

GHS-BUILT-H R2023A average net building height (ANBH, epoch 2018, 100 m, World
Mollweide ESRI:54009). Provides `:building_height` — the mean height in metres of
the built-up pixels within each cell. A height of `0` over non-built land is a
valid value, not missing, so the field is not inpainted; only the product no-data
is masked to `NaN`.

Because it is a global 100 m product on a 1000 km Mollweide tile grid, it is read
in regional windows only: construct the `Metadatum` with a longitude/latitude
`BoundingBox`. The windowed tile download + Mollweide→EPSG:4326 warp is performed
by `ext/NumericalEarthArchGDALExt.jl` and requires `using ArchGDAL`.

Reference: Pesaresi, M. & Politis, P. (2023), *GHS-BUILT-H R2023A*, European
Commission JRC, doi:10.2905/85005901-3A49-48DD-9D19-6261354F56FE.

```jldoctest
julia> using NumericalEarth.DataWrangling.GHSL

julia> GHSBuiltH()
GHSBuiltH()
```
"""
struct GHSBuiltH <: AbstractGHSLDataset end

"""
    GHSBuiltS(; resolution = 100, epoch = 2020)

GHS-BUILT-S R2023A built-up surface (World Mollweide ESRI:54009). Provides
`:built_up_fraction` — the plan-area fraction `λp ∈ [0, 1]` of each cell covered by
buildings, obtained from the built-up surface (m² per cell) by dividing by the
native cell area and clamping to `[0, 1]`. A fraction of `0` over non-built land is
a valid value, not missing.

`resolution` selects the native pixel size in metres (`10` or `100`); `epoch` the
reference year. The 10 m product is only published for `epoch = 2018`; the 100 m
product covers 1975–2030 in 5-year steps. The default `100`/`2020` pairs naturally
with a ~100 m model grid and the epoch-2018 [`GHSBuiltH`](@ref) height; pass
`resolution = 10` for the finest built fraction (much larger tiles).

Read in regional windows only (see [`GHSBuiltH`](@ref)); the tile download +
Mollweide→EPSG:4326 warp requires `using ArchGDAL`.

Reference: Pesaresi, M. & Politis, P. (2023), *GHS-BUILT-S R2023A*, European
Commission JRC, doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA.

```jldoctest
julia> using NumericalEarth.DataWrangling.GHSL

julia> GHSBuiltS()
GHSBuiltS(resolution = 100 m, epoch = 2020)

julia> GHSBuiltS(resolution = 10)
GHSBuiltS(resolution = 10 m, epoch = 2018)
```
"""
struct GHSBuiltS <: AbstractGHSLDataset
    resolution :: Int
    epoch :: Int
end

function GHSBuiltS(; resolution = 100, epoch = resolution == 10 ? 2018 : 2020)
    resolution ∈ (10, 100) ||
        throw(ArgumentError("GHSBuiltS resolution must be 10 or 100 metres, got $resolution."))
    valid_epochs = resolution == 10 ? (2018,) : Tuple(1975:5:2030)
    epoch ∈ valid_epochs ||
        throw(ArgumentError("GHSBuiltS at $(resolution) m is only published for epochs " *
                            "$(valid_epochs), got $epoch."))
    return GHSBuiltS(resolution, epoch)
end

Base.summary(::GHSBuiltH) = "GHSBuiltH()"
Base.summary(dataset::GHSBuiltS) =
    string("GHSBuiltS(resolution = ", dataset.resolution, " m, epoch = ", dataset.epoch, ")")
Base.show(io::IO, dataset::AbstractGHSLDataset) = print(io, summary(dataset))

const GHSBuiltHMetadatum = Metadatum{<:GHSBuiltH}
const GHSBuiltSMetadatum = Metadatum{<:GHSBuiltS}
const GHSLMetadatum      = Metadatum{<:AbstractGHSLDataset}

#####
##### Variables
#####

GHSBuiltH_variable_names = Dict(:building_height   => "ANBH")
GHSBuiltS_variable_names = Dict(:built_up_fraction => "built_up_fraction")

DataWrangling.available_variables(::GHSBuiltH) = GHSBuiltH_variable_names
DataWrangling.available_variables(::GHSBuiltS) = GHSBuiltS_variable_names

DataWrangling.dataset_variable_name(data::GHSBuiltHMetadatum) = GHSBuiltH_variable_names[data.name]
DataWrangling.dataset_variable_name(data::GHSBuiltSMetadatum) = GHSBuiltS_variable_names[data.name]

#####
##### Dataset interface
#####

DataWrangling.default_download_directory(::AbstractGHSLDataset) = download_GHSL_cache

# The reprojected regional raster is on a plain lat/lon grid; the native hull is the
# global lat/lon extent and the shared regrid restricts it to the BoundingBox.
DataWrangling.longitude_interfaces(::AbstractGHSLDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::AbstractGHSLDataset)  = (-90, 90)

# Native pixel size of each product in metres (sets the windowed-read target Δ = 360/Nx).
native_resolution(::GHSBuiltH) = 100
native_resolution(dataset::GHSBuiltS) = dataset.resolution

# Nominal global native size in EPSG:4326 (only the windowed portion is materialized);
# 1° ≈ 111320 m at the equator sets the degree pixel size Δ = resolution / 111320.
function Base.size(dataset::AbstractGHSLDataset, variable)
    Δ = native_resolution(dataset) / 111320
    Nx = round(Int, 360 / Δ)
    Ny = round(Int, 180 / Δ)
    return (Nx, Ny, 1)
end

dataset_prefix(::GHSBuiltH) = "GHSBuiltH"
dataset_prefix(dataset::GHSBuiltS) =
    string("GHSBuiltS_", dataset.resolution, "m_", dataset.epoch)

DataWrangling.metadata_filename(dataset::AbstractGHSLDataset, name, date, region) =
    string(dataset_prefix(dataset), "_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::GHSLMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        prefix = dataset_prefix(metadata.dataset)
        error("$(summary(metadata.dataset)) must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:$(metadata.name); dataset = $(summary(metadata.dataset)),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

#####
##### Metadatum interface
#####

DataWrangling.is_three_dimensional(::GHSLMetadatum) = false

# The regional NetCDF we materialize stores coordinates as "lon"/"lat".
DataWrangling.longitude_name(::GHSLMetadatum) = "lon"
DataWrangling.latitude_name(::GHSLMetadatum)  = "lat"

# NEVER inpaint: a building height / built fraction of 0 over non-built land is a
# valid value, not a gap. The no-data is masked to NaN in the warp step.
DataWrangling.default_inpainting(::GHSLMetadatum) = nothing

Oceananigans.Fields.location(::GHSLMetadatum) = (Center, Center, Center)

#####
##### GHSL World-Mollweide (ESRI:54009) tile grid — pure, dependency-free
#####
#####
##### R2023A tiles a global Mollweide grid of 18 rows × 36 columns of 1000 km
##### squares, indexed `R{row}_C{col}` from the upper-left (NW) corner. Selecting the
##### tiles intersecting a lat/lon window needs the forward Mollweide projection and
##### the tile-index arithmetic below; the tile download + warp itself lives in the
##### ArchGDAL extension.
#####

# ESRI:54009 World Mollweide is defined on a sphere of the WGS84 semi-major radius.
const MOLLWEIDE_RADIUS = 6378137.0

# GHSL tile grid (verified against the JRC tiling schema): origin at the upper-left
# corner, 1000 km square tiles, capped at 36 columns / 18 rows.
const GHSL_TILE_SIZE    = 1_000_000.0
const GHSL_ORIGIN_X     = -18_041_000.0
const GHSL_ORIGIN_Y     =   9_000_000.0
const GHSL_COLUMNS      = 36
const GHSL_ROWS         = 18

"""
    longitude_latitude_to_mollweide(longitude, latitude)

Forward World-Mollweide (ESRI:54009) projection: map `(longitude, latitude)` in
degrees (central meridian 0°) to `(x, y)` in metres. The auxiliary angle `θ` solves
`2θ + sin 2θ = π sin(latitude)` by Newton iteration.
"""
function longitude_latitude_to_mollweide(longitude, latitude)
    λ = deg2rad(longitude)
    φ = deg2rad(latitude)
    θ = φ  # initial guess
    if abs(φ) < π / 2 - eps(φ)
        for _ in 1:20
            θ -= (2θ + sin(2θ) - π * sin(φ)) / (2 + 2cos(2θ))
        end
    else
        θ = copysign(π / 2, φ)
    end
    x = MOLLWEIDE_RADIUS * (2sqrt(2) / π) * λ * cos(θ)
    y = MOLLWEIDE_RADIUS * sqrt(2) * sin(θ)
    return x, y
end

"""
    ghsl_tile_index(longitude, latitude)

Return the GHSL R2023A `(row, column)` tile index (1-based, from the NW corner)
whose 1000 km Mollweide cell contains `(longitude, latitude)`.

```jldoctest
julia> using NumericalEarth.DataWrangling.GHSL: ghsl_tile_index

julia> ghsl_tile_index(2.35, 48.85)   # Paris
(4, 19)
```
"""
function ghsl_tile_index(longitude, latitude)
    x, y = longitude_latitude_to_mollweide(longitude, latitude)
    column = clamp(floor(Int, (x - GHSL_ORIGIN_X) / GHSL_TILE_SIZE) + 1, 1, GHSL_COLUMNS)
    row    = clamp(floor(Int, (GHSL_ORIGIN_Y - y) / GHSL_TILE_SIZE) + 1, 1, GHSL_ROWS)
    return row, column
end

"""
    ghsl_tiles_in_bbox(region::BoundingBox)

Return the sorted `(row, column)` GHSL tiles whose 1000 km Mollweide cells intersect
`region`. A lat/lon window maps to a curved quadrilateral in Mollweide, so rather than
point-sample (which silently skips tiles once a window spans more than a few), take the
window's Mollweide `(x, y)` bounding box and return every tile in the corresponding index
range: `y` is monotone in latitude, and `x` is linear in longitude and scales with `cos θ`
(largest at the equator-most latitude), so the extremes are attained among the corners and
the equator-most edge. Exact for regional windows; a complete (never-missing) superset for
any window.
"""
function ghsl_tiles_in_bbox(region::BoundingBox)
    λ₁, λ₂ = region.longitude
    φ₁, φ₂ = region.latitude
    φ_maxx = clamp(0, min(φ₁, φ₂), max(φ₁, φ₂))
    xmin = ymin =  Inf
    xmax = ymax = -Inf
    for φ in (φ₁, φ₂, φ_maxx), λ in (λ₁, λ₂)
        x, y = longitude_latitude_to_mollweide(λ, φ)
        xmin = min(xmin, x); xmax = max(xmax, x)
        ymin = min(ymin, y); ymax = max(ymax, y)
    end
    col_lo = clamp(floor(Int, (xmin - GHSL_ORIGIN_X) / GHSL_TILE_SIZE) + 1, 1, GHSL_COLUMNS)
    col_hi = clamp(floor(Int, (xmax - GHSL_ORIGIN_X) / GHSL_TILE_SIZE) + 1, 1, GHSL_COLUMNS)
    row_lo = clamp(floor(Int, (GHSL_ORIGIN_Y - ymax) / GHSL_TILE_SIZE) + 1, 1, GHSL_ROWS)
    row_hi = clamp(floor(Int, (GHSL_ORIGIN_Y - ymin) / GHSL_TILE_SIZE) + 1, 1, GHSL_ROWS)
    return [(row, col) for row in row_lo:row_hi for col in col_lo:col_hi]
end

#####
##### Download-URL construction (JRC jeodpp open-data HTTPS)
#####

const GHSL_FTP = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"

# The product-folder / file basename stem shared by a dataset's tiles and mosaic.
product_stem(::GHSBuiltH) = "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100"
product_stem(dataset::GHSBuiltS) =
    string("GHS_BUILT_S_E", dataset.epoch, "_GLOBE_R2023A_54009_", dataset.resolution)

product_group(::GHSBuiltH) = "GHS_BUILT_H_GLOBE_R2023A"
product_group(::GHSBuiltS) = "GHS_BUILT_S_GLOBE_R2023A"

"""
    ghsl_tile_url(dataset, row, column)

Return the JRC open-data HTTPS URL of the `(row, column)` GHSL tile `.zip` for
`dataset` (each archive holds one Mollweide GeoTIFF of the built-up variable).
"""
function ghsl_tile_url(dataset::AbstractGHSLDataset, row, column)
    stem = product_stem(dataset)
    return string(GHSL_FTP, "/", product_group(dataset), "/", stem,
                  "/V1-0/tiles/", stem, "_V1_0_R", row, "_C", column, ".zip")
end

"""
    ghsl_tile_urls(dataset, region::BoundingBox)

`.zip` URLs of the GHSL tiles for `dataset` intersecting `region`, ready to be
downloaded, unzipped, and mosaicked/reprojected by GDAL.
"""
ghsl_tile_urls(dataset::AbstractGHSLDataset, region::BoundingBox) =
    [ghsl_tile_url(dataset, row, column) for (row, column) in ghsl_tiles_in_bbox(region)]

# GeoTIFF basename inside a tile archive: the stem with the same tile token, `.tif`.
ghsl_tile_tif_name(dataset::AbstractGHSLDataset, row, column) =
    string(product_stem(dataset), "_V1_0_R", row, "_C", column, ".tif")

#####
##### Pure no-data masking + built-surface → fraction conversion (unit-testable core)
#####

"""
    mask_building_height(value)

Map a GHS-BUILT-H ANBH `value` (metres) to a masked `Float64`, keeping every physical
height (including the legitimate non-built value `0`). GHSL declares its no-data as a
positive sentinel (`255`) that the Mollweide→EPSG:4326 warp has already written as `NaN`;
this pass carries that gap through and defensively maps any negative to `NaN`.

```jldoctest
julia> using NumericalEarth.DataWrangling.GHSL: mask_building_height

julia> mask_building_height(12.5)
12.5

julia> mask_building_height(0)
0.0

julia> isnan(mask_building_height(NaN))   # the warp writes the no-data gap as NaN
true
```
"""
@inline mask_building_height(value) =
    ifelse((value < 0) | !isfinite(value), oftype(float(value), NaN), float(value))

"""
    built_surface_to_fraction(surface, cell_area)

Convert a GHS-BUILT-S built-up `surface` (m² of buildings within a native cell) to
the plan-area fraction `λp = surface / cell_area`, clamped to `[0, 1]`. `cell_area` is the
native cell area in m² (`100 × 100` at 100 m, `10 × 10` at 10 m). GHSL declares its no-data
as a positive sentinel (`65535` at 100 m, `255` at 10 m) that the warp excludes from the
bilinear blend and writes as `NaN`, so a non-finite `surface` (or a spurious negative) maps
to `NaN` without a no-data cell ever contaminating a resampled mean.

```jldoctest
julia> using NumericalEarth.DataWrangling.GHSL: built_surface_to_fraction

julia> built_surface_to_fraction(2500.0, 10_000.0)   # 25% built
0.25

julia> built_surface_to_fraction(10_000.0, 10_000.0)  # fully built, clamped
1.0

julia> isnan(built_surface_to_fraction(-1.0, 10_000.0))
true
```
"""
@inline function built_surface_to_fraction(surface, cell_area)
    S = float(surface)
    return ifelse((S < 0) | !isfinite(S), oftype(S, NaN), clamp(S / cell_area, zero(S), one(S)))
end

#####
##### Download (regional Mollweide tiles → reprojected NetCDF via the ArchGDAL ext)
#####

function Downloads.download(metadatum::GHSLMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        ghsl_tiles_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded. The
# fallback below fires only when the extension is not active (mirrors
# CopernicusDEM.zarr_to_netcdf / CanopyHeight.canopy_height_cog_to_netcdf).
ghsl_tiles_to_netcdf(metadatum, nc_path) =
    error("Reading the $(summary(metadatum.dataset)) Mollweide GeoTIFF tiles requires " *
          "the ArchGDAL package (for the ESRI:54009 → EPSG:4326 reprojection). " *
          "Load it with `using ArchGDAL`.")

end # module GHSL
