module GHSL

export GHSBuiltH, GHSBuiltS, GHSBuiltSResolution, GHSBuiltS10m, GHSBuiltS100m, GHSBuilt

using Downloads: Downloads
using Oceananigans: Center, CPU, Face
using Oceananigans.DistributedComputations: @root
using Oceananigans.Grids: x_domain, y_domain, λnodes, φnodes

using NumericalEarth.Lands: Lands

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, native_grid, BoundingBox, bounding_box_suffix

using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES

import Oceananigans

download_GHSL_cache::String = ""
function __init__()
    global download_GHSL_cache = DataWrangling.download_cache("GHSL")
    return nothing
end

#####
##### Dataset types
#####

"""
    AbstractGHSLDataset

Supertype for the Global Human Settlement Layer (GHSL) R2023A built-up rasters
(European Commission JRC) and their [`GHSBuilt`](@ref) pairing. The rasters are
distributed in the World Mollweide projection (ESRI:54009) and so require a
reprojection to EPSG:4326 in the read path. Open access, no authentication.

Data source: https://human-settlement.emergency.copernicus.eu
"""
abstract type AbstractGHSLDataset <: AbstractStaticDataset end

"""
    GHSBuiltH

GHS-BUILT-H R2023A average net building height (ANBH, epoch 2018, 100 m, World
Mollweide ESRI:54009). Provides `:building_height` — the mean height in meters of
the built-up pixels within each cell. A height of `0` over non-built land is a
valid value, not missing, so the field is not inpainted. Only the product no-data
is masked to `NaN`.

Because it is a global 100 m product on a 1000 km Mollweide tile grid, it is read
in regional windows only: construct the `Metadatum` with a longitude/latitude
`BoundingBox`. The windowed tile download and Mollweide→EPSG:4326 warp require
`using ArchGDAL`.

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
    GHSBuiltSResolution

The two published GHS-BUILT-S native pixel sizes: `GHSBuiltS10m` (10 m) and
`GHSBuiltS100m` (100 m).
"""
@enum GHSBuiltSResolution GHSBuiltS10m GHSBuiltS100m

"""
    resolution_meters(resolution::GHSBuiltSResolution)

The native pixel size in meters of a [`GHSBuiltSResolution`](@ref).
"""
function resolution_meters(resolution::GHSBuiltSResolution)
    resolution === GHSBuiltS10m  && return 10
    resolution === GHSBuiltS100m && return 100
    throw(ArgumentError("unhandled GHS-BUILT-S resolution $resolution"))
end

"""
    GHSBuiltS(; resolution = GHSBuiltS100m, epoch = 2020)

GHS-BUILT-S R2023A built-up surface (World Mollweide ESRI:54009). Provides
`:built_up_fraction` — the plan-area index `λᵖ ∈ [0, 1]` of each cell covered by
buildings, obtained from the built-up surface (m² per cell) by dividing by the
native cell area and clamping to `[0, 1]`. A fraction of `0` over non-built land is
a valid value, not missing.

`resolution` selects the native pixel size, `GHSBuiltS10m` or `GHSBuiltS100m` (see
[`GHSBuiltSResolution`](@ref)); `epoch` the reference year. The 10 m product is only
published for `epoch = 2018`; the 100 m product covers 1975–2030 in 5-year steps.
The default is `GHSBuiltS100m`/`2020`; pass `resolution = GHSBuiltS10m` for the
epoch-2018 built fraction (much larger tiles).

Read in regional windows only (see [`GHSBuiltH`](@ref)); the tile download and
Mollweide→EPSG:4326 warp require `using ArchGDAL`.

Reference: Pesaresi, M. & Politis, P. (2023), *GHS-BUILT-S R2023A*, European
Commission JRC, doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA.

```jldoctest
julia> using NumericalEarth.DataWrangling.GHSL

julia> GHSBuiltS()
GHSBuiltS(resolution = GHSBuiltS100m, epoch = 2020)

julia> GHSBuiltS(resolution = GHSBuiltS10m)
GHSBuiltS(resolution = GHSBuiltS10m, epoch = 2018)
```
"""
struct GHSBuiltS <: AbstractGHSLDataset
    resolution :: GHSBuiltSResolution
    epoch :: Int
end

function GHSBuiltS(; resolution = GHSBuiltS100m,
                     epoch = resolution === GHSBuiltS10m ? 2018 : 2020)
    valid_epochs = resolution === GHSBuiltS10m ? (2018,) : Tuple(1975:5:2030)
    epoch ∈ valid_epochs ||
        throw(ArgumentError("GHSBuiltS at $(resolution_meters(resolution)) m is only published " *
                            "for epochs $(valid_epochs), got $epoch."))
    return GHSBuiltS(resolution, epoch)
end

Base.summary(::GHSBuiltH) = "GHSBuiltH()"
Base.summary(dataset::GHSBuiltS) =
    string("GHSBuiltS(resolution = ", dataset.resolution, ", epoch = ", dataset.epoch, ")")
Base.show(io::IO, dataset::AbstractGHSLDataset) = print(io, summary(dataset))

"""
$(TYPEDEF)

The GHSL built-up pair that [`building_morphometry`](@ref NumericalEarth.Lands.building_morphometry) reads together: the built-up
`surface` ([`GHSBuiltS`](@ref)) and the building `height` ([`GHSBuiltH`](@ref)).
Provides `:built_up_fraction` and `:building_volume` (m³ per m²) on a common grid at
the surface product's resolution: the volume of each surface pixel is its built fraction
times the height of the 100 m height pixel containing it. Built area and volume are
conservatively remapped onto the regional longitude/latitude grid, where built area under a
missing or zero height takes the mean height of the cell's built area with a height; a cell with
built area but no heights has `NaN` volume. `:measured_built_fraction` and
`:measured_building_volume` retain the area and volume of pixels with positive heights.
Heights are always the epoch-2018 ANBH, whatever the surface epoch. The default surface is the 10 m,
epoch-2018 product, matching the height epoch.

```jldoctest
julia> using NumericalEarth

julia> GHSBuilt()
GHSBuilt(GHSBuiltS(resolution = GHSBuiltS10m, epoch = 2018), GHSBuiltH())

julia> GHSBuilt(surface = GHSBuiltS())
GHSBuilt(GHSBuiltS(resolution = GHSBuiltS100m, epoch = 2020), GHSBuiltH())
```
"""
struct GHSBuilt <: AbstractGHSLDataset
    surface :: GHSBuiltS
    height  :: GHSBuiltH
end

GHSBuilt(; surface = GHSBuiltS(resolution = GHSBuiltS10m)) = GHSBuilt(surface, GHSBuiltH())

Base.summary(dataset::GHSBuilt) = string("GHSBuilt(", summary(dataset.surface), ", ", summary(dataset.height), ")")

const GHSLMetadatum = Metadatum{<:AbstractGHSLDataset}

# The 10 m built-up rasters are large and regridded by window rather than whole.
DataWrangling.windowed_retrieval(::AbstractGHSLDataset) = true

#####
##### Variables
#####

GHSBuiltH_variable_names = Dict(:building_height   => "ANBH")
GHSBuiltS_variable_names = Dict(:built_up_fraction => "built_up_fraction")
GHSBuilt_variable_names = Dict(:built_up_fraction => "built_up_fraction", :building_volume => "building_volume",
                              :measured_built_fraction => "measured_built_fraction",
                              :measured_building_volume => "measured_building_volume")

DataWrangling.available_variables(::GHSBuiltH) = GHSBuiltH_variable_names
DataWrangling.available_variables(::GHSBuiltS) = GHSBuiltS_variable_names
DataWrangling.available_variables(::GHSBuilt) = GHSBuilt_variable_names

DataWrangling.dataset_variable_name(data::GHSLMetadatum) = DataWrangling.available_variables(data.dataset)[data.name]

#####
##### Dataset interface
#####

DataWrangling.default_download_directory(::AbstractGHSLDataset) = download_GHSL_cache

# Global extent even though reads are windowed: the regrid restricts to the BoundingBox.
DataWrangling.longitude_interfaces(::AbstractGHSLDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::AbstractGHSLDataset)  = (-90, 90)

# Native pixel size in meters.
native_resolution(::GHSBuiltH) = 100
native_resolution(dataset::GHSBuiltS) = resolution_meters(dataset.resolution)
native_resolution(dataset::GHSBuilt) = native_resolution(dataset.surface)

# Nominal global size in EPSG:4326 (only the windowed portion is ever materialized);
# 1° ≈ 111320 m at the equator sets the degree pixel size.
function Base.size(dataset::AbstractGHSLDataset, variable)
    Δ = native_resolution(dataset) / 111320
    Nx = round(Int, 360 / Δ)
    Ny = round(Int, 180 / Δ)
    return (Nx, Ny, 1)
end

dataset_prefix(::GHSBuiltH) = "GHSBuiltH"
dataset_prefix(dataset::GHSBuiltS) =
    string("GHSBuiltS_", native_resolution(dataset), "m_", dataset.epoch)
dataset_prefix(dataset::GHSBuilt) =
    string("GHSBuilt_", native_resolution(dataset), "m_", dataset.surface.epoch, "_measured_moments")

DataWrangling.metadata_filename(dataset::AbstractGHSLDataset, name, date, region) =
    string(dataset_prefix(dataset), "_", bounding_box_suffix(region), ".nc")

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

# Float64 preserves spherical cell areas at 10 m spacing.
Base.eltype(::GHSLMetadatum) = Float64

# The regional NetCDF we materialize stores coordinates as "lon"/"lat".
DataWrangling.longitude_name(::GHSLMetadatum) = "lon"
DataWrangling.latitude_name(::GHSLMetadatum)  = "lat"

# Never inpaint: a building height / built fraction of 0 over non-built land is a
# valid value, not a gap. The no-data is masked to NaN in the warp step.
DataWrangling.default_inpainting(::GHSLMetadatum) = nothing

Oceananigans.Fields.location(::GHSLMetadatum) = (Center, Center, Center)

#####
##### GHSL World-Mollweide (ESRI:54009) tile grid
#####
##### R2023A tiles a global Mollweide grid of 18 rows × 36 columns of 1000 km
##### squares, indexed `R{row}_C{col}` from the upper-left (NW) corner.
#####

# ESRI:54009 World Mollweide is defined on a sphere of the WGS84 semi-major radius.
const MOLLWEIDE_RADIUS = 6378137.0

# Tile-grid origin (NW corner) and extent, from the JRC tiling schema.
const GHSL_TILE_SIZE    = 1_000_000.0
const GHSL_ORIGIN_X     = -18_041_000.0
const GHSL_ORIGIN_Y     =   9_000_000.0
const GHSL_COLUMNS      = 36
const GHSL_ROWS         = 18

"""
    longitude_latitude_to_mollweide(longitude, latitude)

Forward World-Mollweide (ESRI:54009) projection: map `(longitude, latitude)` in
degrees (central meridian 0°) to `(x, y)` in meters. The auxiliary angle `θ` solves
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
$(TYPEDSIGNATURES)

Return the sorted `(row, column)` GHSL tiles whose 1000 km Mollweide cells intersect the
Mollweide bounding box of `region`. Exact for regional windows; a complete (never-missing)
superset for any window.
"""
function ghsl_tiles_in_bbox(region::BoundingBox)
    xmin, xmax, ymin, ymax = ghsl_mollweide_bounds(region)
    col_lo = clamp(floor(Int, (xmin - GHSL_ORIGIN_X) / GHSL_TILE_SIZE) + 1, 1, GHSL_COLUMNS)
    col_hi = clamp(floor(Int, (xmax - GHSL_ORIGIN_X) / GHSL_TILE_SIZE) + 1, 1, GHSL_COLUMNS)
    row_lo = clamp(floor(Int, (GHSL_ORIGIN_Y - ymax) / GHSL_TILE_SIZE) + 1, 1, GHSL_ROWS)
    row_hi = clamp(floor(Int, (GHSL_ORIGIN_Y - ymin) / GHSL_TILE_SIZE) + 1, 1, GHSL_ROWS)
    return [(row, col) for row in row_lo:row_hi for col in col_lo:col_hi]
end

"""
$(TYPEDSIGNATURES)

The Mollweide bounding box `(xmin, xmax, ymin, ymax)` of `region`: `y` is monotone in
latitude and `|x|` is largest at the equator-most latitude, so the extremes lie among the
region's corners and its equator-most edge.
"""
function ghsl_mollweide_bounds(region)
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
    return xmin, xmax, ymin, ymax
end

"""
$(TYPEDSIGNATURES)

The Mollweide bounds `(xmin, xmax, ymin, ymax)` of `region`, widened outward onto the faces
of the GHSL native pixels of `pixel_size` meters.
"""
function ghsl_pixel_window(region, pixel_size)
    xmin, xmax, ymin, ymax = ghsl_mollweide_bounds(region)
    return (GHSL_ORIGIN_X + pixel_size * floor((xmin - GHSL_ORIGIN_X) / pixel_size),
            GHSL_ORIGIN_X + pixel_size * ceil((xmax - GHSL_ORIGIN_X) / pixel_size),
            GHSL_ORIGIN_Y + pixel_size * floor((ymin - GHSL_ORIGIN_Y) / pixel_size),
            GHSL_ORIGIN_Y + pixel_size * ceil((ymax - GHSL_ORIGIN_Y) / pixel_size))
end

#####
##### Download-URL construction (JRC jeodpp open-data HTTPS)
#####

const GHSL_FTP = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"

# The product-folder / file basename stem shared by a dataset's tiles and mosaic.
product_stem(::GHSBuiltH) = "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100"
product_stem(dataset::GHSBuiltS) =
    string("GHS_BUILT_S_E", dataset.epoch, "_GLOBE_R2023A_54009_", native_resolution(dataset))

product_group(::GHSBuiltH) = "GHS_BUILT_H_GLOBE_R2023A"
product_group(::GHSBuiltS) = "GHS_BUILT_S_GLOBE_R2023A"

"""
    ghsl_tile_url(dataset, row, column)

Return the JRC open-data HTTPS URL of the `(row, column)` GHSL tile `.zip` for
`dataset` (each archive holds one Mollweide GeoTIFF of the built-up variable).
"""
function ghsl_tile_url(dataset::Union{GHSBuiltH, GHSBuiltS}, row, column)
    stem = product_stem(dataset)
    return string(GHSL_FTP, "/", product_group(dataset), "/", stem,
                  "/V1-0/tiles/", stem, "_V1_0_R", row, "_C", column, ".zip")
end

"""
    ghsl_tile_urls(dataset, region::BoundingBox)

`.zip` URLs of the GHSL tiles for `dataset` intersecting `region`, ready to be
downloaded, unzipped, and mosaicked/reprojected by GDAL.
"""
ghsl_tile_urls(dataset::Union{GHSBuiltH, GHSBuiltS}, region::BoundingBox) =
    [ghsl_tile_url(dataset, row, column) for (row, column) in ghsl_tiles_in_bbox(region)]

# GeoTIFF basename inside a tile `.zip` archive.
ghsl_tile_tif_name(dataset::Union{GHSBuiltH, GHSBuiltS}, row, column) =
    string(product_stem(dataset), "_V1_0_R", row, "_C", column, ".tif")

#####
##### Pure no-data masking + built-surface → fraction conversion (unit-testable core)
#####

"""
    mask_building_height(value)

Map a GHS-BUILT-H ANBH `value` (meters) to a masked `Float64`, keeping every physical
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
the plan-area index `λᵖ = surface / cell_area`, clamped to `[0, 1]`. `cell_area` is the
native cell area in m² (`100 × 100` at 100 m, `10 × 10` at 10 m). GHSL declares its no-data
as a positive sentinel (`65535` at 100 m, `255` at 10 m) that the warp has already written
as `NaN`; a non-finite or negative `surface` maps to `NaN`.

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

"""
    ghsl_regional_raster(metadatum)

Geometry of the regional raster materialized for `metadatum`, taken from `native_grid`
so that the file the reprojection writes lands on the cells the read path indexes.
Returns the `region` to reproject onto — the native grid's extent, which reaches up to
a cell beyond the requested window, and so is also the window whose tiles are needed —
along with the cell counts `Nx`, `Ny` and the cell-center `longitude` and `latitude`.

```jldoctest
julia> using NumericalEarth.DataWrangling: BoundingBox, Metadatum

julia> using NumericalEarth.DataWrangling.GHSL: GHSBuiltH, ghsl_regional_raster

julia> region = BoundingBox(longitude = (-0.11, -0.07), latitude = (51.505, 51.525));

julia> raster = ghsl_regional_raster(Metadatum(:building_height; dataset = GHSBuiltH(), region));

julia> (raster.Nx, raster.Ny)
(46, 25)
```
"""
function ghsl_regional_raster(metadatum::GHSLMetadatum)
    grid = native_grid(metadatum, CPU())
    λ₁, λ₂ = x_domain(grid)
    φ₁, φ₂ = y_domain(grid)
    Nx, Ny, _ = size(grid)

    return (; region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)),
              Nx, Ny,
              longitude_faces = collect(λnodes(grid, Face())),
              latitude_faces = collect(φnodes(grid, Face())),
              longitude = collect(λnodes(grid, Center())),
              latitude  = collect(φnodes(grid, Center())))
end

function Downloads.download(metadatum::GHSLMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        ghsl_tiles_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

ghsl_tiles_to_netcdf(metadatum, nc_path) =
    error("Reading $(summary(metadatum.dataset)) Mollweide GeoTIFF tiles requires `using ArchGDAL`.")

"""
$(TYPEDSIGNATURES)

Plan-area index and built-area-weighted mean building height (m) on `grid` (a
`LatitudeLongitudeGrid`) from the GHSL built-up surface and building-height rasters
of `dataset`, as the NamedTuple of `Field`s
`(; plan_area_index, mean_building_height, measured_plan_area_index)`: the area mean of the native-pixel built
fraction, the mean height of the built pixels with a positive height, and their plan-area index. A pixel missing
its fraction counts as unbuilt; a cell with built pixels but no heights has a `NaN` mean
height. The rasters are read in windows of
at most `maximum_raster_cells` native pixels (default `100_000_000`) per product, including
when a grid cell spans multiple windows. Missing ocean tiles contribute zero built area
and volume. Downloaded source tiles remain cached. Requires `using ArchGDAL`.
"""
Lands.building_morphometry(grid, dataset::GHSBuilt; kw...) =
    error("Reading $(summary(dataset)) building morphometry requires `using ArchGDAL`.")

end # module GHSL
