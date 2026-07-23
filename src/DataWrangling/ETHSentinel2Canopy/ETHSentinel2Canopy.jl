module ETHSentinel2Canopy

export ETHSentinel2CanopyHeight, canopy_height_field

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, BoundingBox

import Oceananigans

download_ETHSentinel2Canopy_cache::String = ""
function __init__()
    global download_ETHSentinel2Canopy_cache = DataWrangling.download_cache("ETHSentinel2Canopy")
end

#####
##### Dataset type
#####

"""
    ETHSentinel2CanopyHeight

ETH Global Sentinel-2 10 m Canopy Height Model (Lang et al. 2023, version 1,
epoch 2020). A continuous, global, EPSG:4326 Cloud-Optimized-GeoTIFF (COG)
raster distributed in 3°×3° tiles. Two layers per tile:

- `"Map"` — canopy height in meters (variable `:canopy_height`);
- `"SD"`  — per-pixel standard deviation / uncertainty
  (variable `:canopy_height_uncertainty`).

Canopy height over non-forest is a legitimate value of `0` m (not missing), so
these fields are **not inpainted** and zeros are kept — only the product's
explicit no-data code is masked to `NaN`.

Because it is a global 10 m product, it is read in regional windows only:
construct the `Metadatum` with a longitude/latitude `BoundingBox`. The windowed
COG read (`/vsicurl/`, anonymous) is performed by
`ext/NumericalEarthArchGDALExt.jl` and requires `using ArchGDAL`.

Reference: Lang, N. et al. (2023), *A high-resolution canopy height model of the
Earth*, Nat. Ecol. Evol. 7:1778–1789, doi:10.1038/s41559-023-02206-6; data DOI
10.3929/ethz-b-000609802. License CC-BY 4.0.

Data source: https://langnico.github.io/globalcanopyheight/
"""
struct ETHSentinel2CanopyHeight <: AbstractStaticDataset end

const ETHSentinel2CanopyHeightMetadatum = Metadatum{<:ETHSentinel2CanopyHeight}

#####
##### Variables
#####

# ETH ships both a height map and an uncertainty layer.
ETHSentinel2CanopyHeight_variable_names = Dict(
    :canopy_height             => "Map",
    :canopy_height_uncertainty => "SD",
)

DataWrangling.available_variables(::ETHSentinel2CanopyHeight) = ETHSentinel2CanopyHeight_variable_names

DataWrangling.dataset_variable_name(data::ETHSentinel2CanopyHeightMetadatum) = ETHSentinel2CanopyHeight_variable_names[data.name]

#####
##### Dataset interface
#####

DataWrangling.default_download_directory(::ETHSentinel2CanopyHeight) = download_ETHSentinel2Canopy_cache

# Already geographic (EPSG:4326), global hull.
DataWrangling.longitude_interfaces(::ETHSentinel2CanopyHeight) = (-180, 180)
DataWrangling.latitude_interfaces(::ETHSentinel2CanopyHeight)  = (-90, 90)

# Global native pixel count (used to set the windowed-read target cell size
# Δ = 360/Nx): ETH is 10 m = 1/12000° → 4_320_000 × 2_160_000.
Base.size(::ETHSentinel2CanopyHeight, variable) = (4_320_000, 2_160_000, 1)

dataset_prefix(::ETHSentinel2CanopyHeight) = "ETHSentinel2CanopyHeight"

# One regional NetCDF per variable per region, materialized from the COG tiles
# (ETH ships separate `_Map` height and `_Map_SD` uncertainty layers).
DataWrangling.metadata_filename(dataset::ETHSentinel2CanopyHeight, name, date, region) =
    string(dataset_prefix(dataset), "_", string(name), "_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::ETHSentinel2CanopyHeightMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        prefix = dataset_prefix(metadata.dataset)
        error("$(prefix)() must be used with a bounded region. " *
              "Read it onto a grid directly with\n" *
              "    canopy_height_field(grid, $(prefix)())\n" *
              "or build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:canopy_height; dataset = $(prefix)(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

#####
##### Metadatum interface
#####

DataWrangling.is_three_dimensional(::ETHSentinel2CanopyHeightMetadatum) = false

# The regional NetCDF we materialize stores coordinates as "lon"/"lat".
DataWrangling.longitude_name(::ETHSentinel2CanopyHeightMetadatum) = "lon"
DataWrangling.latitude_name(::ETHSentinel2CanopyHeightMetadatum)  = "lat"

# NEVER inpaint: a canopy height of 0 over non-forest is a valid value, not a gap.
# The no-data byte (255) is masked to NaN in the COG read (see `mask_eth`), so the
# on-disk sentinel is already NaN and the default `missing_value` (NaN passthrough) applies.
DataWrangling.default_inpainting(::ETHSentinel2CanopyHeightMetadatum) = nothing

Oceananigans.Fields.location(::ETHSentinel2CanopyHeightMetadatum) = (Center, Center, Center)

#####
##### Pure no-data masking (unit-testable core)
#####

"""
    mask_eth(x, missing_value = 255)

Map a raw ETH canopy-height value `x` to a masked `Float64` height: the no-data
byte (`missing_value`, default `255`) becomes `NaN`; all valid heights
(including the legitimate non-forest value `0`) are kept unchanged.
"""
@inline mask_eth(x, missing_value = 255) =
    ifelse(x == missing_value, oftype(float(x), NaN), float(x))

#####
##### Download (regional COG → NetCDF via the ArchGDAL extension)
#####

# ETH ships the full-resolution 10 m product as 3°×3° Cloud-Optimized GeoTIFF tiles
# (EPSG:4326, 255 = no-data), named by their SW-corner lat/lon token (e.g. "N45E009"),
# with a `_Map.tif` canopy-height layer and a `_Map_SD.tif` uncertainty layer. They are
# served anonymously from an ETH libdrive public share exposed as a Nextcloud WebDAV
# endpoint. That endpoint honours HTTP range requests — so `/vsicurl/` windows the COGs
# without downloading whole 3° tiles — but only when the request carries a browser
# User-Agent and the public read-only share token as basic-auth credentials; both are
# set on the GDAL HTTP driver in the ArchGDAL extension.
const ETH_WEBDAV_HOST = "https://libdrive.ethz.ch/public.php/webdav/3deg_cogs"
const ETH_LIBDRIVE_TOKEN = "cO8or7iOe5dT2Rt"     # public read-only share token (CC-BY data)
const ETH_TILE_RESOLUTION = 1 / 12000            # degrees (~9.3 m), the native 10 m grid
const ETH_BROWSER_USER_AGENT =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " *
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"

"""
    eth_tile_token(longitude, latitude)

Return the ETH 3°×3° tile token (`"<N|S>lat<E|W>lon"`, SW-corner on the 3°
lattice, zero-padded to 2/3 digits) containing `(longitude, latitude)`,
e.g. `(4.2, 51.7) → "N51E003"`.
"""
function eth_tile_token(longitude, latitude)
    lat0 = 3 * fld(latitude, 3)
    lon0 = 3 * fld(longitude, 3)
    ns = lat0 >= 0 ? "N" : "S"
    ew = lon0 >= 0 ? "E" : "W"
    return string(ns, lpad(abs(Int(lat0)), 2, '0'), ew, lpad(abs(Int(lon0)), 3, '0'))
end

"""
    eth_tiles_in_bbox(region::BoundingBox)

Return the sorted unique ETH tile tokens whose 3° cells intersect `region`.
"""
function eth_tiles_in_bbox(region::BoundingBox)
    λ₁, λ₂ = region.longitude
    φ₁, φ₂ = region.latitude
    (λ₁ < λ₂ && φ₁ < φ₂) ||
        error("BoundingBox bounds must be ascending, got longitude = ($λ₁, $λ₂), latitude = ($φ₁, $φ₂). " *
              "Windowed COG reads do not support inverted or antimeridian-crossing bounds; " *
              "split an antimeridian-crossing region at ±180°.")
    tokens = String[]
    for lat0 in (3 * fld(φ₁, 3)):3:(3 * fld(φ₂, 3))
        for lon0 in (3 * fld(λ₁, 3)):3:(3 * fld(λ₂, 3))
            push!(tokens, eth_tile_token(lon0, lat0))
        end
    end
    return sort!(unique!(tokens))
end

# COG filename suffix per variable: the height layer is `_Map`, the uncertainty `_Map_SD`.
eth_layer_suffix(::Val{:canopy_height})             = "Map"
eth_layer_suffix(::Val{:canopy_height_uncertainty}) = "Map_SD"
eth_layer_suffix(name::Symbol) = eth_layer_suffix(Val(name))

"""
    eth_tile_urls(region::BoundingBox, name)

`/vsicurl/`-prefixed WebDAV URLs of the ETH 10 m COG tiles for variable `name`
(`:canopy_height` → `_Map`, `:canopy_height_uncertainty` → `_Map_SD`) whose 3° cells
intersect `region`, ready to be mosaicked and windowed by GDAL.
"""
eth_tile_urls(region::BoundingBox, name) =
    ["/vsicurl/" * ETH_WEBDAV_HOST * "/ETH_GlobalCanopyHeight_10m_2020_" *
     token * "_" * eth_layer_suffix(name) * ".tif" for token in eth_tiles_in_bbox(region)]

function Downloads.download(metadatum::ETHSentinel2CanopyHeightMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        canopy_height_cog_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded. The
# fallback below fires only when the extension is not active (mirrors
# CopernicusDEM.zarr_to_netcdf).
canopy_height_cog_to_netcdf(metadatum, nc_path) =
    error("Reading the $(dataset_prefix(metadatum.dataset)) Cloud-Optimized GeoTIFF " *
          "requires the ArchGDAL package. Load it with `using ArchGDAL`.")

"""
    canopy_height_field(grid, dataset; name = :canopy_height, resampling = "average")

Read `dataset` canopy height directly onto `grid`, area-averaging (`-r average`, with the
no-data byte excluded from each cell mean) the native COG pixels within each grid cell —
coarse-graining rather than point interpolation, the correct reduction from a 10 m raster
onto a coarse model cell. Only the windowed COG blocks are read (anonymous `/vsicurl/`), so
no full-resolution regional file is materialized.

Returns a `Field{Center, Center, Nothing}(grid)`: canopy height over non-forest is a valid
`0`, tiles absent over ocean are skipped, and the product no-data code is masked to `NaN`.
Requires the `ArchGDAL` package (`using ArchGDAL`).
"""
canopy_height_field(grid, dataset; kw...) =
    error("Reading a canopy-height Cloud-Optimized GeoTIFF onto a grid requires the " *
          "ArchGDAL package. Load it with `using ArchGDAL`.")

end # module ETHSentinel2Canopy
