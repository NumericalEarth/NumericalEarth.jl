module CanopyHeight

export ETHCanopyHeight, GLADCanopyHeight

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, BoundingBox

import Oceananigans

download_CanopyHeight_cache::String = ""
function __init__()
    global download_CanopyHeight_cache = DataWrangling.download_cache("CanopyHeight")
end

#####
##### Dataset types
#####

"""
    ETHCanopyHeight

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
struct ETHCanopyHeight <: AbstractStaticDataset end

"""
    GLADCanopyHeight

GLAD Global Forest Canopy Height (Potapov et al. 2021, epoch 2019). A 30 m
(0.00025°), EPSG:4326, 8-bit GeoTIFF calibrated to GEDI RH95. Valid heights are
`0–60` m; the fill codes `101` (water), `102` (snow/ice) and `103` (no-data)
must be masked to `NaN` *before* any spatial averaging (see [`mask_glad`](@ref)).

Shares the COG read path with [`ETHCanopyHeight`](@ref); documented as a coarser
fallback and cross-check (forest-focused, GEDI-blind above ±51.6° latitude).

Reference: Potapov, P. et al. (2021), *Mapping global forest canopy height…*,
Remote Sens. Environ. 253, 112165, doi:10.1016/j.rse.2020.112165.

Data source: https://glad.umd.edu/dataset/gedi/
"""
struct GLADCanopyHeight <: AbstractStaticDataset end

const CanopyHeightDataset  = Union{ETHCanopyHeight, GLADCanopyHeight}
const CanopyHeightMetadatum = Metadatum{<:CanopyHeightDataset}

const ETHCanopyHeightMetadatum  = Metadatum{<:ETHCanopyHeight}
const GLADCanopyHeightMetadatum = Metadatum{<:GLADCanopyHeight}

#####
##### Variables
#####

# ETH ships both a height map and an uncertainty layer; GLAD ships only the map.
ETHCanopyHeight_variable_names = Dict(
    :canopy_height             => "Map",
    :canopy_height_uncertainty => "SD",
)

GLADCanopyHeight_variable_names = Dict(
    :canopy_height => "Map",
)

DataWrangling.available_variables(::ETHCanopyHeight)  = ETHCanopyHeight_variable_names
DataWrangling.available_variables(::GLADCanopyHeight) = GLADCanopyHeight_variable_names

DataWrangling.dataset_variable_name(data::ETHCanopyHeightMetadatum)  = ETHCanopyHeight_variable_names[data.name]
DataWrangling.dataset_variable_name(data::GLADCanopyHeightMetadatum) = GLADCanopyHeight_variable_names[data.name]

#####
##### Dataset interface
#####

DataWrangling.default_download_directory(::CanopyHeightDataset) = download_CanopyHeight_cache

# Both products are already geographic (EPSG:4326), global hull.
DataWrangling.longitude_interfaces(::CanopyHeightDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::CanopyHeightDataset)  = (-90, 90)

# Global native pixel counts (used to set the windowed-read target cell size
# Δ = 360/Nx). ETH is 10 m = 1/12000° → 4_320_000 × 2_160_000; GLAD is 30 m =
# 0.00025° = 1/4000°.
Base.size(::ETHCanopyHeight,  variable) = (4_320_000, 2_160_000, 1)
Base.size(::GLADCanopyHeight, variable) = (1_440_000, 720_000, 1)

dataset_prefix(::ETHCanopyHeight)  = "ETHCanopyHeight"
dataset_prefix(::GLADCanopyHeight) = "GLADCanopyHeight"

# One regional NetCDF per product per variable per region, materialized from the
# COG tiles (ETH ships separate `_Map` height and `_Map_SD` uncertainty layers).
DataWrangling.metadata_filename(dataset::CanopyHeightDataset, name, date, region) =
    string(dataset_prefix(dataset), "_", string(name), "_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::CanopyHeightMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        prefix = dataset_prefix(metadata.dataset)
        error("$(prefix)() must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:canopy_height; dataset = $(prefix)(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

#####
##### Metadatum interface
#####

DataWrangling.is_three_dimensional(::CanopyHeightMetadatum) = false

# The regional NetCDF we materialize stores coordinates as "lon"/"lat".
DataWrangling.longitude_name(::CanopyHeightMetadatum) = "lon"
DataWrangling.latitude_name(::CanopyHeightMetadatum)  = "lat"

# NEVER inpaint: a canopy height of 0 over non-forest is a valid value, not a gap.
# Only explicit no-data / GLAD fill codes are masked to NaN (in the COG read).
DataWrangling.default_inpainting(::CanopyHeightMetadatum) = nothing

# ETH `Map`/`SD` COGs use 255 as the no-data byte; canopy heights never reach it.
# GLAD's categorical fill codes (101/102/103) are masked to NaN at read time
# (see `mask_glad`), so no scalar missing_value applies.
DataWrangling.missing_value(::ETHCanopyHeightMetadatum) = 255

Oceananigans.Fields.location(::CanopyHeightMetadatum) = (Center, Center, Center)

#####
##### Pure no-data masking (unit-testable core)
#####

"""
    mask_glad(code)

Map a raw GLAD canopy-height byte `code` to a masked `Float64` height: the fill
codes `101` (water), `102` (snow/ice) and `103` (no-data) — and anything above
them — become `NaN`; all valid heights (including the legitimate non-forest
value `0`) are kept unchanged. Must be applied *before* any spatial averaging so
the categorical fill codes never corrupt a cell mean.

```jldoctest
julia> using NumericalEarth.DataWrangling.CanopyHeight: mask_glad

julia> mask_glad(0)
0.0

julia> mask_glad(37)
37.0

julia> isnan(mask_glad(101))
true
```
"""
@inline mask_glad(code) = ifelse(code >= 101, oftype(float(code), NaN), float(code))

"""
    mask_eth(x, missing_value = 255)

Map a raw ETH canopy-height value `x` to a masked `Float64` height: the no-data
byte (`missing_value`, default `255`) becomes `NaN`; all valid heights
(including the legitimate non-forest value `0`) are kept unchanged.
"""
@inline mask_eth(x, missing_value = 255) =
    ifelse(x == missing_value, oftype(float(x), NaN), float(x))

#####
##### Antialiased downsampling (10 m → coarse cell); NaN-mask first
#####

"""
    coarsen_canopy_height(fine, factor)

Antialiased block-mean downsampling of a fine 2-D canopy-height array `fine`
onto a grid coarsened by integer `factor` in each dimension — the reference for
taking a 10 m raster onto a ~1 km model cell. No-data must already be `NaN`
(via [`mask_glad`](@ref) / [`mask_eth`](@ref)): each coarse cell averages only
the finite fine cells beneath it, and is `NaN` only if every contributing fine
cell is `NaN`. This mirrors the NaN-aware, mask-before-average discipline of the
multi-pass coarsening in `interpolate_bathymetry_in_passes`.
"""
function coarsen_canopy_height(fine::AbstractMatrix, factor::Integer)
    factor >= 1 || throw(ArgumentError("coarsening factor must be ≥ 1, got $factor"))
    Nx, Ny = size(fine)
    Cx = cld(Nx, factor)
    Cy = cld(Ny, factor)
    coarse = fill(convert(float(eltype(fine)), NaN), Cx, Cy)

    for J in 1:Cy, I in 1:Cx
        Σ = zero(float(eltype(fine)))
        n = 0
        for j in ((J - 1) * factor + 1):min(J * factor, Ny)
            for i in ((I - 1) * factor + 1):min(I * factor, Nx)
                v = fine[i, j]
                if !isnan(v)
                    Σ += v
                    n += 1
                end
            end
        end
        if n > 0
            coarse[I, J] = Σ / n
        end
    end

    return coarse
end

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

# GLAD Global Forest Canopy Height 2019 ships as seven continental 30 m GeoTIFF
# mosaics (not one global file) on the GLAD geog host, named by a continent code:
# `Forest_height_2019_<CONT>.tif`. NOTE (verified 2026-07): every candidate tile
# URL returned HTTP 404 from this environment (the host resolves but rejects the
# path), so the GLAD read is documented best-effort / unverified — ETH is primary.
const GLAD_COG_HOST = "https://glad.geog.umd.edu/Potapov/Forest_height_2019"

# Continent codes covering the GLAD tiling (used to pick the intersecting tile).
const GLAD_CONTINENTS = ("NAM", "SAM", "EURA", "NAFR", "SAFR", "AUS", "SASIA", "NASIA")

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

"""
    glad_continent(longitude, latitude)

Coarse map of a point to the GLAD continental-mosaic code (`"NAM"`, `"SAM"`,
`"EURA"`, `"NAFR"`, `"SAFR"`, `"AUS"`, `"SASIA"`, `"NASIA"`). This is a
best-effort classifier for picking the intersecting GLAD tile; the boundaries are
approximate (GLAD's tiles follow continental outlines, not a lat/lon lattice).
"""
function glad_continent(longitude, latitude)
    λ = longitude
    φ = latitude
    if λ < -30                                   # Americas
        return φ >= 12 ? "NAM" : "SAM"
    elseif λ < 60                                # Europe / Africa
        return φ >= 12 ? (φ >= 36 ? "EURA" : "NAFR") : (φ >= -35 ? "NAFR" : "SAFR")
    elseif λ < 120                               # W/Central Asia
        return φ >= 30 ? (φ >= 55 ? "NASIA" : "EURA") : "SASIA"
    else                                         # E Asia / Oceania
        return φ >= 30 ? "NASIA" : (φ >= -10 ? "SASIA" : "AUS")
    end
end

"""
    glad_tile_urls(region::BoundingBox)

`/vsicurl/`-prefixed URLs of the GLAD continental mosaics intersecting `region`
(deduplicated over the bbox corners). Best-effort: see [`GLAD_COG_HOST`](@ref).
"""
function glad_tile_urls(region::BoundingBox)
    λ₁, λ₂ = region.longitude
    φ₁, φ₂ = region.latitude
    continents = unique!([glad_continent(λ, φ) for λ in (λ₁, λ₂) for φ in (φ₁, φ₂)])
    return ["/vsicurl/" * GLAD_COG_HOST * "/Forest_height_2019_" * c * ".tif"
            for c in continents]
end

function Downloads.download(metadatum::CanopyHeightMetadatum)
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

end # module CanopyHeight
