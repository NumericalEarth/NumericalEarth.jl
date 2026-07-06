module CanopyHeight

export ETHCanopyHeight, GLADCanopyHeight, RoughnessFromCanopyHeight

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, BoundingBox

using NCDatasets: Dataset

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

# Global native pixel counts (only used to set the windowed-read target cell size
# Δ = 360/Nx). GLAD is 30 m = 0.00025° = 1/4000°. ETH's native 10 m 3° COG tiles
# are no longer served anonymously (see the host note below): the anonymously
# readable ETH product is the pre-downsampled global mosaic, whose finest grid is
# 0.001° (~111 m) → 360000 × 144000 over −60°…84° latitude, so 360/Nx = 0.001°.
Base.size(::ETHCanopyHeight,  variable) = (360_000,   144_000, 1)
Base.size(::GLADCanopyHeight, variable) = (1_440_000, 720_000, 1)

dataset_prefix(::ETHCanopyHeight)  = "ETHCanopyHeight"
dataset_prefix(::GLADCanopyHeight) = "GLADCanopyHeight"

# One regional NetCDF per product per region, materialized from the COG(s); it
# holds all variables of the product (ETH: Map + SD).
DataWrangling.metadata_filename(dataset::CanopyHeightDataset, name, date, region) =
    string(dataset_prefix(dataset), "_", region_suffix(region), ".nc")

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
##### Roughness / displacement closure (exposed coefficients, separate from data)
#####

"""
    roughness_length(canopy_height, coefficient)

Momentum roughness length `ℓ_m = coefficient · canopy_height`. The coefficient
is a modeling choice (`0.10` is a common surface-layer value; ClimaLand uses
`0.13` for vegetation), so it is exposed, never hardcoded in the data layer.
"""
@inline roughness_length(canopy_height, coefficient) = coefficient * canopy_height

"""
    displacement_height(canopy_height, coefficient)

Zero-plane displacement height `d = coefficient · canopy_height` (`0.70` is the
standard surface-layer value).
"""
@inline displacement_height(canopy_height, coefficient) = coefficient * canopy_height

"""
    RoughnessFromCanopyHeight(FT=Float64; momentum_roughness_coefficient = 0.10,
                                          displacement_coefficient = 0.70)

A small closure turning a canopy height `h_c` into aerodynamic surface-layer
properties via the standard rules

    ℓ_m = momentum_roughness_coefficient · h_c        (default 0.10)
    d   = displacement_coefficient       · h_c        (default 0.70)

The coefficients belong to the roughness *closure*, not the data layer — the
canopy-height dataset delivers a clean `h_c` field, and this callable derives
`ℓ_m` and `d` from it. ClimaLand's `z0m = 0.13·h_c` is an alternative choice for
`momentum_roughness_coefficient`. Calling the closure on a height returns a
`NamedTuple` `(; momentum_roughness_length, displacement_height)`; it broadcasts
elementwise over arrays and `Field`s.

```jldoctest
julia> using NumericalEarth.DataWrangling.CanopyHeight

julia> roughness = RoughnessFromCanopyHeight();

julia> roughness(20.0)
(momentum_roughness_length = 2.0, displacement_height = 14.0)
```
"""
struct RoughnessFromCanopyHeight{FT}
    momentum_roughness_coefficient :: FT
    displacement_coefficient       :: FT
end

function RoughnessFromCanopyHeight(FT = Float64;
                                   momentum_roughness_coefficient = 0.10,
                                   displacement_coefficient = 0.70)
    return RoughnessFromCanopyHeight{FT}(convert(FT, momentum_roughness_coefficient),
                                         convert(FT, displacement_coefficient))
end

@inline (r::RoughnessFromCanopyHeight)(canopy_height) =
    (momentum_roughness_length = roughness_length(canopy_height, r.momentum_roughness_coefficient),
     displacement_height       = displacement_height(canopy_height, r.displacement_coefficient))

Base.summary(r::RoughnessFromCanopyHeight) =
    string("RoughnessFromCanopyHeight(momentum_roughness_coefficient = ",
           r.momentum_roughness_coefficient,
           ", displacement_coefficient = ", r.displacement_coefficient, ")")

Base.show(io::IO, r::RoughnessFromCanopyHeight) = print(io, summary(r))

#####
##### Download (regional COG → NetCDF via the ArchGDAL extension)
#####

# ETH share host, 3°×3° full-resolution (10 m) COG tiles named by their SW-corner
# lat/lon token (e.g. "N51E004"). NOTE (verified 2026-07 against the live host):
# this `.../version1/3deg_cogs/` path no longer serves the tiles anonymously — it
# now 301-redirects to the dataset DOI landing page, so the 10 m tiles are only
# reachable through the DOI record, not via `/vsicurl/`. The `eth_tile_*` helpers
# below still encode the (unchanged, correct) 3° tile addressing for that archive.
const ETH_COG_HOST = "https://share.phys.ethz.ch/~pf/nlangdata/" *
                     "ETH_GlobalCanopyHeight_10m_2020_version1/3deg_cogs"

# The anonymously readable ETH product is the pre-downsampled global mosaic
# (single COG, EPSG:4326, 255 = no-data). The finest publicly served grid is
# 0.001° (~111 m) — adequate for ~1 km land cells — and is windowed via /vsicurl.
const ETH_DOWNSAMPLED_HOST = "https://share.phys.ethz.ch/~pf/nlangdata/" *
                             "ETH_GlobalCanopyHeight_10m_2020_version1_downsampled"

const ETH_MOSAIC_RESOLUTION = 0.001  # degrees; finest anonymously served ETH grid

eth_mosaic_url() = "/vsicurl/" * ETH_DOWNSAMPLED_HOST *
    "/ETH_GlobalCanopyHeight_10m_2020_mosaic_Map_0.001deg.tif"

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

"""
    eth_tile_urls(region::BoundingBox, variable_layer)

`/vsicurl/`-prefixed URLs of the ETH COG tiles (for the `"Map"` or `"SD"` layer)
intersecting `region`, ready to be mosaicked and windowed by GDAL.
"""
eth_tile_urls(region::BoundingBox, layer) =
    ["/vsicurl/" * ETH_COG_HOST * "/ETH_GlobalCanopyHeight_10m_2020_" *
     token * "_" * layer * ".tif" for token in eth_tiles_in_bbox(region)]

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
