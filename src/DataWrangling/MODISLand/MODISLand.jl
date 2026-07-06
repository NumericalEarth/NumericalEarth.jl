module MODISLand

export MCD43Albedo, MCD15A3H, MCD15A2H, MOD15A2H, MCD12Q1

using Dates: Dates, DateTime
using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum, Metadata,
                       BoundingBox, Dataset, metadata_path

import Oceananigans

download_MODISLand_cache::String = ""
function __init__()
    return global download_MODISLand_cache = DataWrangling.download_cache("MODISLand")
end

#####
##### Pure decode / mask / blend / aggregation helpers
#####
#####
##### These carry the physics of the MODIS-land ingest and are the core, testable
##### deliverable: they operate on raw digital numbers (DN) read from an HDF-EOS
##### granule (or a reprojected NetCDF of raw DN) and require no file IO and no
##### credentials. All follow the MODIS `scale × DN (+ offset)` decode convention
##### with fill/QA masking applied *before* scaling.
#####

# MCD43A3 albedo: int16 DN, scale 0.001, offset 0, valid 0–32766, fill 32767.
const MODIS_ALBEDO_FILL = 32767
const MODIS_ALBEDO_SCALE = 0.001

"""
    decode_albedo(DN)

Decode a MODIS MCD43A3 albedo digital number `DN` to a physical albedo in `[0, 1]`.
The fill sentinel `32767` maps to `NaN`; the mask is applied *before* the `0.001`
scaling so a fill does not become a spurious albedo of `32.767`.
"""
@inline decode_albedo(DN) = ifelse(DN == MODIS_ALBEDO_FILL, NaN, MODIS_ALBEDO_SCALE * DN)

"""
    bluesky_blend(α_bs, α_ws, f_diff)

Blend black-sky albedo `α_bs` (direct-beam) and white-sky albedo `α_ws` (fully
diffuse) into the blue-sky (actual) albedo the surface energy budget uses, with
`f_diff` the diffuse fraction (`skyl`) of downwelling shortwave:

    α_blue = (1 − f_diff) · α_bs + f_diff · α_ws

`f_diff = 0` returns the black-sky value and `f_diff = 1` the white-sky value.
"""
@inline bluesky_blend(α_bs, α_ws, f_diff) = (1 - f_diff) * α_bs + f_diff * α_ws

# MCD43A3 mandatory QA (uint8): 0 = full BRDF inversion (best), 1 = magnitude
# inversion, 2–7 = inversions with band-5/6 fill, 255 = fill.
"""
    albedo_quality_ok(mandatory_qc, maximum_quality = 0)

Return `true` when the MCD43A3 mandatory-QA value `mandatory_qc` is a retained
retrieval: not fill (`255`) and no worse than `maximum_quality` (default `0` keeps
only the full-BRDF inversion; pass `1` to also keep magnitude inversions).
"""
@inline albedo_quality_ok(mandatory_qc, maximum_quality) =
    (mandatory_qc != 0xff) & (mandatory_qc <= maximum_quality)
@inline albedo_quality_ok(mandatory_qc) = albedo_quality_ok(mandatory_qc, 0x00)

# MCD15 LAI/FPAR: uint8 DN, valid 0–100. DN 249–255 are non-retrieval / land-cover
# special codes (urban, wetland, snow, barren, water, fill) and must be rejected
# *before* scaling. LAI scale 0.1 (→ 0–10 m²/m²); FPAR scale 0.01 (→ 0–1).
const MODIS_LAI_MAXIMUM_VALID = 100
const MODIS_LAI_SCALE = 0.1
const MODIS_FPAR_SCALE = 0.01

"""
    decode_lai(DN)

Decode a MODIS MCD15 `Lai_500m` digital number `DN` to LAI in `m²/m²`. Any
`DN > 100` (fill and the 249–255 land-cover special codes) maps to `NaN`; the mask
is applied *before* the `0.1` scaling so a fill (e.g. `255`) does not become a
spurious LAI of `25.5`.
"""
@inline decode_lai(DN) = ifelse(DN > MODIS_LAI_MAXIMUM_VALID, NaN, MODIS_LAI_SCALE * DN)

"""
    decode_fpar(DN)

Decode a MODIS MCD15 `Fpar_500m` digital number `DN` to FPAR in `[0, 1]`, masking
`DN > 100` to `NaN` before the `0.01` scaling (see [`decode_lai`](@ref)).
"""
@inline decode_fpar(DN) = ifelse(DN > MODIS_LAI_MAXIMUM_VALID, NaN, MODIS_FPAR_SCALE * DN)

# FparLai_QC (uint8) packed bitfield: bit 0 MODLAND_QC (0 good), bits 5–7 SCF_QC
# (0 main-RT best, 1 main-RT w/ saturation, 2/3 backup empirical, 4 not produced).
@inline modland_quality_control(qc) = qc & 0x01
@inline scf_quality_control(qc) = (qc >> 0x05) & 0x07

"""
    lai_quality_ok(qc)

Return `true` when the MCD15 `FparLai_QC` byte `qc` denotes a retained retrieval:
`MODLAND_QC == 0` (good quality) and `SCF_QC ∈ {0, 1}` (the main radiative-transfer
retrievals, excluding the less-reliable backup empirical algorithm).
"""
@inline lai_quality_ok(qc) =
    (modland_quality_control(qc) == 0) & (scf_quality_control(qc) <= 0x01)

# MCD12Q1 land-cover / PFT: uint8 class codes, fill 255. `LC_Type5` PFT ∈ [0, 11].
const MODIS_LANDCOVER_FILL = 255

"""
    mask_landcover(code, fill = 255)

Map a categorical land-cover `code` to itself as a `Float64`, or to `NaN` when it
equals `fill` (default `255`). Used to carry a categorical field through the
NaN-aware machinery without inventing intermediate codes.
"""
@inline mask_landcover(code, fill) = ifelse(code == fill, NaN, Float64(code))
@inline mask_landcover(code) = mask_landcover(code, MODIS_LANDCOVER_FILL)

"""
    mode_aggregate(codes, fill = 255)

Return the most frequent non-`fill` class code in `codes` (the majority class), or
`fill` when every entry is `fill`. This is the correct aggregation for categorical
fields (land-cover / PFT) when coarsening a fine raster onto a model cell — the
result is always one of the input codes, never an averaged (invented) intermediate.
"""
function mode_aggregate(codes, fill)
    best = fill
    best_count = 0
    for candidate in unique(codes)
        candidate == fill && continue
        n = count(isequal(candidate), codes)
        if n > best_count
            best_count = n
            best = candidate
        end
    end
    return best
end
mode_aggregate(codes) = mode_aggregate(codes, MODIS_LANDCOVER_FILL)

"""
    class_fraction(codes, class, fill = 255)

Return the areal fraction of the valid (non-`fill`) entries in `codes` that equal
`class`, in `[0, 1]`, or `0` when there are no valid entries. Summing
`class_fraction` over all classes present in `codes` yields `1`.
"""
function class_fraction(codes, class, fill)
    valid = count(!isequal(fill), codes)
    valid == 0 && return 0.0
    return count(isequal(class), codes) / valid
end
class_fraction(codes, class) = class_fraction(codes, class, MODIS_LANDCOVER_FILL)

#####
##### MODIS sinusoidal (SIN) grid geometry (pure, dependency-free)
#####
#####
##### Sphere R = 6371007.181 m; PROJ `+proj=sinu +lon_0=0 +a=R +b=R`. Global upper-left
##### corner (−20015109.354, 10007554.677) m; tile side 1111950.5196 m (10° at equator).
#####

const MODIS_SPHERE_RADIUS = 6371007.181
const MODIS_GLOBAL_UPPER_LEFT_X = -20015109.354
const MODIS_GLOBAL_UPPER_LEFT_Y = 10007554.677
const MODIS_TILE_SIDE = 1111950.5196

"""
    sinusoidal_tile_bounds(h, v)

Return the sinusoidal-projection bounds `(x_min, y_max, x_max, y_min)` in metres of
MODIS tile `(h, v)` (`h ∈ 0:35`, `v ∈ 0:17`) — the upper-left and lower-right
corners of the tile on the SIN grid.
"""
function sinusoidal_tile_bounds(h, v)
    x_min = MODIS_GLOBAL_UPPER_LEFT_X + h * MODIS_TILE_SIDE
    y_max = MODIS_GLOBAL_UPPER_LEFT_Y - v * MODIS_TILE_SIDE
    return (x_min, y_max, x_min + MODIS_TILE_SIDE, y_max - MODIS_TILE_SIDE)
end

"""
    sinusoidal_to_longitude_latitude(x, y)

Inverse sinusoidal projection: map SIN coordinates `(x, y)` in metres to
`(longitude, latitude)` in degrees.
"""
function sinusoidal_to_longitude_latitude(x, y)
    latitude = y / MODIS_SPHERE_RADIUS
    longitude = x / (MODIS_SPHERE_RADIUS * cos(latitude))
    return rad2deg(longitude), rad2deg(latitude)
end

"""
    longitude_latitude_to_sinusoidal(longitude, latitude)

Forward sinusoidal projection: map `(longitude, latitude)` in degrees to SIN
coordinates `(x, y)` in metres.
"""
function longitude_latitude_to_sinusoidal(longitude, latitude)
    φ = deg2rad(latitude)
    λ = deg2rad(longitude)
    x = MODIS_SPHERE_RADIUS * λ * cos(φ)
    y = MODIS_SPHERE_RADIUS * φ
    return x, y
end

"""
    cmr_granules_url(short_name, version, bbox; temporal = nothing, page_size = 200)

Build the NASA CMR (Common Metadata Repository) granule-search URL for the product
`short_name` / `version` intersecting the `bbox` [`BoundingBox`](@ref). `bounding_box`
is encoded `W,S,E,N`; `temporal` (an ISO-8601 `start,end` string) is appended for
time-series products. CMR search is anonymous; only the download itself needs
Earthdata credentials.
"""
function cmr_granules_url(short_name, version, bbox::BoundingBox; temporal = nothing, page_size = 200)
    west, east = bbox.longitude
    south, north = bbox.latitude
    url = string("https://cmr.earthdata.nasa.gov/search/granules.json",
                 "?short_name=", short_name,
                 "&version=", version,
                 "&bounding_box=", west, ",", south, ",", east, ",", north,
                 "&page_size=", page_size)
    isnothing(temporal) || (url = string(url, "&temporal=", temporal))
    return url
end

#####
##### Dataset types
#####

"""
    AbstractMODISLandDataset

Supertype for the MODIS land-property datasets served on the sinusoidal grid in
HDF-EOS2 format (albedo, LAI/FPAR, land-cover/PFT). Concrete subtypes carry only
their product-specific pieces (short name, SDS-name map, decode/QA policy); the
shared SIN-grid read scaffold, CMR discovery, Earthdata download, and
`scale × DN` decode discipline live here.
"""
abstract type AbstractMODISLandDataset <: AbstractStaticDataset end

"""
    MCD43Albedo(; diffuse_fraction = 0.2)

MODIS MCD43A3 V061 BRDF/Albedo (500 m, daily 16-day retrieval). Provides
`:albedo` (broadband shortwave) and optionally `:albedo_visible` / `:albedo_nir`,
each blended from the black-sky and white-sky albedos into the blue-sky (actual)
albedo with the diffuse fraction `diffuse_fraction` (`skyl` weight; see
[`bluesky_blend`](@ref)).

Read in regional windows only: build the `Metadatum` with a longitude/latitude
[`BoundingBox`](@ref). The raw HDF-EOS read + sinusoidal→lat/lon reprojection
require `ArchGDAL` (with the HDF4 driver) and Earthdata credentials.

Data source: MCD43A3.061, `10.5067/MODIS/MCD43A3.061`.

```jldoctest
julia> using NumericalEarth

julia> MCD43Albedo()
MCD43Albedo(0.2)
```
"""
struct MCD43Albedo <: AbstractMODISLandDataset
    diffuse_fraction :: Float64
end

MCD43Albedo(; diffuse_fraction = 0.2) = MCD43Albedo(diffuse_fraction)

"""
    MODISLAIDataset

Supertype for the MODIS combined/Terra LAI–FPAR products, which share identical
SDS names, scales, and QA conventions and differ only in cadence and sensor.
"""
abstract type MODISLAIDataset <: AbstractMODISLandDataset end

"""
    MCD15A3H()

MODIS MCD15A3H V061 combined Terra+Aqua LAI/FPAR (500 m, 4-day). Provides
`:leaf_area_index` (and `:fpar`, `:leaf_area_index_uncertainty`). The 4-day cadence
gives the densest seasonal sampling. A single-composite `Field` is supported now;
a seasonal `FieldTimeSeries` is a later refinement.

Data source: MCD15A3H.061, `10.5067/MODIS/MCD15A3H.061`.

```jldoctest
julia> using NumericalEarth

julia> MCD15A3H()
MCD15A3H()
```
"""
struct MCD15A3H <: MODISLAIDataset end

"""
    MCD15A2H()

MODIS MCD15A2H V061 combined Terra+Aqua LAI/FPAR (500 m, 8-day). The 8-day sibling
of [`MCD15A3H`](@ref); use when the 4-day volume is excessive.

Data source: MCD15A2H.061, `10.5067/MODIS/MCD15A2H.061`.
"""
struct MCD15A2H <: MODISLAIDataset end

"""
    MOD15A2H()

MODIS MOD15A2H V061 Terra-only LAI/FPAR (500 m, 8-day). The longest single-sensor
record (from 2000-02-18); use for a long climatology.

Data source: MOD15A2H.061, `10.5067/MODIS/MOD15A2H.061`.
"""
struct MOD15A2H <: MODISLAIDataset end

"""
    MCD12Q1(; legend = :PFT)

MODIS MCD12Q1 V061 land-cover / plant-functional-type (500 m, annual). The `legend`
selects the classification layer: `:PFT` (`LC_Type5`, codes 0–11, the default and
the only product with an explicit PFT legend), `:IGBP` (`LC_Type1`, 1–17), or
`:LCCS1` (`LC_Prop1`). Categorical — aggregated by mode / class fraction, never
averaged (see [`mode_aggregate`](@ref), [`class_fraction`](@ref)). Land cover
changes slowly; treat a chosen year as static (and heed the User Guide's
no-change-detection warning).

Data source: MCD12Q1.061, `10.5067/MODIS/MCD12Q1.061`.

```jldoctest
julia> using NumericalEarth

julia> MCD12Q1()
MCD12Q1(:PFT)
```
"""
struct MCD12Q1 <: AbstractMODISLandDataset
    legend :: Symbol
end

MCD12Q1(; legend = :PFT) = MCD12Q1(legend)

#####
##### Product identity (short name / version) and variable-name maps
#####

modis_short_name(::MCD43Albedo) = "MCD43A3"
modis_short_name(::MCD15A3H)    = "MCD15A3H"
modis_short_name(::MCD15A2H)    = "MCD15A2H"
modis_short_name(::MOD15A2H)    = "MOD15A2H"
modis_short_name(::MCD12Q1)     = "MCD12Q1"

modis_version(::AbstractMODISLandDataset) = "061"

# Albedo needs a *pair* of SDS per variable (black-sky + white-sky), unlike the
# 1:1 maps of the other products, because blue-sky blending reads both.
const MCD43Albedo_variable_names = Dict(
    :albedo         => ("Albedo_BSA_shortwave", "Albedo_WSA_shortwave"),
    :albedo_visible => ("Albedo_BSA_vis",       "Albedo_WSA_vis"),
    :albedo_nir     => ("Albedo_BSA_nir",       "Albedo_WSA_nir"),
)

const MODISLAI_variable_names = Dict(
    :leaf_area_index             => "Lai_500m",
    :fpar                        => "Fpar_500m",
    :leaf_area_index_uncertainty => "LaiStdDev_500m",
)

const MCD12Q1_variable_names = Dict(
    :plant_functional_type => "LC_Type5",
    :landcover_igbp        => "LC_Type1",
    :landcover_lccs        => "LC_Prop1",
    :landcover_confidence  => "LC_Prop1_Assessment",
)

#####
##### Metadatum aliases
#####

const MCD43AlbedoMetadatum = Metadatum{<:MCD43Albedo}
const MODISLAIMetadatum    = Metadatum{<:MODISLAIDataset}
const MCD12Q1Metadatum     = Metadatum{<:MCD12Q1}
const MODISLandMetadatum   = Metadatum{<:AbstractMODISLandDataset}

#####
##### Shared dataset-level specializations
#####

DataWrangling.default_download_directory(::AbstractMODISLandDataset) = download_MODISLand_cache

# The reprojected regional raster is on a plain lat/lon grid; the native hull is the
# global lat/lon extent and `construct_native_grid` restricts it to the BoundingBox.
DataWrangling.longitude_interfaces(::AbstractMODISLandDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::AbstractMODISLandDataset)  = (-90, 90)

# Nominal global size at the 500 m native resolution (≈ 240 cells per degree); only
# the BoundingBox-restricted portion is ever materialized.
Base.size(::AbstractMODISLandDataset, variable) = (86400, 43200, 1)

DataWrangling.available_variables(::MCD43Albedo)         = MCD43Albedo_variable_names
DataWrangling.available_variables(::MODISLAIDataset)     = MODISLAI_variable_names
DataWrangling.available_variables(::MCD12Q1)             = MCD12Q1_variable_names

#####
##### Shared metadatum-level specializations
#####

DataWrangling.is_three_dimensional(::MODISLandMetadatum) = false
DataWrangling.default_inpainting(::MODISLandMetadatum) = nothing
DataWrangling.longitude_name(::MODISLandMetadatum) = "lon"
DataWrangling.latitude_name(::MODISLandMetadatum)  = "lat"
Oceananigans.Fields.location(::MODISLandMetadatum) = (Center, Center, Center)

# Albedo exposes the black-sky SDS as its representative variable name; both SDS
# are read (and blended) in `retrieve_data`.
DataWrangling.dataset_variable_name(md::MCD43AlbedoMetadatum) = first(MCD43Albedo_variable_names[md.name])
DataWrangling.dataset_variable_name(md::MODISLAIMetadatum) = MODISLAI_variable_names[md.name]
DataWrangling.dataset_variable_name(md::MCD12Q1Metadatum)  = MCD12Q1_variable_names[md.name]

#####
##### Region-keyed filenames + coverage validation (require a BoundingBox)
#####

DataWrangling.metadata_filename(dataset::AbstractMODISLandDataset, name, date, region) =
    string(modis_short_name(dataset), "_", string(name), "_",
           date_suffix(date), "_", region_suffix(region), ".nc")

date_suffix(::Nothing) = "static"
date_suffix(date) = Dates.format(DateTime(date), "yyyymmdd")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::MODISLandMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        name = modis_short_name(metadata.dataset)
        error("$(name) must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:$(metadata.name); dataset = $(summary(metadata.dataset)),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

#####
##### retrieve_data — reads the reprojected regional NetCDF of raw DN and applies
##### the pure decode / blend / mask above, returning a physical array on the
##### regional lat/lon grid.
#####

function DataWrangling.retrieve_data(metadata::MCD43AlbedoMetadatum)
    path = metadata_path(metadata)
    black_sky_name, white_sky_name = MCD43Albedo_variable_names[metadata.name]
    f_diff = metadata.dataset.diffuse_fraction

    ds = Dataset(path)
    black_sky = decode_albedo.(ds[black_sky_name][:, :])
    white_sky = decode_albedo.(ds[white_sky_name][:, :])
    close(ds)

    return bluesky_blend.(black_sky, white_sky, f_diff)
end

function DataWrangling.retrieve_data(metadata::MODISLAIMetadatum)
    path = metadata_path(metadata)
    name = MODISLAI_variable_names[metadata.name]

    ds = Dataset(path)
    decoded = metadata.name === :fpar ? decode_fpar.(ds[name][:, :]) : decode_lai.(ds[name][:, :])
    # Keep only main radiative-transfer retrievals if the QA layer was retained.
    # `FparLai_QC` is a bit-packed byte stored as raw DN (Float64 in the NetCDF);
    # round back to a UInt8 before the bitwise QA decode (`&`, `>>`).
    if haskey(ds, "FparLai_QC")
        quality = round.(UInt8, ds["FparLai_QC"][:, :])
        decoded = ifelse.(lai_quality_ok.(quality), decoded, NaN)
    end
    close(ds)

    return decoded
end

function DataWrangling.retrieve_data(metadata::MCD12Q1Metadatum)
    path = metadata_path(metadata)
    name = MCD12Q1_variable_names[metadata.name]

    ds = Dataset(path)
    codes = mask_landcover.(ds[name][:, :])
    close(ds)

    return codes
end

#####
##### Download — the real fetch (CMR discovery, Earthdata GET, SIN→lat/lon warp)
##### lives in ext/NumericalEarthArchGDALExt.jl. The module entry points below
##### fall back to a clear error when that extension is not loaded.
#####

function Downloads.download(metadatum::MODISLandMetadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        modis_granules_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
modis_granules_to_netcdf(metadatum, nc_path) =
    error("Reading MODIS HDF-EOS granules requires ArchGDAL (built with the HDF4 driver) " *
          "and NASA Earthdata credentials. Load ArchGDAL with `using ArchGDAL`, provide " *
          "credentials via EARTHDATA_USERNAME / EARTHDATA_PASSWORD (or a ~/.netrc entry for " *
          "urs.earthdata.nasa.gov), and ensure GDAL_jll was built with the HDF4 driver " *
          "(GDAL.jl issue #84). See future_plans/00_shared_ingestion_infrastructure.md (Part B.3).")

earthdata_cmr_granules(short_name, version, bbox; temporal = nothing) =
    error("Resolving MODIS granule URLs via CMR requires network access; this helper is " *
          "provided by the ArchGDAL extension. Load it with `using ArchGDAL`.")

end # module MODISLand
