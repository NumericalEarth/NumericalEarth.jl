module LandSurfaceTemperature

# High-resolution land surface temperature (LST) as a *supervision target*.
#
# Unlike the boundary-condition grabs (albedo, emissivity, porosity, ...), LST is
# a training/validation target: it supervises the model's diurnal skin
# temperature `Tˡᵃ`. Consequently it is wired toward the loss / observation
# operator (Half B, `H_LST` below), NOT the radiation/flux slots, and its cloud
# gaps are the operator's valid mask — so we never inpaint them
# (`default_inpainting = nothing`, `missing_value = NaN`).
#
# Two products are ingested (Half A):
#   - `GOES_LST`      — geostationary hourly, netCDF4, anonymous AWS. Regular
#                       hourly `all_dates`. Lowest-friction; diurnal-shape backbone.
#   - `ECOSTRESS_L2G` — gridded lat/lon 0.0006° float32 Kelvin HDF5,
#                       Earthdata-gated. *Irregular* `all_dates` (actual overpass
#                       timestamps). High-resolution, all-hours anchor.
#
# The *pure* decode/parse core (GOES DN → Kelvin, ECOSTRESS float32 K + cloud
# mask, granule-timestamp parse) is fully implemented and unit-testable with no
# IO or credentials. Real reads are gated behind ArchGDAL / Earthdata / HDF5 with
# fallback `error(...)`s, mirroring `CopernicusDEM.zarr_to_netcdf`.

export GOES_LST, ECOSTRESS_L2G
export goes_lst, ecostress_lst, granule_timestamp
export LSTObservationOperator, lst_masked_residual

using Dates: DateTime, Day, Hour, Minute, Second, year, hour, minute, second, dayofyear
using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, Metadata, Metadatum, BoundingBox, metadata_path

import Oceananigans

download_LandSurfaceTemperature_cache::String = ""
function __init__()
    global download_LandSurfaceTemperature_cache = DataWrangling.download_cache("LandSurfaceTemperature")
end

#####
##### Pure decode / parse core (no IO, no credentials — the main deliverable)
#####

# GOES-R ABI L2 LST encoding: `LST` is uint16 with scale 0.0025, offset 190.0 K,
# fill 65535; the physically valid retrieval range is 213–343 K.
const GOES_LST_SCALE  = 0.0025
const GOES_LST_OFFSET = 190.0
const GOES_LST_FILL   = 65535
const GOES_LST_VALID_MIN = 213.0
const GOES_LST_VALID_MAX = 343.0

"""
    goes_lst(DN)

Decode a single GOES-R ABI `LST` digital number `DN` (uint16) to Kelvin:

```math
K = 0.0025 · DN + 190.0
```

The fill value (`65535`) and any value falling outside the valid retrieval range
`[213, 343]` K decode to `NaN` (the operator's cloud/no-retrieval gap — never
inpainted).
"""
@inline function goes_lst(DN::Integer)
    K = GOES_LST_SCALE * DN + GOES_LST_OFFSET
    invalid = (DN == GOES_LST_FILL) | (K < GOES_LST_VALID_MIN) | (K > GOES_LST_VALID_MAX)
    return ifelse(invalid, NaN, K)
end

"""
    goes_lst(DN, DQF)

Decode a GOES-R `LST` digital number `DN` with its data-quality flag `DQF`.
`DQF` runs `0` (high quality) … `3` (no retrieval / cloudy); a value of `3` (or
any fill ≥ 3) masks the pixel to `NaN` *before* the scale/offset decode is
trusted. See [`goes_lst(::Integer)`](@ref) for the scale/offset and valid-range
handling of the retained pixels.
"""
@inline function goes_lst(DN::Integer, DQF::Integer)
    masked = DQF >= 3
    return ifelse(masked, NaN, goes_lst(DN))
end

"""
    ecostress_lst(LST, cloud)

Decode a single ECOSTRESS ECO_L2G_LSTE pixel. The gridded product stores `LST`
directly in float32 Kelvin with `NaN` fill (no scale/offset), so the physical
value passes through unchanged — except that a nonzero `cloud` mask flag forces
the pixel to `NaN` (the gap is the operator's valid mask; never inpainted).
`NaN` inputs pass through as `NaN`.
"""
@inline function ecostress_lst(LST::Real, cloud::Integer)
    K = float(LST)
    return ifelse(cloud != 0, oftype(K, NaN), K)
end

"""
    granule_timestamp(name)

Parse the acquisition time embedded in a satellite granule name of the form
`…sYYYYDDDThhmmss…` (GOES-R start-of-scan `s`-tag; ECOSTRESS shares the
`YYYYDDDThhmmss` day-of-year form). Returns a `DateTime`.

The acquisition time must be carried with every LST slice so the observation
operator ([`LSTObservationOperator`](@ref)) can sample the model at the
observation's *local time of day* — comparing without matching the hour compares
apples to oranges.
"""
function granule_timestamp(name::AbstractString)
    m = match(r"s(\d{4})(\d{3})T(\d{2})(\d{2})(\d{2})", name)
    m === nothing &&
        throw(ArgumentError("could not parse an sYYYYDDDThhmmss timestamp from granule name \"$name\""))
    year   = parse(Int, m.captures[1])
    doy    = parse(Int, m.captures[2])
    hour   = parse(Int, m.captures[3])
    minute = parse(Int, m.captures[4])
    second = parse(Int, m.captures[5])
    return DateTime(year) + Day(doy - 1) + Hour(hour) + Minute(minute) + Second(second)
end

#####
##### Dataset types
#####

abstract type AbstractGOESDataset end
abstract type AbstractECOSTRESSDataset end

"""
    GOES_LST(; satellite = :goes16)

GOES-R ABI Level-2 Land Surface Temperature (`ABI-L2-LSTC`), the hourly
geostationary diurnal-shape backbone for LST supervision.

- netCDF4 on the geostationary fixed grid (`goes_imager_projection`); `LST` is
  uint16 (scale `0.0025`, offset `190.0` K, fill `65535`, valid `213–343` K) and
  `DQF` is `0`…`3` (`3` / fill → `NaN`). Decoded by [`goes_lst`](@ref).
- Read from **anonymous** AWS `s3://noaa-goes16|18|19/ABI-L2-LSTC/YYYY/DDD/HH/`
  (`us-east-1`, no-sign-request) — no Earthdata credentials required.
- Because LST is a training target, the cloud/no-retrieval gaps are the
  observation operator's valid mask: `default_inpainting = nothing`,
  `missing_value = NaN`.

`satellite` selects the platform (`:goes16`, `:goes18`, or `:goes19`). Must be
used with a `BoundingBox` region (the full disk is too large to grab).

The anonymous S3 fetch and the geostationary → lat/lon reprojection are gated
behind `ArchGDAL.jl`; the pure `goes_lst` decode needs neither.

```jldoctest
julia> using NumericalEarth

julia> GOES_LST()
GOES_LST(satellite=:goes16)
```
"""
Base.@kwdef struct GOES_LST <: AbstractGOESDataset
    satellite :: Symbol = :goes16
end

"""
    ECOSTRESS_L2G(; version = "002")

ECOSTRESS ECO_L2G_LSTE gridded Land Surface Temperature — the primary
high-resolution (70 m), all-hours LST supervision target.

- HDF5 on a geographic lat/lon `0.0006°` grid; `LST` and `LST_err` are float32
  Kelvin with `NaN` fill (no scale/offset), `cloud` is a uint8 mask. Decoded by
  [`ecostress_lst`](@ref).
- Earthdata Login required; granules are discovered through CMR (concept
  `C2076113037-LPCLOUD`). Because the ISS orbit precesses, overpasses are
  *opportunistic*: `all_dates` returns the actual (irregular) overpass
  timestamps — a deliberate departure from ERA5's regular hourly range.
- Cloud gaps are the operator's valid mask: `default_inpainting = nothing`,
  `missing_value = NaN`.

Must be used with a `BoundingBox` region. The HDF5 read and the Earthdata
download are gated (see [`GOES_LST`](@ref) for the analogous gating rationale);
the pure `ecostress_lst` decode needs neither.

```jldoctest
julia> using NumericalEarth

julia> ECOSTRESS_L2G()
ECOSTRESS_L2G(version="002")
```
"""
Base.@kwdef struct ECOSTRESS_L2G <: AbstractECOSTRESSDataset
    version :: String = "002"
end

Base.show(io::IO, d::GOES_LST) = print(io, "GOES_LST(satellite=:", d.satellite, ")")
Base.show(io::IO, d::ECOSTRESS_L2G) = print(io, "ECOSTRESS_L2G(version=\"", d.version, "\")")

const GOESMetadata{D} = Metadata{<:AbstractGOESDataset, D}
const GOESMetadatum   = Metadatum{<:AbstractGOESDataset}
const ECOSTRESSMetadata{D} = Metadata{<:AbstractECOSTRESSDataset, D}
const ECOSTRESSMetadatum   = Metadatum{<:AbstractECOSTRESSDataset}

const LSTDataset = Union{AbstractGOESDataset, AbstractECOSTRESSDataset}
const LSTMetadatum = Union{GOESMetadatum, ECOSTRESSMetadatum}

#####
##### Variable-name maps
#####

GOES_LST_variable_names = Dict(
    :land_surface_temperature => "LST",
    :data_quality_flag        => "DQF",
)

ECOSTRESS_LST_variable_names = Dict(
    :land_surface_temperature => "LST",
    :lst_uncertainty          => "LST_err",
    :cloud_mask               => "cloud",
)

DataWrangling.available_variables(::AbstractGOESDataset) = GOES_LST_variable_names
DataWrangling.available_variables(::AbstractECOSTRESSDataset) = ECOSTRESS_LST_variable_names

DataWrangling.dataset_variable_name(md::GOESMetadatum) = GOES_LST_variable_names[md.name]
DataWrangling.dataset_variable_name(md::ECOSTRESSMetadatum) = ECOSTRESS_LST_variable_names[md.name]

#####
##### Shared interface
#####

DataWrangling.default_download_directory(::LSTDataset) = download_LandSurfaceTemperature_cache
DataWrangling.reversed_latitude_axis(::LSTDataset) = false

# Both products are ingested as regional 2-D lat/lon windows (GOES after
# geostationary → lat/lon reprojection; ECOSTRESS is already geographic). The
# regional grid is bracketed at read time from the `BoundingBox`, so these hulls
# just declare the products' maximal geographic coverage.
DataWrangling.longitude_interfaces(::AbstractGOESDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::AbstractGOESDataset)  = (-90, 90)
DataWrangling.longitude_interfaces(::AbstractECOSTRESSDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::AbstractECOSTRESSDataset)  = (-52, 52) # ECOSTRESS land coverage
DataWrangling.z_interfaces(::LSTDataset) = (0, 1)

# LST is a 2-D surface field; the true regional (Nx, Ny) is only known once the
# reprojected/subset array is read (reads are gated), so the dataset-level size
# is a placeholder singleton — see `retrieve_data`.
Base.size(::LSTDataset, variable) = (1, 1, 1)

DataWrangling.is_three_dimensional(::LSTMetadatum) = false
DataWrangling.default_inpainting(::LSTMetadatum) = nothing # cloud gaps are the operator's mask
DataWrangling.missing_value(::LSTMetadatum) = NaN
Base.eltype(::LSTMetadatum) = Float32

Oceananigans.Fields.location(::LSTMetadatum) = (Center, Center, Center)

#####
##### `all_dates`
#####

# GOES-R ABI LST is hourly and continuous. GOES-16 operations began 2017-07-10;
# we expose a practical hourly range.
DataWrangling.all_dates(::AbstractGOESDataset, variable) =
    range(DateTime("2017-07-10"), stop = DateTime("2024-12-31"), step = Hour(1))

# ECOSTRESS overpasses are opportunistic (ISS, non-sun-synchronous), so the time
# axis is IRREGULAR — the departure from ERA5's regular range. The real date set
# is discovered from CMR for a region+window (`ecostress_cmr_overpasses`, gated).
# Absent network access we return a small, deliberately *unevenly spaced* stub so
# downstream machinery (irregular-time `FieldTimeSeries`) can be exercised.
DataWrangling.all_dates(::AbstractECOSTRESSDataset, variable) = ecostress_overpass_dates_stub()

function ecostress_overpass_dates_stub()
    return [DateTime(2021, 7, 1,  3, 12, 45),
            DateTime(2021, 7, 3, 14, 51,  8),
            DateTime(2021, 7, 4, 22,  6, 30),
            DateTime(2021, 7, 8,  9, 40, 17)]
end

"""
    ecostress_cmr_overpasses(region, start_date, end_date; version="002")

Discover the actual (irregular) ECOSTRESS ECO_L2G_LSTE overpass timestamps
intersecting `region` over `[start_date, end_date]` by querying NASA's Common
Metadata Repository (CMR). This is the real source of the irregular `all_dates`;
it requires network access and is therefore gated with a fallback `error`.
"""
ecostress_cmr_overpasses(region, start_date, end_date; version = "002") =
    error("ecostress_cmr_overpasses requires network access to NASA CMR " *
          "(https://cmr.earthdata.nasa.gov/search/granules.json). It is not " *
          "available in this environment; the stub overpass dates are used instead.")

#####
##### Region-encoded filenames + coverage validation (per CopernicusDEM)
#####

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

date_suffix(::Nothing) = "alldates"
function date_suffix(date::DateTime)
    y = lpad(year(date), 4, '0')
    d = lpad(dayofyear(date), 3, '0')
    h = lpad(hour(date), 2, '0')
    m = lpad(minute(date), 2, '0')
    s = lpad(second(date), 2, '0')
    return string("s", y, d, "T", h, m, s)
end

goes_prefix(d::GOES_LST) = string("GOES_", uppercase(string(d.satellite)), "_LST")
ecostress_prefix(d::ECOSTRESS_L2G) = string("ECOSTRESS_L2G_v", d.version)

DataWrangling.metadata_filename(dataset::GOES_LST, name, date, region) =
    string(goes_prefix(dataset), "_", date_suffix(date), "_", region_suffix(region), ".nc")

DataWrangling.metadata_filename(dataset::ECOSTRESS_L2G, name, date, region) =
    string(ecostress_prefix(dataset), "_", name, "_", date_suffix(date), "_", region_suffix(region), ".nc")

function DataWrangling.validate_dataset_coverage(grid, metadata::LSTMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("$(typeof(metadata.dataset)) must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:land_surface_temperature; dataset = $(typeof(metadata.dataset))(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))")
    end
    return nothing
end

#####
##### Download + read (gated; fallback errors mirror CopernicusDEM.zarr_to_netcdf)
#####

function Downloads.download(metadatum::GOESMetadatum)
    path = metadata_path(metadatum)
    @root if !isfile(path)
        download_goes_granule(metadatum, path)
    end
    return path
end

function Downloads.download(metadatum::ECOSTRESSMetadatum)
    path = metadata_path(metadatum)
    @root if !isfile(path)
        download_ecostress_granule(metadatum, path)
    end
    return path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded. The
# fallbacks below fire only when the extension / network path is not available.
download_goes_granule(metadatum, path) =
    error("Fetching GOES-R ABI-L2-LSTC granules from anonymous AWS " *
          "(s3://noaa-$(metadatum.dataset.satellite)/ABI-L2-LSTC/, us-east-1, no-sign-request) " *
          "requires the ArchGDAL package and network access. Load it with `using ArchGDAL`.")

read_goes_lst_lonlat(metadatum, path) =
    error("Reading GOES-R LST requires ArchGDAL.jl to reproject the geostationary " *
          "fixed grid to lat/lon (geos → EPSG:4326). Load it with `using ArchGDAL`.")

# ECOSTRESS L2G is HDF5; `HDF5.jl` is not a project dependency (and must not be
# added — AGENTS.md rule 10), so the read is routed through GDAL's HDF5 driver in
# the ArchGDAL extension. The download is Earthdata-gated.
download_ecostress_granule(metadatum, path) =
    error("Fetching ECOSTRESS ECO_L2G_LSTE granules requires NASA Earthdata " *
          "credentials (EARTHDATA_USERNAME / EARTHDATA_PASSWORD or a .netrc entry) " *
          "and CMR discovery of overpasses — not available in this environment.")

read_ecostress_l2g(metadatum, path) =
    error("Reading ECOSTRESS ECO_L2G_LSTE (HDF5) requires either HDF5.jl + Earthdata " *
          "credentials, or ArchGDAL.jl via GDAL's HDF5 driver. HDF5.jl is not a " *
          "project dependency; load ArchGDAL with `using ArchGDAL` to enable the read.")

DataWrangling.retrieve_data(metadatum::GOESMetadatum) =
    read_goes_lst_lonlat(metadatum, metadata_path(metadatum))

DataWrangling.retrieve_data(metadatum::ECOSTRESSMetadatum) =
    read_ecostress_l2g(metadatum, metadata_path(metadatum))

#####
##### Half B — the `H_LST` observation operator (SCAFFOLD)
#####

"""
    lst_masked_residual(T_model, LST_obs, LST_err, cloudy)

The pure kernel of the LST observation operator: form the variance-normalized
residual between the modeled skin temperature `T_model` and the observed LST
`LST_obs`, weighted by the observation uncertainty `LST_err`.

The observation's cloud/QC mask is applied **before** the residual is formed: a
pixel that is `cloudy`, has `NaN` observed LST, or has non-positive `LST_err`
contributes exactly zero — it is *not* a valid comparison (LST only exists under
clear sky). Retained pixels contribute `(T_model − LST_obs) / LST_err`.

This mirrors the masked, variance-normalized residual used by
`DatasetRestoring`, specialized to a land `Tˡᵃ` supervision target.
"""
@inline function lst_masked_residual(T_model, LST_obs, LST_err, cloudy::Bool)
    valid = (!cloudy) & !isnan(LST_obs) & (LST_err > 0)
    normalized = (T_model - LST_obs) / LST_err
    return ifelse(valid, normalized, zero(normalized))
end

"""
    LSTObservationOperator(lst_observations, cloud_mask, lst_uncertainty, variable_name; rate = 1)

**Scaffold** for the LST observation operator `H_LST` (Half B of the LST-target
plan). It bundles the ingredients needed to compare a model's diurnal skin
temperature `Tˡᵃ` against high-resolution LST observations:

- `lst_observations`: an irregular-time `FieldTimeSeries` of observed LST (from
  [`GOES_LST`](@ref) / [`ECOSTRESS_L2G`](@ref)), each slice tagged with its
  acquisition time (see [`granule_timestamp`](@ref)).
- `cloud_mask`: the observation cloud/QC mask (`true`/nonzero ⇒ drop).
- `lst_uncertainty`: `LST_err` for variance normalization.
- `variable_name`: the model field to supervise (the skin temperature `Tˡᵃ`).
- `rate`: an optional scalar loss weight.

Applying the operator would, for each observation:

1. sample the model `Tˡᵃ(t_obs)` at the observation's acquisition time and
   locations (the model must be evaluated at the observation's *local hour*);
2. apply the observation's cloud/QC mask (drop cloudy / low-quality pixels);
3. form the masked, variance-normalized residual via [`lst_masked_residual`](@ref);
4. feed it into the trajectory loss.

!!! note "Scaffold status"
    Steps 1–3's pure arithmetic is implemented in [`lst_masked_residual`](@ref)
    and unit-tested. Full wiring into a live model's loss (sampling the running
    model at `t_obs`, accumulating over a trajectory, and extending
    `restoring.jl`'s masked-residual machinery to a land `Tˡᵃ` variable) is
    out of scope for this ingestion PR and is left as documented next steps.
"""
struct LSTObservationOperator{FTS, M, E, V, R}
    lst_observations :: FTS
    cloud_mask :: M
    lst_uncertainty :: E
    variable_name :: V
    rate :: R
end

LSTObservationOperator(lst_observations, cloud_mask, lst_uncertainty, variable_name; rate = 1) =
    LSTObservationOperator(lst_observations, cloud_mask, lst_uncertainty, variable_name, rate)

function Base.show(io::IO, H::LSTObservationOperator)
    print(io, "LSTObservationOperator (H_LST) [scaffold]:", '\n',
              "├── variable_name: ", H.variable_name, '\n',
              "├── rate: ", H.rate, '\n',
              "├── lst_observations: ", summary(H.lst_observations), '\n',
              "├── cloud_mask: ", summary(H.cloud_mask), '\n',
              "└── lst_uncertainty: ", summary(H.lst_uncertainty))
end

end # module LandSurfaceTemperature
