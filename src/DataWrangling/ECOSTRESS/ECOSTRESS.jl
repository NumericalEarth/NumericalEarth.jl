module ECOSTRESS

# ECOSTRESS land surface temperature (LST) ingestion.
#
# ECOSTRESS is a thermal-infrared radiometer on the ISS; its ECO_L2G_LSTE product
# gives ~70 m gridded skin temperature — the high-resolution, all-hours diurnal
# `Tˡᵃ` supervision target a ~100 m coupled LES needs. Because it is a *training
# target* (not a boundary-condition grab like albedo or porosity), cloud/no-
# retrieval gaps are the operator's valid mask, so they are never inpainted
# (`default_inpainting = nothing`, `missing_value = NaN`).
#
# The ISS orbit precesses, so overpasses are opportunistic: there is no regular
# `all_dates` range. Overpass timestamps intersecting a region and time window are
# discovered from NASA's Common Metadata Repository (CMR) with
# [`ecostress_overpasses`](@ref); those explicit dates then drive `Metadata`.
#
# The pure decode/parse core (`ecostress_lst`, `granule_timestamp`) and the CMR
# discovery need neither credentials nor GDAL. The granule read is HDF5, routed
# through GDAL's HDF5 driver in `ext/NumericalEarthArchGDALExt.jl` (HDF5.jl is not
# a project dependency); the module keeps a fallback `error` mirroring
# `CopernicusDEM.zarr_to_netcdf`.

export ECOSTRESSL2G, ecostress_lst, granule_timestamp, ecostress_overpasses

using Dates: Dates, DateTime
using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, Metadata, Metadatum, BoundingBox, Dataset,
                       metadata_path, netrc_downloader

import Oceananigans

download_ECOSTRESS_cache::String = ""
function __init__()
    global download_ECOSTRESS_cache = DataWrangling.download_cache("ECOSTRESS")
    return nothing
end

#####
##### Pure decode / parse core (no IO, no credentials)
#####

"""
    ecostress_lst(LST, cloud)

Decode a single ECOSTRESS ECO_L2G_LSTE pixel. The gridded product stores `LST`
directly in Kelvin (no scale/offset). The pixel decodes to `NaN` — the operator's
cloud/no-retrieval gap, never inpainted — when the `cloud` mask flag is nonzero,
when `LST` is the `0` fill (or otherwise non-positive), or when `LST` is `NaN`;
otherwise the Kelvin value passes through unchanged.
"""
@inline function ecostress_lst(LST::Real, cloud::Integer)
    K = float(LST)
    invalid = (cloud != 0) | !(K > 0)
    return ifelse(invalid, oftype(K, NaN), K)
end

"""
    granule_timestamp(name)

Parse the acquisition time embedded in an ECOSTRESS granule name of the form
`…YYYYMMDDThhmmss…` (e.g. `ECOv002_L2G_LSTE_16928_005_20210701T082749_0710_01`),
returning a `DateTime`. Every LST slice carries its acquisition time so the
observation operator can sample the model at the observation's local time of day.
"""
function granule_timestamp(name::AbstractString)
    m = match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", name)
    m === nothing &&
        throw(ArgumentError("could not parse a YYYYMMDDThhmmss timestamp from granule name \"$name\""))
    return DateTime(parse.(Int, m.captures)...)
end

#####
##### Dataset type
#####

abstract type AbstractECOSTRESSDataset end

"""
    ECOSTRESSL2G(; version = "002")

ECOSTRESS ECO_L2G_LSTE gridded Land Surface Temperature — a ~70 m, all-hours skin
temperature (`Tˡᵃ`) supervision target for atmosphere-coupled LES.

- HDF5 on a geographic lon/lat grid; `LST` and `LST_err` are Kelvin and `cloud` is
  a mask flag. Decoded by [`ecostress_lst`](@ref).
- NASA Earthdata Login required (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, the
  variables the `earthaccess` library also honours — see this module's README).
  Granules are discovered from CMR (short name `ECO_L2G_LSTE`), so the time axis is
  *irregular*: [`ecostress_overpasses`](@ref) returns the actual overpass times.
- Must be used with a lon/lat [`BoundingBox`](@ref) region; the swath footprint is
  windowed at download time.
- Cloud/no-retrieval gaps are the operator's valid mask: `default_inpainting =
  nothing`, `missing_value = NaN`.

The HDF5 read is gated behind `ArchGDAL.jl` (GDAL HDF5 driver); the pure
`ecostress_lst` decode and the CMR discovery need neither ArchGDAL nor credentials.

```jldoctest
julia> using NumericalEarth

julia> ECOSTRESSL2G()
ECOSTRESSL2G(version="002")
```
"""
Base.@kwdef struct ECOSTRESSL2G <: AbstractECOSTRESSDataset
    version :: String = "002"
end

Base.show(io::IO, dataset::ECOSTRESSL2G) = print(io, "ECOSTRESSL2G(version=\"", dataset.version, "\")")

const ECOSTRESSMetadata{D} = Metadata{<:AbstractECOSTRESSDataset, D}
const ECOSTRESSMetadatum   = Metadatum{<:AbstractECOSTRESSDataset}

#####
##### Variables
#####

# NumericalEarth names → the layers written into the local regional NetCDF at
# download time (the ECO_L2G_LSTE `SDS/LST` and `SDS/LST_err` fields). The `cloud`
# mask is applied to `LST` at decode time, so it is not a separately read variable.
const ECOSTRESS_variable_names = Dict(:land_surface_temperature => "LST",
                                      :lst_uncertainty          => "LST_err")

DataWrangling.available_variables(::AbstractECOSTRESSDataset) = ECOSTRESS_variable_names
DataWrangling.dataset_variable_name(metadata::ECOSTRESSMetadatum) = ECOSTRESS_variable_names[metadata.name]

#####
##### Grid traits
#####

DataWrangling.default_download_directory(::AbstractECOSTRESSDataset) = download_ECOSTRESS_cache
DataWrangling.reversed_latitude_axis(::AbstractECOSTRESSDataset) = false
DataWrangling.is_three_dimensional(::ECOSTRESSMetadatum) = false
DataWrangling.default_inpainting(::ECOSTRESSMetadatum) = nothing
DataWrangling.missing_value(::ECOSTRESSMetadatum) = NaN
Base.eltype(::ECOSTRESSMetadatum) = Float32

# The regional NetCDF written at download time stores its coordinates as `lon`/`lat`.
DataWrangling.longitude_name(::ECOSTRESSMetadatum) = "lon"
DataWrangling.latitude_name(::ECOSTRESSMetadatum)  = "lat"

Oceananigans.Fields.location(::ECOSTRESSMetadatum) = (Center, Center, Center)

# ECO_L2G_LSTE is posted on a 0.0006° (~70 m) geographic grid; the ISS covers land
# out to ≈±52° latitude. These hulls and the nominal size just declare the maximal
# coverage at native resolution — `construct_native_grid` clips them to the region's
# `BoundingBox`, and only the windowed portion is ever materialized.
const ECOSTRESS_RESOLUTION = 0.0006

DataWrangling.longitude_interfaces(::AbstractECOSTRESSDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::AbstractECOSTRESSDataset)  = (-52, 52)
DataWrangling.z_interfaces(::AbstractECOSTRESSDataset) = (0, 1)

Base.size(::AbstractECOSTRESSDataset, variable) =
    (round(Int, 360 / ECOSTRESS_RESOLUTION), round(Int, 104 / ECOSTRESS_RESOLUTION), 1)

#####
##### Dates — irregular overpasses discovered from CMR
#####

function DataWrangling.all_dates(::AbstractECOSTRESSDataset, variable)
    error("ECOSTRESS overpasses are irregular (the ISS orbit precesses), so there is no " *
          "dataset-wide date range. Discover the overpasses intersecting your region and " *
          "time window with `ecostress_overpasses(region, start_date, end_date)` and pass " *
          "them as explicit `dates` to `Metadata` (or a single `date` to `Metadatum`).")
end

cmr_time(date::DateTime) = string(Dates.format(date, "yyyy-mm-ddTHH:MM:SS"), "Z")

"""
    ecostress_cmr_url(version, region, start_date, end_date; page_size = 200)

Build the NASA CMR granule-search URL (JSON) for `ECO_L2G_LSTE.<version>` granules
intersecting the lon/lat [`BoundingBox`](@ref) `region` over `[start_date, end_date]`.
Pure — constructs the query string only.
"""
function ecostress_cmr_url(version, region::BoundingBox, start_date, end_date; page_size = 200)
    (!isnothing(region.longitude) && !isnothing(region.latitude)) ||
        throw(ArgumentError("ecostress_cmr_url requires a bounded (longitude, latitude) BoundingBox."))
    west, east = region.longitude
    south, north = region.latitude
    return string("https://cmr.earthdata.nasa.gov/search/granules.json",
                  "?short_name=ECO_L2G_LSTE",
                  "&version=", version,
                  "&bounding_box=", west, ",", south, ",", east, ",", north,
                  "&temporal=", cmr_time(start_date), ",", cmr_time(end_date),
                  "&page_size=", page_size)
end

"""
    ecostress_cmr_granules(version, region, start_date, end_date)

Query CMR and return the `(timestamp, download_url)` pairs of the `ECO_L2G_LSTE`
granules intersecting `region` over `[start_date, end_date]`, sorted by time.
Requires network access. The granule `.h5` URLs and their `YYYYMMDDThhmmss`
acquisition tags are read straight out of the JSON response text.
"""
function ecostress_cmr_granules(version, region::BoundingBox, start_date, end_date)
    url = ecostress_cmr_url(version, region, start_date, end_date)
    prefix = "ECOv" * version * "_L2G_LSTE_"
    granules = Tuple{DateTime, String}[]

    mktempdir() do tmp
        json = joinpath(tmp, "cmr.json")
        Downloads.download(url, json)
        text = read(json, String)
        pattern = Regex("https://[^\"]+/(" * prefix * "[^\"/]+)\\.h5")
        for m in eachmatch(pattern, text)
            granule = m.captures[1]
            occursin("_mvs", granule) && continue   # skip the median-view quicklook aux
            t = granule_timestamp(granule)
            push!(granules, (t, m.match))
        end
    end

    unique!(granules)
    sort!(granules; by = first)
    return granules
end

"""
    ecostress_overpasses(region, start_date, end_date; version = "002")

Discover the actual (irregular) ECOSTRESS ECO_L2G_LSTE overpass times intersecting
the lon/lat [`BoundingBox`](@ref) `region` over `[start_date, end_date]`, by querying
NASA's Common Metadata Repository. Returns a sorted `Vector{DateTime}`; pass it as
`dates` to [`Metadata`](@ref) to build an irregular-time series. Requires network
access.
"""
ecostress_overpasses(region::BoundingBox, start_date, end_date; version = "002") =
    first.(ecostress_cmr_granules(version, region, start_date, end_date))

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
date_suffix(date::DateTime) = string("s", Dates.format(date, "yyyymmddTHHMMSS"))

# One local file per overpass and region holds every variable (LST, LST_err), so the
# filename is keyed by date and region but not by variable — mirroring the shared-file
# scheme in CopernicusLandAlbedo.
DataWrangling.metadata_filename(dataset::ECOSTRESSL2G, name, date, region) =
    string("ECOSTRESS_L2G_LSTE_v", dataset.version, "_",
           date_suffix(date), "_", region_suffix(region), ".nc")

function DataWrangling.validate_dataset_coverage(grid, metadata::ECOSTRESSMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("ECOSTRESSL2G must be used with a bounded region. Build the metadatum with a " *
              "longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:land_surface_temperature; dataset = ECOSTRESSL2G(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)),\n" *
              "                          date = DateTime(2021, 7, 1, 8, 27, 49))")
    end
    return nothing
end

#####
##### Download + read
#####
##### Each overpass is fetched, decoded (via the pure `ecostress_lst` core), and
##### clipped to a clean regional lon/lat NetCDF (`lon`, `lat`, `LST`/`LST_err` in
##### Kelvin with `NaN` gaps) at download time — mirroring the CopernicusLandAlbedo /
##### MODIS-style ingest — so the generic `Field` / `set_region_data!` machinery
##### brackets that raster onto the native grid. The real fetch (Earthdata download +
##### GDAL HDF5 read) lives in `ext/NumericalEarthArchGDALExt.jl`; the fallback below
##### fires only when `ArchGDAL` is not loaded, mirroring `CopernicusDEM.zarr_to_netcdf`.
#####

function Downloads.download(metadatum::ECOSTRESSMetadatum)
    path = metadata_path(metadatum)
    @root if !isfile(path)
        ecostress_granule_to_netcdf(metadatum, path)
    end
    return path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
ecostress_granule_to_netcdf(metadatum, path) =
    error("Fetching ECOSTRESS ECO_L2G_LSTE granules requires ArchGDAL.jl (for the GDAL " *
          "HDF5 driver), NASA Earthdata credentials (EARTHDATA_USERNAME / EARTHDATA_PASSWORD), " *
          "and CMR discovery of overpasses. Load ArchGDAL with `using ArchGDAL`.")

"""
    earthdata_download(url, path)

Download `url` to `path` through a temporary `.netrc` authenticated against
`urs.earthdata.nasa.gov`, using the `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`
environment variables. Errors if the credentials are unset.
"""
function earthdata_download(url, path)
    username = get(ENV, "EARTHDATA_USERNAME", nothing)
    password = get(ENV, "EARTHDATA_PASSWORD", nothing)
    (isnothing(username) || isnothing(password)) &&
        error("NASA Earthdata credentials not found. Set EARTHDATA_USERNAME and " *
              "EARTHDATA_PASSWORD (register free at https://urs.earthdata.nasa.gov). " *
              "See NumericalEarth.jl/src/DataWrangling/ECOSTRESS/README.md for instructions.")
    mktempdir() do tmp
        downloader = netrc_downloader(username, password, "urs.earthdata.nasa.gov", tmp)
        Downloads.download(url, path; downloader)
    end
    return path
end

# The download step decodes each layer into Kelvin (with `NaN` gaps) at write time,
# so the read is the generic 2-D NetCDF slurp — no scale/offset, no vertical axis.
function DataWrangling.retrieve_data(metadata::ECOSTRESSMetadatum)
    path = metadata_path(metadata)
    name = DataWrangling.dataset_variable_name(metadata)
    data = Dataset(path) do ds
        Float32.(ds[name][:, :])
    end
    return reshape(data, size(data, 1), size(data, 2), 1)
end

end # module ECOSTRESS
