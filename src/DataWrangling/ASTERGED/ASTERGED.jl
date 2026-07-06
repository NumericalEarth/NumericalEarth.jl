module ASTERGED

export ASTERGEDv3

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       BoundingBox, Dataset, metadata_path, dataset_variable_name

import Oceananigans

download_ASTERGED_cache::String = ""
function __init__()
    global download_ASTERGED_cache = DataWrangling.download_cache("ASTERGED")
end

#####
##### Broadband-emissivity synthesis coefficients
#####

# Narrowband → broadband regression slope coefficients for ASTER TIR bands
# 10–14 (8.3, 8.6, 9.1, 10.6, 11.3 µm) from the Ogawa et al. (2003) /
# Ogawa & Schmugge broadband-emissivity regression over the 8–13.5 µm window.
# The published regression carries an intercept; we drop it and normalize the
# slopes to sum to unity so that `broadband_emissivity` is a convex combination
# of the five narrowband emissivities — guaranteeing the broadband value stays
# within the range of its inputs (and hence in [0.7, 1.0] over land). This is a
# documented modeling choice; expose the vector as a struct field so it is
# tunable. References: Ogawa et al. (2003), Int. J. Remote Sens.; Cheng et al.
# (2013) narrowband→broadband emissivity regressions.
const OGAWA_2003_SLOPES = (0.025, 0.057, 0.237, 0.333, 0.146)
const OGAWA_2003_BROADBAND_COEFFICIENTS = collect(OGAWA_2003_SLOPES ./ sum(OGAWA_2003_SLOPES))

# Land/Water map coding. The ASTER GED `/Land_Water_Map/LWmap` layer coding is
# documented ambiguously (LP DAAC User Guide vs GEE), so we verified it on a real
# AG100 v003 tile: the `/Land_Water_Map/LWmap` values are 0 (land) and 1 (water)
# — the GEE-style 0/1 coding, NOT the 1/2 coding. On the Grand Canyon tile
# `AG100.v003.37.-112` the only non-zero cells (851 of 10⁶) trace the Colorado
# River, confirming water = 1. Exposed as a keyword in case a future tile differs.
const ASTERGED_WATER_CODE = 1

#####
##### Dataset type
#####

"""
    ASTERGEDv3{R} <: AbstractStaticDataset

ASTER Global Emissivity Dataset (GED) v3: a static (2000–2008 clear-sky mean)
climatology of land-surface emissivity on a plain geographic (WGS84 lat/lon)
grid, distributed as HDF5 in 1°×1° tiles. Two resolutions are supported:

- `:AG100` — 100 m (3 arcsec, 1000×1000 px/tile). Primary, highest resolution.
- `:AG1km` — 1 km (30 arcsec, 100×100 px/tile). Coarser sibling.

The dataset provides five narrowband emissivities (ASTER TIR bands 10–14); a
longwave scheme needs a single broadband value, so `retrieve_data` collapses the
five bands to one broadband emissivity using `broadband_coefficients` (a 5-vector;
default [`OGAWA_2003_BROADBAND_COEFFICIENTS`](@ref)).

Because ASTER GED is a fine regional-window raster, it is read in regional
windows only: construct the `Metadatum` with a longitude/latitude `BoundingBox`.

Reading the HDF5 tiles requires `ArchGDAL` (with the HDF5 driver) and NASA
Earthdata credentials (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`); that path
lives in `ext/NumericalEarthArchGDALExt.jl` and is gated behind a fallback error
when `ArchGDAL` is not loaded. The pure decode/broadband/water-mask core
(`decode_mean`, `decode_sdev`, `broadband_emissivity`) needs neither and is
unit-tested directly.

Data source: https://www.earthdata.nasa.gov/data/catalog/lpcloud-ag100-003
Reference: Hulley et al. (2015), GRL, doi:10.1002/2015GL065564.
"""
struct ASTERGEDv3{R} <: AbstractStaticDataset
    resolution :: Symbol           # :AG100 (100 m) or :AG1km (1 km)
    broadband_coefficients :: R    # 5-vector for the ε_broadband synthesis
end

"""
    ASTERGEDv3(; resolution = :AG100,
                 broadband_coefficients = OGAWA_2003_BROADBAND_COEFFICIENTS)

Construct an [`ASTERGEDv3`](@ref) dataset. `resolution` is `:AG100` (100 m) or
`:AG1km` (1 km). `broadband_coefficients` is the 5-band narrowband→broadband
emissivity synthesis vector (default derived from Ogawa et al. (2003)).

```jldoctest
julia> using NumericalEarth

julia> ASTERGEDv3()
ASTERGEDv3(resolution = :AG100)

julia> ASTERGEDv3(resolution = :AG1km)
ASTERGEDv3(resolution = :AG1km)
```
"""
function ASTERGEDv3(; resolution = :AG100,
                      broadband_coefficients = OGAWA_2003_BROADBAND_COEFFICIENTS)
    resolution ∈ (:AG100, :AG1km) ||
        throw(ArgumentError("ASTERGEDv3 resolution must be :AG100 or :AG1km, got :$resolution"))
    return ASTERGEDv3(resolution, broadband_coefficients)
end

Base.summary(dataset::ASTERGEDv3) = string("ASTERGEDv3(resolution = :", dataset.resolution, ")")
Base.show(io::IO, dataset::ASTERGEDv3) = print(io, summary(dataset))

const ASTERGEDMetadatum = Metadatum{<:ASTERGEDv3}

# Verbose NumericalEarth name → HDF5 dataset path within the tile.
ASTERGED_dataset_variable_names = Dict(
    :emissivity             => "/Emissivity/Mean",
    :emissivity_uncertainty => "/Emissivity/SDev",
)

#####
##### Pure, unit-testable core (no credentials / IO)
#####

"""
    decode_mean(DN)

Decode a raw `/Emissivity/Mean` digital number to emissivity: fill value −9999
maps to `NaN`, otherwise scale by **0.001** (`ε = 0.001 · DN`).
"""
@inline decode_mean(DN) = ifelse(DN == -9999, NaN, 0.001 * DN)

"""
    decode_sdev(DN)

Decode a raw `/Emissivity/SDev` digital number to an emissivity standard
deviation: fill value −9999 maps to `NaN`, otherwise scale by **0.0001**
(`σ = 1.0e-4 · DN`). Note the scale differs from [`decode_mean`](@ref)'s 0.001
by 10× — decoding SDev with the Mean scale is a silent 10× error.
"""
@inline decode_sdev(DN) = ifelse(DN == -9999, NaN, 1.0e-4 * DN)

"""
    broadband_emissivity(ε_vector, coefficients)

Collapse the five decoded narrowband emissivities (ASTER bands 10–14) into a
single broadband emissivity as the dot product with `coefficients`. With
`coefficients` summing to unity this is a convex combination, so the result lies
within the range of the input band emissivities (`NaN` in any band propagates).
"""
broadband_emissivity(ε_vector, coefficients) =
    sum(coefficients[b] * ε_vector[b] for b in eachindex(coefficients))

"""
    broadband_uncertainty(σ_vector, coefficients)

Propagate the five decoded per-band standard deviations to a broadband
uncertainty by linear error propagation under the same broadband weights,
assuming independent band errors: `σ = √(Σ cᵢ² σᵢ²)`.
"""
broadband_uncertainty(σ_vector, coefficients) =
    sqrt(sum(coefficients[b]^2 * σ_vector[b]^2 for b in eachindex(coefficients)))

"""
    broadband_emissivity_map(decoded_bands, coefficients)

Collapse a decoded emissivity array of shape `(5, Nx, Ny)` (band index first, as
in the HDF5 `/Emissivity/Mean` layout) to a broadband `(Nx, Ny)` array via
[`broadband_emissivity`](@ref) along the band dimension.
"""
function broadband_emissivity_map(decoded_bands, coefficients)
    nbands, Nx, Ny = size(decoded_bands)
    result = Array{Float64}(undef, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        result[i, j] = broadband_emissivity(view(decoded_bands, :, i, j), coefficients)
    end
    return result
end

"""
    broadband_uncertainty_map(decoded_sdev_bands, coefficients)

As [`broadband_emissivity_map`](@ref) but combines per-band standard deviations
(shape `(5, Nx, Ny)`) into a broadband uncertainty `(Nx, Ny)` via
[`broadband_uncertainty`](@ref).
"""
function broadband_uncertainty_map(decoded_sdev_bands, coefficients)
    nbands, Nx, Ny = size(decoded_sdev_bands)
    result = Array{Float64}(undef, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        result[i, j] = broadband_uncertainty(view(decoded_sdev_bands, :, i, j), coefficients)
    end
    return result
end

"""
    mask_water(field, land_water_map; water_code = ASTERGED_WATER_CODE)

Return a copy of `field` with cells where `land_water_map == water_code` set to
`NaN`. The default `water_code = 1` matches the ASTER GED `/Land_Water_Map/LWmap`
0/1 (land/water) coding verified on a real AG100 v003 tile; see
[`ASTERGED_WATER_CODE`](@ref).
"""
mask_water(field, land_water_map; water_code = ASTERGED_WATER_CODE) =
    ifelse.(land_water_map .== water_code, NaN, field)

#####
##### Dataset interface (per Part D.2)
#####

DataWrangling.available_variables(::ASTERGEDv3) = ASTERGED_dataset_variable_names
DataWrangling.default_download_directory(::ASTERGEDv3) = download_ASTERGED_cache

# The regional NetCDF written by the download step (see the ArchGDAL extension)
# already stores latitude south→north to match the model grid, so no flip here.
DataWrangling.reversed_latitude_axis(::ASTERGEDv3) = false

# Follow CopernicusDEM: the native lat/lon hull is global integer-degree tile
# boundaries; `construct_native_grid` restricts it to the requested BoundingBox.
DataWrangling.longitude_interfaces(::ASTERGEDv3) = (-180, 180)
DataWrangling.latitude_interfaces(::ASTERGEDv3) = (-90, 90)

# Global pixel counts (Nx, Ny, Nz) used only to set the native resolution Δ;
# `retrieve_data` returns the restricted regional window.
# AG100: 1000 px/deg (100 m); AG1km: 100 px/deg (1 km).
global_pixels(::Val{:AG100}) = (360_000, 180_000, 1)
global_pixels(::Val{:AG1km}) = (36_000, 18_000, 1)

Base.size(dataset::ASTERGEDv3) = global_pixels(Val(dataset.resolution))
Base.size(dataset::ASTERGEDv3, variable) = size(dataset)

resolution_code(::Val{:AG100}) = "AG100"
resolution_code(::Val{:AG1km}) = "AG1km"

DataWrangling.metadata_filename(dataset::ASTERGEDv3, name, date, region) =
    string("ASTERGED_", resolution_code(Val(dataset.resolution)), "_",
           string(name), "_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::ASTERGEDMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("ASTERGEDv3() must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:emissivity; dataset = ASTERGEDv3(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

#####
##### Metadatum interface
#####

DataWrangling.is_three_dimensional(::ASTERGEDMetadatum) = false
DataWrangling.default_inpainting(::ASTERGEDMetadatum) = nothing
DataWrangling.missing_value(::ASTERGEDMetadatum) = -9999
DataWrangling.dataset_variable_name(metadata::ASTERGEDMetadatum) =
    ASTERGED_dataset_variable_names[metadata.name]

# Coordinate variable names in the regional NetCDF written by the download step.
DataWrangling.longitude_name(::ASTERGEDMetadatum) = "lon"
DataWrangling.latitude_name(::ASTERGEDMetadatum)  = "lat"

Oceananigans.Fields.location(::ASTERGEDMetadatum) = (Center, Center, Center)

#####
##### Product identity + CMR granule discovery
#####

# NASA CMR short name / version for the ASTER GED product at each resolution.
asterged_short_name(dataset::ASTERGEDv3) = asterged_short_name(Val(dataset.resolution))
asterged_short_name(::Val{:AG100}) = "AG100"
asterged_short_name(::Val{:AG1km}) = "AG1KM"
asterged_version(::ASTERGEDv3) = "003"

"""
    asterged_cmr_granules_url(short_name, version, bbox; page_size = 200)

Build the NASA CMR (Common Metadata Repository) granule-search URL for the ASTER
GED product `short_name` / `version` whose 1°×1° HDF5 tiles intersect the `bbox`
[`BoundingBox`](@ref) (encoded `W,S,E,N`). CMR search is anonymous; only the tile
download itself needs Earthdata credentials.
"""
function asterged_cmr_granules_url(short_name, version, bbox::BoundingBox; page_size = 200)
    west, east = bbox.longitude
    south, north = bbox.latitude
    return string("https://cmr.earthdata.nasa.gov/search/granules.json",
                  "?short_name=", short_name,
                  "&version=", version,
                  "&bounding_box=", west, ",", south, ",", east, ",", north,
                  "&page_size=", page_size)
end

#####
##### Data retrieval — reads the regional NetCDF of raw digital numbers written
##### by the download step and applies the pure decode / broadband / mask core,
##### returning a physical `(Nx, Ny)` array on the regional lat/lon grid.
#####

# NetCDF variable names for the raw digital-number layers written by the download
# step (see the ArchGDAL extension). Band index is the first dimension.
const ASTERGED_MEAN_LAYER  = "emissivity_mean"
const ASTERGED_SDEV_LAYER  = "emissivity_sdev"
const ASTERGED_LWMAP_LAYER = "land_water_map"

"""
    retrieve_data(metadata::ASTERGEDMetadatum)

Read the regional NetCDF of raw ASTER GED digital numbers for `metadata.region`
(written by the download step; see [`asterged_tiles_to_netcdf`](@ref)), decode the
digital numbers, collapse the five narrowband emissivities to a single broadband
value (or propagate the per-band uncertainty for `:emissivity_uncertainty`), and
mask out water. Returns a regional `(Nx, Ny)` array. The decode/broadband/mask
steps are the pure, unit-tested core.
"""
function DataWrangling.retrieve_data(metadata::ASTERGEDMetadatum)
    path = metadata_path(metadata)
    coefficients = metadata.dataset.broadband_coefficients

    ds = DataWrangling.Dataset(path)
    land_water_map = ds[ASTERGED_LWMAP_LAYER][:, :]

    if metadata.name === :emissivity
        mean_dn = ds[ASTERGED_MEAN_LAYER][:, :, :]
        close(ds)
        emissivity = broadband_emissivity_map(decode_mean.(mean_dn), coefficients)
        return mask_water(emissivity, land_water_map)
    elseif metadata.name === :emissivity_uncertainty
        sdev_dn = ds[ASTERGED_SDEV_LAYER][:, :, :]
        close(ds)
        uncertainty = broadband_uncertainty_map(decode_sdev.(sdev_dn), coefficients)
        return mask_water(uncertainty, land_water_map)
    else
        close(ds)
        error("ASTERGEDv3 does not provide variable :$(metadata.name); " *
              "available variables: $(collect(keys(ASTERGED_dataset_variable_names)))")
    end
end

#####
##### Download — the real fetch (CMR discovery, Earthdata GET, HDF5 subdataset
##### read → regional NetCDF of raw DN) lives in ext/NumericalEarthArchGDALExt.jl.
##### The module entry points below fall back to a clear error when that extension
##### is not loaded.
#####

function Downloads.download(metadata::ASTERGEDMetadatum)
    path = metadata_path(metadata)
    @root if !isfile(path)
        asterged_tiles_to_netcdf(metadata, path)
    end
    return path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
asterged_tiles_to_netcdf(metadata, path) =
    error("Reading ASTER GED HDF5 tiles requires ArchGDAL (built with the HDF5 driver) " *
          "and NASA Earthdata credentials. Load ArchGDAL with `using ArchGDAL`, and provide " *
          "credentials via EARTHDATA_USERNAME / EARTHDATA_PASSWORD (register free at " *
          "https://urs.earthdata.nasa.gov). See future_plans/status/aster-ged_STATUS.md. " *
          "The pure decode/broadband/mask core (decode_mean, decode_sdev, " *
          "broadband_emissivity) works without this path.")

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
earthdata_cmr_granules(short_name, version, bbox) =
    error("Resolving ASTER GED granule URLs via CMR requires network access; this helper " *
          "is provided by the ArchGDAL extension. Load it with `using ArchGDAL`.")

end # module ASTERGED
