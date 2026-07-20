module ASTERGED

export ASTERGEDv3

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       BoundingBox, metadata_path, NearestNeighborInpainting
using ...Radiations: default_water_emissivity

import Oceananigans

download_ASTERGED_cache::String = ""
function __init__()
    global download_ASTERGED_cache = DataWrangling.download_cache("ASTERGED")
end

#####
##### Broadband-emissivity synthesis coefficients
#####

# Narrowband → broadband coefficients for ASTER TIR bands 10–14 (8.3, 8.6, 9.1,
# 10.6, 11.3 µm) over the 8.0–13.5 µm window, from Ogawa & Schmugge (2004),
# Earth Interactions 8(7), doi:10.1175/1087-3562(2004)008<0001:MSBEOT>2.0.CO;2.
# The weights sum to exactly unity, so
# `broadband_emissivity` is a convex combination of the five band emissivities —
# the broadband value stays within their range (hence in [0.7, 1.0] over land).
# Intercept-carrying regressions (e.g. Cheng et al. 2013, ε = 0.197 + Σ cᵢ εᵢ)
# cannot be substituted by renormalizing their slopes: dropping the intercept
# biases broadband ε low by ~0.02 over low-emissivity deserts.
const OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS = [0.088, 0.053, 0.174, 0.380, 0.305]

# AG100 v003 tiles hold 0 = land, 1 = water.
# `fill_water` exposes a `water_code` keyword in case a future tile differs.
const ASTERGED_WATER_CODE = 1

#####
##### Dataset type
#####

"""
    ASTERGEDv3 <: AbstractStaticDataset

ASTER Global Emissivity Dataset (GED) v3: a static (2000–2008 clear-sky mean)
climatology of land-surface emissivity on a plain geographic (WGS84 lat/lon)
grid, distributed as HDF5 in 1°×1° tiles. Two resolutions are supported:

- `:high_100m` — 100 m (3 arcsec, 1000×1000 px/tile). Primary.
- `:low_1km` — 1 km (30 arcsec, 100×100 px/tile). Coarser sibling.

Internally these map to NASA's product short names (`AG100` / `AG1KM`,
which appear in CMR queries and tile filenames.

The dataset provides five narrowband emissivities (ASTER TIR bands 10–14); a
longwave scheme needs a single broadband value, so `retrieve_data` collapses the
five bands to one broadband emissivity using `broadband_coefficients` (a 5-vector;
default [`OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS`](@ref)).

ASTER GED reports no emissivity over water, so water cells (per the tile
land-water map) are filled with the constant `water_emissivity`; retrieval gaps
(persistent cloud, e.g. the humid tropics) decode to `NaN` and are filled by the
default `NearestNeighborInpainting` when building a `Field`. The resulting
emissivity `Field` can be passed directly to `SurfaceRadiationProperties` as its
`emissivity`.

Because ASTER GED is a fine regional-window raster, it is read in regional
windows only: construct the `Metadatum` with a longitude/latitude `BoundingBox`.

Reading the HDF5 tiles requires `ArchGDAL` (with the HDF5 driver) and NASA
Earthdata credentials (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`); that path
lives in `ext/NumericalEarthArchGDALExt.jl`.

Data source: https://www.earthdata.nasa.gov/data/catalog/lpcloud-ag100-003
Reference: Hulley et al. (2015), GRL, doi:10.1002/2015GL065564.
"""
struct ASTERGEDv3 <: AbstractStaticDataset
    resolution :: Symbol                        # :high_100m (100 m) or :low_1km (1 km)
    broadband_coefficients :: Vector{Float64}    # 5-vector for the ε_broadband synthesis
    water_emissivity :: Float64                 # constant substituted where the tile land-water map says water
end

"""
    ASTERGEDv3(; resolution = :high_100m,
                 broadband_coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS,
                 water_emissivity = default_water_emissivity)

Construct an [`ASTERGEDv3`](@ref) dataset. `resolution` is `:high_100m`
(100 m) or `:low_1km` (1 km). `broadband_coefficients` is the 5-band narrowband→broadband
emissivity synthesis vector (default from [Ogawa & Schmugge (2004)](@cite ogawa2004mapping),
8.0–13.5 µm window). `water_emissivity` is the emissivity substituted over water cells, where
ASTER GED has no retrieval; the default is the shared `default_water_emissivity`
(0.97) that the ocean-surface radiation defaults also use.

```jldoctest
julia> using NumericalEarth

julia> ASTERGEDv3()
ASTERGEDv3(resolution = :high_100m)

julia> ASTERGEDv3(resolution = :low_1km)
ASTERGEDv3(resolution = :low_1km)
```
"""
function ASTERGEDv3(; resolution = :high_100m,
                      broadband_coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS,
                      water_emissivity = default_water_emissivity)
    resolution ∈ (:high_100m, :low_1km) ||
        throw(ArgumentError("ASTERGEDv3 resolution must be :high_100m or :low_1km, got $(repr(resolution))"))
    return ASTERGEDv3(resolution, broadband_coefficients, water_emissivity)
end

Base.summary(dataset::ASTERGEDv3) = string("ASTERGEDv3(resolution = :", dataset.resolution, ")")
Base.show(io::IO, dataset::ASTERGEDv3) = print(io, summary(dataset))

const ASTERGEDMetadatum = Metadatum{<:ASTERGEDv3}

# Verbose NumericalEarth name → HDF5 dataset path within the tile.
ASTERGED_dataset_variable_names = Dict(
    :emissivity             => "/Emissivity/Mean",
    :emissivity_uncertainty => "/Emissivity/SDev",
)

"""
    asterged_decode_emissivity(DN)

Decode a raw ASTER GED `/Emissivity/Mean` digital number to a `Float32`
emissivity: fill value −9999 maps to `NaN`, otherwise scale by **0.001**
(`ε = 0.001 · DN`).
"""
@inline asterged_decode_emissivity(DN) = ifelse(DN == -9999, NaN32, 0.001f0 * DN)

"""
    asterged_decode_uncertainty(DN)

Decode a raw ASTER GED `/Emissivity/SDev` digital number to a `Float32`
emissivity standard deviation: fill value −9999 maps to `NaN`, otherwise scale
by **0.0001** (`σ = 1.0e-4 · DN`). Note the scale differs from
[`asterged_decode_emissivity`](@ref)'s 0.001 by 10× — decoding SDev with the
Mean scale is a silent 10× error.
"""
@inline asterged_decode_uncertainty(DN) = ifelse(DN == -9999, NaN32, 1f-4 * DN)

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
in the HDF5 `/Emissivity/Mean` layout) to a broadband `(Nx, Ny)` array — the
array form of [`broadband_emissivity`](@ref), a broadcast + reduction along the
band dimension.
"""
function broadband_emissivity_map(decoded_bands, coefficients)
    FT = eltype(decoded_bands)
    weights = reshape(FT.(coefficients), :, 1, 1)
    return dropdims(sum(weights .* decoded_bands; dims = 1); dims = 1)
end

"""
    broadband_uncertainty_map(decoded_sdev_bands, coefficients)

As [`broadband_emissivity_map`](@ref) but combines per-band standard deviations
(shape `(5, Nx, Ny)`) into a broadband uncertainty `(Nx, Ny)` — the array form
of [`broadband_uncertainty`](@ref).
"""
function broadband_uncertainty_map(decoded_sdev_bands, coefficients)
    FT = eltype(decoded_sdev_bands)
    weights = reshape(FT.(coefficients), :, 1, 1)
    return sqrt.(dropdims(sum(weights .^ 2 .* decoded_sdev_bands .^ 2; dims = 1); dims = 1))
end

"""
    fill_water(field, land_water_map, water_value; water_code = ASTERGED_WATER_CODE)

Return a copy of `field` with cells where `land_water_map == water_code` set to
`water_value` — ASTER GED has no emissivity retrieval over water, and a `NaN`
there would otherwise poison downstream flux kernels. The default
`water_code = 1` matches the ASTER GED `/Land_Water_Map/LWmap` 0/1 (land/water)
coding; see [`ASTERGED_WATER_CODE`](@ref).
"""
function fill_water(field, land_water_map, water_value; water_code = ASTERGED_WATER_CODE)
    water_value = convert(eltype(field), water_value)
    return ifelse.(land_water_map .== water_code, water_value, field)
end

#####
##### Dataset interface
#####

DataWrangling.available_variables(::ASTERGEDv3) = ASTERGED_dataset_variable_names
DataWrangling.default_download_directory(::ASTERGEDv3) = download_ASTERGED_cache

DataWrangling.reversed_latitude_axis(::ASTERGEDv3) = false

# Follow CopernicusDEM: the native lat/lon hull is global integer-degree tile
# boundaries; `construct_native_grid` restricts it to the requested BoundingBox.
DataWrangling.longitude_interfaces(::ASTERGEDv3) = (-180, 180)
DataWrangling.latitude_interfaces(::ASTERGEDv3) = (-90, 90)

# Global pixel counts (Nx, Ny, Nz) used only to set the native resolution Δ;
# `retrieve_data` returns the restricted regional window.
# :high_100m: 1000 px/deg (100 m); :low_1km: 100 px/deg (1 km).
global_pixels(::Val{:high_100m}) = (360_000, 180_000, 1)
global_pixels(::Val{:low_1km})   = (36_000, 18_000, 1)

Base.size(dataset::ASTERGEDv3) = global_pixels(Val(dataset.resolution))
Base.size(dataset::ASTERGEDv3, variable) = size(dataset)

DataWrangling.metadata_filename(dataset::ASTERGEDv3, name, date, region) =
    string("ASTERGED_", dataset.resolution, "_",
           string(name), "_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

# Shared by `validate_dataset_coverage` and `Downloads.download` so the guard
# fires on every load path.
function require_bounded_region(metadata::ASTERGEDMetadatum)
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

DataWrangling.validate_dataset_coverage(grid, metadata::ASTERGEDMetadatum) =
    require_bounded_region(metadata)

#####
##### Metadatum interface
#####

DataWrangling.is_three_dimensional(::ASTERGEDMetadatum) = false
DataWrangling.missing_value(::ASTERGEDMetadatum) = -9999
DataWrangling.dataset_variable_name(metadata::ASTERGEDMetadatum) =
    ASTERGED_dataset_variable_names[metadata.name]

# Retrieval gaps (persistent cloud, screened snow) decode to NaN; inpaint them
# from the nearest valid emissivity. `Inf` iterations so no gap is ever left as
# the zero that a capped inpainting writes into unfilled cells.
DataWrangling.default_inpainting(::ASTERGEDMetadatum) = NearestNeighborInpainting(Inf)
DataWrangling.inpainted_metadata_path(metadata::ASTERGEDMetadatum) =
    joinpath(metadata.dir, string("inpainted_", replace(metadata.filename, ".nc" => ".jld2")))

# Coordinate variable names in the regional NetCDF written by the download step.
DataWrangling.longitude_name(::ASTERGEDMetadatum) = "lon"
DataWrangling.latitude_name(::ASTERGEDMetadatum)  = "lat"

# Emissivity is a surface property: a reduced (`Nothing` z-location) field can be
# indexed at any k, as the interface flux kernels do via `stateindex` at k = Nz.
Oceananigans.Fields.location(::ASTERGEDMetadatum) = (Center, Center, Nothing)

#####
##### Product identity + CMR granule discovery
#####

# NASA CMR short name / version for the ASTER GED product at each resolution
# ("AG" abbreviates ASTER GED).
asterged_short_name(dataset::ASTERGEDv3) = asterged_short_name(Val(dataset.resolution))
asterged_short_name(::Val{:high_100m}) = "AG100"
asterged_short_name(::Val{:low_1km})   = "AG1KM"
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
##### by the download step and applies the pure decode / broadband / water-fill
##### core, returning a physical `(Nx, Ny)` array on the regional lat/lon grid.
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
fill water cells with the dataset's `water_emissivity` (zero uncertainty).
Retrieval gaps stay `NaN` for the downstream inpainting. Returns a regional
`(Nx, Ny)` array.
"""
function DataWrangling.retrieve_data(metadata::ASTERGEDMetadatum)
    path = metadata_path(metadata)
    coefficients = metadata.dataset.broadband_coefficients
    water_emissivity = metadata.dataset.water_emissivity

    ds = DataWrangling.Dataset(path)
    land_water_map = ds[ASTERGED_LWMAP_LAYER][:, :]

    if metadata.name === :emissivity
        mean_dn = ds[ASTERGED_MEAN_LAYER][:, :, :]
        close(ds)
        emissivity = broadband_emissivity_map(asterged_decode_emissivity.(mean_dn), coefficients)
        return fill_water(emissivity, land_water_map, water_emissivity)
    elseif metadata.name === :emissivity_uncertainty
        sdev_dn = ds[ASTERGED_SDEV_LAYER][:, :, :]
        close(ds)
        uncertainty = broadband_uncertainty_map(asterged_decode_uncertainty.(sdev_dn), coefficients)
        return fill_water(uncertainty, land_water_map, 0)
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
    require_bounded_region(metadata)
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
          "https://urs.earthdata.nasa.gov).")

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
earthdata_cmr_granules(short_name, version, bbox) =
    error("Resolving ASTER GED granule URLs via CMR requires network access; this helper " *
          "is provided by the ArchGDAL extension. Load it with `using ArchGDAL`.")

end # module ASTERGED
