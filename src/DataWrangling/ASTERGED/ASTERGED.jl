module ASTERGED

export ASTERGEDv3

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       BoundingBox, metadata_path, NearestNeighborInpainting

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
# The weights sum to unity so the broadband value is a convex combination of the
# band emissivities. NOTE these are regressed over the Sahara Desert and applied
# globally here — an extrapolation over humid/vegetated surfaces (see the
# `ASTERGEDv3` docstring). A global multi-biome fit would be more defensible.
const OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS = [0.088, 0.053, 0.174, 0.380, 0.305]

#####
##### Dataset type
#####

"""
    ASTERGEDv3 <: AbstractStaticDataset

ASTER Global Emissivity Dataset (GED) v3: a static (2000–2008 clear-sky mean)
climatology of land-surface emissivity on a plain geographic (WGS84 lat/lon)
grid, distributed as HDF5 in 1°×1° tiles. Two resolutions are supported:

- `:low_1km` — 1 km (36 arcsec, 100×100 px/tile). Default; matches typical
  Earth-system grids without fetching ~100× the data.
- `:high_100m` — 100 m (3.6 arcsec, 1000×1000 px/tile). For sub-km domains.

Internally these map to NASA's product short names (`AG100` / `AG1KM`), which
appear in CMR queries and tile filenames.

ASTER GED provides five narrowband emissivities (TIR bands 10–14). A longwave
scheme needs a single broadband value, so the download step collapses the five
bands to one broadband emissivity using the [Ogawa & Schmugge (2004)](@cite ogawa2004mapping)
coefficients. Cells with no retrieval (persistent cloud, screened snow) and cells
over water — ASTER GED reports no emissivity over water — decode to `NaN` and are
filled by the default `NearestNeighborInpainting` when building a `Field`, so the
returned field is finite everywhere. The emissivity `Field` can be passed directly
to `SurfaceRadiationProperties` as its `emissivity`.

Because ASTER GED is a fine regional-window raster, it is read in regional windows
only: build the `Metadatum` with a longitude/latitude `BoundingBox`, most simply
derived from the model grid,

    region = BoundingBox(grid; padding = default_horizontal_padding(ASTERGEDv3()))

Reading the HDF5 tiles requires `ArchGDAL` (with the HDF5 driver) and NASA
Earthdata credentials (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`); that path
lives in `ext/NumericalEarthArchGDALExt.jl`.

!!! note "Limitations"
    The broadband coefficients are a Sahara-desert regression applied globally.
    Being a static clear-sky mean, the product also misses temporal effects on the
    model's own timescale — soil-moisture-driven emissivity changes during rain/TC
    events, vegetation phenology, and snow (ε is 0.03–0.08 higher under snow) — and
    the clear-sky sampling biases it toward dry states. Over water the returned
    values are inpainted land, not physical; a coupled model should override
    emissivity where its own mask says ocean.

Data source: https://www.earthdata.nasa.gov/data/catalog/lpcloud-ag100-003
Reference: Hulley et al. (2015), GRL, doi:10.1002/2015GL065564.
"""
struct ASTERGEDv3 <: AbstractStaticDataset
    resolution :: Symbol   # :low_1km (1 km) or :high_100m (100 m)
end

"""
    ASTERGEDv3(; resolution = :low_1km)

Construct an [`ASTERGEDv3`](@ref) dataset. `resolution` is `:low_1km` (1 km,
default) or `:high_100m` (100 m).

```jldoctest
julia> using NumericalEarth

julia> ASTERGEDv3()
ASTERGEDv3(resolution = :low_1km)

julia> ASTERGEDv3(resolution = :high_100m)
ASTERGEDv3(resolution = :high_100m)
```
"""
function ASTERGEDv3(; resolution = :low_1km)
    resolution ∈ (:high_100m, :low_1km) ||
        throw(ArgumentError("ASTERGEDv3 resolution must be :high_100m or :low_1km, got $(repr(resolution))"))
    return ASTERGEDv3(resolution)
end

Base.summary(dataset::ASTERGEDv3) = string("ASTERGEDv3(resolution = :", dataset.resolution, ")")
Base.show(io::IO, dataset::ASTERGEDv3) = print(io, summary(dataset))

const ASTERGEDMetadatum = Metadatum{<:ASTERGEDv3}

# Variable name in the regional NetCDF written by the download step (which stores
# the already-decoded broadband floats — see the ArchGDAL extension).
const ASTERGED_variable_names = Dict(
    :emissivity             => "emissivity",
    :emissivity_uncertainty => "emissivity_uncertainty",
)

#####
##### Pure decode / broadband synthesis core (no credentials / IO)
#####

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
by **0.0001** (`σ = 1.0e-4 · DN`). The scale differs from
`asterged_decode_emissivity`'s 0.001 by 10× — decoding SDev with the Mean scale
is a silent 10× error.
"""
@inline asterged_decode_uncertainty(DN) = ifelse(DN == -9999, NaN32, 1f-4 * DN)

"""
    broadband_emissivity(ε_vector, coefficients)

Collapse the five decoded narrowband emissivities (ASTER bands 10–14) to a single
broadband emissivity as the dot product with `coefficients` (`NaN` in any band
propagates).
"""
broadband_emissivity(ε_vector, coefficients) =
    sum(coefficients[b] * ε_vector[b] for b in eachindex(coefficients))

"""
    broadband_uncertainty(σ_vector, coefficients)

Propagate the five per-band standard deviations to a broadband uncertainty as the
**fully-correlated** upper bound `σ = Σ cᵢ σᵢ`. ASTER TES band emissivities share
one temperature retrieval and atmospheric correction, so their errors are strongly
positively correlated.
"""
broadband_uncertainty(σ_vector, coefficients) =
    sum(coefficients[b] * σ_vector[b] for b in eachindex(coefficients))

"""
    broadband_map(decoded_bands, coefficients)

Collapse a decoded `(5, Nx, Ny)` array (band index first, as in the HDF5
`/Emissivity/*` layout) to a broadband `(Nx, Ny)` array — the array form of
`broadband_emissivity` / `broadband_uncertainty` (both are the same nonnegative
linear combination). `NaN` in any band propagates.
"""
function broadband_map(decoded_bands, coefficients)
    FT = eltype(decoded_bands)
    weights = reshape(FT.(coefficients), :, 1, 1)
    return dropdims(sum(weights .* decoded_bands; dims = 1); dims = 1)
end

"""
    place_tile!(field, tile_values, tile_longitude, tile_latitude, longitude, latitude)

Block-copy a decoded tile's cells onto the analytic regional grid `(longitude,
latitude)` — both uniform and (for ASTER GED's integer-degree tiles) aligned — by
mapping each tile-cell center to its native index. `NaN` tile cells are skipped so
a valid value from an adjacent tile at a shared boundary is not overwritten. Cells
outside the tile's footprint are left untouched. Value-based indexing, so it does
not care whether the tile stores latitude ascending or descending.
"""
function place_tile!(field, tile_values, tile_longitude, tile_latitude, longitude, latitude)
    Nx, Ny = size(field)
    Δλ = (longitude[end] - longitude[1]) / max(length(longitude) - 1, 1)
    Δφ = (latitude[end]  - latitude[1])  / max(length(latitude)  - 1, 1)
    for (jl, φ) in enumerate(tile_latitude)
        jr = round(Int, (φ - latitude[1]) / Δφ) + 1
        (1 ≤ jr ≤ Ny) || continue
        for (il, λ) in enumerate(tile_longitude)
            ic = round(Int, (λ - longitude[1]) / Δλ) + 1
            (1 ≤ ic ≤ Nx) || continue
            v = tile_values[il, jl]
            isnan(v) || (@inbounds field[ic, jr] = v)
        end
    end
    return field
end

#####
##### Dataset interface
#####

DataWrangling.available_variables(::ASTERGEDv3) = ASTERGED_variable_names
DataWrangling.default_download_directory(::ASTERGEDv3) = download_ASTERGED_cache

# A couple of native 1 km cells of margin for interpolation stencils at the edge.
DataWrangling.default_horizontal_padding(::ASTERGEDv3) = 0.02

DataWrangling.reversed_latitude_axis(::ASTERGEDv3) = false

# Follow CopernicusDEM: the native lat/lon hull is the global integer-degree tile
# boundaries; `construct_native_grid` restricts it to the requested BoundingBox.
DataWrangling.longitude_interfaces(::ASTERGEDv3) = (-180, 180)
DataWrangling.latitude_interfaces(::ASTERGEDv3) = (-90, 90)

# Global pixel counts (Nx, Ny, Nz) set only the native resolution Δ; the download
# returns the restricted regional window.
# :high_100m: 1000 px/deg (100 m); :low_1km: 100 px/deg (1 km).
global_pixels(::Val{:high_100m}) = (360_000, 180_000, 1)
global_pixels(::Val{:low_1km})   = (36_000, 18_000, 1)

Base.size(dataset::ASTERGEDv3) = global_pixels(Val(dataset.resolution))
Base.size(dataset::ASTERGEDv3, variable) = size(dataset)

# Region-keyed but variable-independent: one regional NetCDF holds both the
# emissivity and its uncertainty (the tile download produces both at once).
DataWrangling.metadata_filename(dataset::ASTERGEDv3, name, date, region) =
    string("ASTERGED_", dataset.resolution, "_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

# Shared by `validate_dataset_coverage` and `Downloads.download` so the guard fires
# on every load path.
function require_bounded_region(metadata::ASTERGEDMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("ASTERGEDv3() must be used with a bounded region. Derive it from the " *
              "model grid and the dataset's padding:\n" *
              "    region = BoundingBox(grid; padding = default_horizontal_padding(ASTERGEDv3()))\n" *
              "    metadatum = Metadatum(:emissivity; dataset = ASTERGEDv3(), region)\n" *
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
DataWrangling.dataset_variable_name(metadata::ASTERGEDMetadatum) =
    ASTERGED_variable_names[metadata.name]

# Retrieval gaps and water decode to NaN; inpaint from the nearest valid cell.
# `Inf` iterations so no gap is left as the zero a capped inpainting would write.
DataWrangling.default_inpainting(::ASTERGEDMetadatum) = NearestNeighborInpainting(Inf)

# The regional NetCDF is variable-independent, so key the inpainted cache on the
# variable name too (otherwise emissivity and uncertainty would collide).
DataWrangling.inpainted_metadata_path(metadata::ASTERGEDMetadatum) =
    joinpath(metadata.dir,
             string("inpainted_", metadata.name, "_", replace(metadata.filename, ".nc" => ".jld2")))

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
`BoundingBox` (encoded `W,S,E,N`). CMR search is anonymous; only the tile download
itself needs Earthdata credentials.
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
##### Data retrieval — the regional NetCDF already holds the decoded broadband
##### floats (with NaN over gaps/water), so this is a dumb reader; the decode /
##### broadband / assembly happens at download time (see the ArchGDAL extension).
#####

"""
    retrieve_data(metadata::ASTERGEDMetadatum)

Read the regional broadband `Float32` field for `metadata.name` from the NetCDF
written by the download step (see `asterged_tiles_to_netcdf`). Gaps and water are
`NaN` for the downstream inpainting. Returns a regional `(Nx, Ny)` array.
"""
function DataWrangling.retrieve_data(metadata::ASTERGEDMetadatum)
    haskey(ASTERGED_variable_names, metadata.name) ||
        error("ASTERGEDv3 does not provide variable :$(metadata.name); " *
              "available variables: $(collect(keys(ASTERGED_variable_names)))")
    ds = DataWrangling.Dataset(metadata_path(metadata))
    data = ds[DataWrangling.dataset_variable_name(metadata)][:, :]
    close(ds)
    return data
end

#####
##### Download — the real fetch (CMR discovery, Earthdata GET, HDF5 read, decode,
##### broadband collapse, regional-NetCDF write) lives in the ArchGDAL extension.
##### The entry points below fall back to a clear error when it is not loaded.
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
