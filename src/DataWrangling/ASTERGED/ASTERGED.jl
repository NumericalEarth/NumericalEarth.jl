module ASTERGED

export ASTERGEDv3

using Downloads: Downloads
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       BoundingBox, metadata_path, dataset_variable_name

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

# Land/Water map coding. The ASTER GED `/Land_Water_Map/LWmap` layer uses a
# coding that differs between documentation sources (LP DAAC vs GEE: land/water
# as 1/2 vs 0/1). The LP DAAC User Guide V3 codes water as 2 (land as 1); we
# take that as the default and expose it as a keyword so it can be corrected
# once verified empirically on a real tile.
const ASTERGED_WATER_CODE = 2

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

Reading the HDF5 tiles requires `HDF5.jl` and NASA Earthdata credentials; that
path is gated behind a fallback error (see the module status report). The pure
decode/broadband/water-mask core (`decode_mean`, `decode_sdev`,
`broadband_emissivity`) needs neither and is unit-tested directly.

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
`NaN`. See [`ASTERGED_WATER_CODE`](@ref) for the water-coding caveat.
"""
mask_water(field, land_water_map; water_code = ASTERGED_WATER_CODE) =
    ifelse.(land_water_map .== water_code, NaN, field)

#####
##### Dataset interface (per Part D.2)
#####

DataWrangling.available_variables(::ASTERGEDv3) = ASTERGED_dataset_variable_names
DataWrangling.default_download_directory(::ASTERGEDv3) = download_ASTERGED_cache

# Native raster is stored N→S from each tile's NW corner; flip to S→N.
DataWrangling.reversed_latitude_axis(::ASTERGEDv3) = true

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

Oceananigans.Fields.location(::ASTERGEDMetadatum) = (Center, Center, Center)

#####
##### Data retrieval (pure processing over a gated HDF5 read)
#####

"""
    retrieve_data(metadata::ASTERGEDMetadatum)

Read the raw ASTER GED tiles for `metadata.region`, decode the digital numbers,
collapse the five narrowband emissivities to a single broadband value (or
propagate the per-band uncertainty for `:emissivity_uncertainty`), and mask out
water. Returns a regional `(Nx, Ny)` array. The HDF5 read is gated (see
[`read_asterged_region`](@ref)); the decode/broadband/mask steps are the pure,
unit-tested core.
"""
function DataWrangling.retrieve_data(metadata::ASTERGEDMetadatum)
    raw = read_asterged_region(metadata)
    coefficients = metadata.dataset.broadband_coefficients

    if metadata.name === :emissivity
        emissivity = broadband_emissivity_map(decode_mean.(raw.mean_dn), coefficients)
        return mask_water(emissivity, raw.land_water_map)
    elseif metadata.name === :emissivity_uncertainty
        uncertainty = broadband_uncertainty_map(decode_sdev.(raw.sdev_dn), coefficients)
        return mask_water(uncertainty, raw.land_water_map)
    else
        error("ASTERGEDv3 does not provide variable :$(metadata.name); " *
              "available variables: $(collect(keys(ASTERGED_dataset_variable_names)))")
    end
end

"""
    read_asterged_region(metadata::ASTERGEDMetadatum)

Read and mosaic the raw ASTER GED HDF5 tiles intersecting `metadata.region`,
returning a NamedTuple of raw digital-number arrays
`(; mean_dn, sdev_dn, land_water_map)` (`mean_dn`/`sdev_dn` shaped `(5, Nx, Ny)`,
band index first; `land_water_map` shaped `(Nx, Ny)`).

This is the credentials- and format-gated path. ASTER GED tiles are HDF5, which
`HDF5.jl` reads directly, but `HDF5.jl` is intentionally not a dependency of
NumericalEarth (see AGENTS.md rule 10), and the download needs NASA Earthdata
credentials. This fallback therefore errors clearly; wire an actual reader (and
CMR/Earthdata download) before using `Field(::ASTERGEDMetadatum, grid)`.
"""
read_asterged_region(metadata::ASTERGEDMetadatum) =
    error("Reading ASTER GED requires HDF5.jl and NASA Earthdata credentials, " *
          "which are not wired in this build. HDF5.jl cannot be added to the root " *
          "Project.toml (AGENTS.md rule 10); implement the HDF5 read + Earthdata/CMR " *
          "download separately. See future_plans/status/aster-ged_STATUS.md. " *
          "The pure decode/broadband/mask core (decode_mean, decode_sdev, " *
          "broadband_emissivity) works without this path.")

function Downloads.download(metadata::ASTERGEDMetadatum)
    path = metadata_path(metadata)
    @root if !isfile(path)
        download_asterged(metadata, path)
    end
    return path
end

# Gated: the real download resolves tiles via CMR and fetches them with Earthdata
# credentials (.netrc / bearer token). Fallback errors when not wired.
download_asterged(metadata, path) =
    error("Downloading ASTER GED requires NASA Earthdata credentials and a CMR/HTTPS " *
          "download path that is not wired in this build. " *
          "See future_plans/status/aster-ged_STATUS.md.")

end # module ASTERGED
