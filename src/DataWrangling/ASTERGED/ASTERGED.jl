module ASTERGED

export ASTERGEDv3

using Downloads: Downloads
using Oceananigans: Center, CPU
using Oceananigans.DistributedComputations: @root
using Oceananigans.Fields: Field, interior

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       BoundingBox, metadata_path, native_grid, inpaint_mask!,
                       NearestNeighborInpainting, bounding_box_suffix

import Oceananigans

download_ASTERGED_cache::String = ""
function __init__()
    global download_ASTERGED_cache = DataWrangling.download_cache("ASTERGED")
end

#####
##### Broadband-emissivity synthesis coefficients
#####

# Narrowband → broadband coefficients for ASTER TIR bands 10–14 over the 8.0–13.5 µm
# window, from Ogawa & Schmugge (2004). The weights sum to unity, so the broadband
# value is a convex combination of the band emissivities.
const OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS = [0.088, 0.053, 0.174, 0.380, 0.305]

#####
##### Dataset type
#####

"""
    ASTERGEDv3 <: AbstractStaticDataset

Global Emissivity Dataset (GED) v3 from the Advanced Spaceborne Thermal Emission and
Reflection Radiometer (ASTER) aboard NASA's Terra satellite: a static 2000–2008
clear-sky mean of land-surface emissivity on a WGS84 lat/lon grid, distributed as HDF5
in 1°×1° tiles. Two resolutions are supported:

- `:low_1km` — 1 km (36 arcsec), the default.
- `:high_100m` — 100 m (3.6 arcsec), for sub-km domains.

ASTER GED provides five narrowband emissivities (TIR bands 10–14). A longwave scheme
needs a single broadband value, so the download collapses the five bands using the
[Ogawa & Schmugge (2004)](@cite ogawa2004mapping) coefficients. The resulting `Field`
is finite everywhere and can be passed to `SurfaceRadiationProperties` as its
`emissivity`.

The product also retrieves an emissivity over water (open-ocean broadband ε ≈ 0.98).
Genuine gaps appear only where a clear-sky retrieval was never obtained — persistent
tropical cloud, summertime snow screened as cloud — and are inpainted when the `Field`
is built, land from land and water from water, so a coastal gap never inherits the
ocean value.

ASTER GED is read in regional windows only, so build the `Metadatum` with a
longitude/latitude `BoundingBox`, most simply the one the model grid implies,

    region = default_region(ASTERGEDv3(), grid)

Reading the HDF5 tiles requires `ArchGDAL` (with the HDF5 driver) and NASA Earthdata
credentials (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`).

!!! note "Limitations"
    The broadband coefficients are a Sahara-desert regression applied globally, an
    extrapolation over humid and vegetated surfaces. As a static clear-sky mean the
    product also misses soil moisture, vegetation phenology, and snow (ε is 0.03–0.08
    higher under snow), and its clear-sky sampling biases it toward dry states.

Reference: Hulley et al. (2015), https://doi.org/10.1002/2015GL065564
Data source: https://www.earthdata.nasa.gov/data/catalog/lpcloud-ag100-003
"""
struct ASTERGEDv3 <: AbstractStaticDataset
    resolution :: Symbol
end

"""
    ASTERGEDv3(resolution = :low_1km)

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

const ASTERGED_variables = (:emissivity, :emissivity_uncertainty)

#####
##### Decode and broadband synthesis
#####

"""
    asterged_decode_emissivity(DN)

Decode a raw `/Emissivity/Mean` digital number to a `Float32` emissivity: fill value
−9999 maps to `NaN`, otherwise `ε = 0.001 · DN` clamped to `[0, 1]`. The clamp is
needed because ASTER's Temperature/Emissivity Separation retrieval carries no `ε ≤ 1`
constraint, and the product stores band emissivities above unity.
"""
@inline asterged_decode_emissivity(DN) = ifelse(DN == -9999, NaN32, clamp(0.001f0 * DN, 0, 1))

"""
    asterged_decode_uncertainty(DN)

Decode a raw `/Emissivity/SDev` digital number to a `Float32` emissivity standard
deviation: fill value −9999 maps to `NaN`, otherwise `σ = 1e-4 · DN`. Note the scale
is 10× smaller than `asterged_decode_emissivity`'s.
"""
@inline asterged_decode_uncertainty(DN) = ifelse(DN == -9999, NaN32, 1f-4 * DN)

"""
    broadband_emissivity(ε_vector, coefficients)

Collapse the five decoded narrowband emissivities (ASTER bands 10–14) to one broadband
emissivity, the dot product with `coefficients`. `NaN` in any band propagates.
"""
broadband_emissivity(ε_vector, coefficients) =
    sum(coefficients[b] * ε_vector[b] for b in eachindex(coefficients))

"""
    broadband_uncertainty(σ_vector, coefficients)

Propagate the five per-band standard deviations to a broadband uncertainty as the
fully-correlated upper bound `σ = Σ cᵢ σᵢ`. ASTER TES band emissivities share one
temperature retrieval and atmospheric correction, so their errors are strongly
correlated.
"""
broadband_uncertainty(σ_vector, coefficients) =
    sum(coefficients[b] * σ_vector[b] for b in eachindex(coefficients))

"""
    broadband_map(decoded_bands, coefficients)

Array form of `broadband_emissivity` and `broadband_uncertainty`: collapse a decoded
`(5, Nx, Ny)` array (band index first, as in the HDF5 `/Emissivity/*` layout) to a
broadband `(Nx, Ny)` array.
"""
function broadband_map(decoded_bands, coefficients)
    FT = eltype(decoded_bands)
    weights = reshape(FT.(coefficients), :, 1, 1)
    return dropdims(sum(weights .* decoded_bands; dims = 1); dims = 1)
end

# Fall back to the tile's spacing on a single-cell axis, whose span is zero.
function axis_spacing(axis, tile_axis)
    length(axis)      > 1 && return (axis[end] - axis[1]) / (length(axis) - 1)
    length(tile_axis) > 1 && return abs(tile_axis[2] - tile_axis[1])
    return one(eltype(axis))
end

"""
    place_tile!(field, tile_values, tile_longitude, tile_latitude, longitude, latitude)

Block-copy a decoded tile onto the regional grid `(longitude, latitude)` by mapping
each tile-cell center to its native index. Indexing is by value, so the tile may store
latitude in either direction; longitudes are folded into `[-180, 180]`, so a tile lands
on a grid labeled in any convention (this assumes the region spans less than 180° of
longitude). `NaN` tile cells are skipped, leaving a valid value from an adjacent tile
at a shared boundary in place, and cells outside the tile's footprint are untouched.
"""
function place_tile!(field, tile_values, tile_longitude, tile_latitude, longitude, latitude)
    Nx, Ny = size(field)
    Δλ = axis_spacing(longitude, tile_longitude)
    Δφ = axis_spacing(latitude,  tile_latitude)
    for (jl, φ) in enumerate(tile_latitude)
        jr = round(Int, (φ - latitude[1]) / Δφ) + 1
        (1 ≤ jr ≤ Ny) || continue
        for (il, λ) in enumerate(tile_longitude)
            ic = round(Int, rem(λ - longitude[1], 360, RoundNearest) / Δλ) + 1
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

DataWrangling.available_variables(::ASTERGEDv3) = ASTERGED_variables
DataWrangling.default_download_directory(::ASTERGEDv3) = download_ASTERGED_cache

# A couple of native 1 km cells of margin for interpolation stencils at the edge.
DataWrangling.default_horizontal_padding(::ASTERGEDv3) = 0.02

# Tiled 1 km/100 m coverage is read in windows, so window it to the grid.
DataWrangling.default_region(dataset::ASTERGEDv3, grid) =
    DataWrangling.dataset_bounding_box(dataset, grid)

DataWrangling.reversed_latitude_axis(::ASTERGEDv3) = false

DataWrangling.longitude_interfaces(::ASTERGEDv3) = (-180, 180)
DataWrangling.latitude_interfaces(::ASTERGEDv3) = (-90, 90)

# Global pixel counts set the native resolution Δ; the download returns the regional window.
global_pixels(::Val{:high_100m}) = (360_000, 180_000, 1)
global_pixels(::Val{:low_1km})   = (36_000, 18_000, 1)

Base.size(dataset::ASTERGEDv3) = global_pixels(Val(dataset.resolution))
Base.size(dataset::ASTERGEDv3, variable) = size(dataset)

# Region-keyed but variable-independent: one regional NetCDF holds both the emissivity
# and its uncertainty, since the tile download produces both at once.
DataWrangling.metadata_filename(dataset::ASTERGEDv3, name, date, region) =
    string("ASTERGED_", dataset.resolution, "_", bounding_box_suffix(region), ".nc")

function require_bounded_region(metadata::ASTERGEDMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("ASTERGEDv3() must be used with a bounded region. Derive it from the " *
              "model grid:\n" *
              "    region = default_region(ASTERGEDv3(), grid)\n" *
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
DataWrangling.dataset_variable_name(metadata::ASTERGEDMetadatum) = string(metadata.name)

# `Inf` iterations, so no gap is left as the zero a capped inpainting would write.
DataWrangling.default_inpainting(::ASTERGEDMetadatum) = NearestNeighborInpainting(Inf)

# The regional NetCDF is variable-independent, so key the inpainted cache on the
# variable name too; otherwise emissivity and uncertainty would collide.
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
##### Product identity
#####

# NASA CMR short name and version at each resolution ("AG" abbreviates ASTER GED).
asterged_short_name(dataset::ASTERGEDv3) = asterged_short_name(Val(dataset.resolution))
asterged_short_name(::Val{:high_100m}) = "AG100"
asterged_short_name(::Val{:low_1km})   = "AG1KM"
asterged_version(::ASTERGEDv3) = "003"

#####
##### Data retrieval
#####

"""
    retrieve_data(metadata::ASTERGEDMetadatum)

Read the regional `(Nx, Ny)` broadband field for `metadata.name` and fill its land gaps
from land. Water gaps are left `NaN` for the downstream inpainting. The decode and
broadband collapse happen at download time, in the ArchGDAL extension.
"""
function DataWrangling.retrieve_data(metadata::ASTERGEDMetadatum)
    metadata.name ∈ ASTERGED_variables ||
        error("ASTERGEDv3 does not provide variable :$(metadata.name); " *
              "available variables: $(ASTERGED_variables)")

    ds = DataWrangling.Dataset(metadata_path(metadata))
    data = ds[DataWrangling.dataset_variable_name(metadata)][:, :]
    # Only 1 is water, so the map's own fill and cells outside every tile count as land.
    water = ds["land_water_map"][:, :] .== 1
    close(ds)

    return fill_land_gaps(data, water, metadata)
end

"""
    fill_land_gaps(data, water, metadata)

Fill the clear-sky retrieval gaps over land from land alone. Water is blanked to `NaN`
first so it cannot donate: a coastal land gap would otherwise inherit the open-ocean
ε ≈ 0.98 against 0.94–0.96 over land. Water keeps its own retrievals, and its gaps stay
`NaN` for the ungated inpainting that `Field` runs next.
"""
function fill_land_gaps(data, water, metadata)
    land = .!water
    any(land .& .!isnan.(data)) || return data   # no land donor

    blanked = copy(data)
    blanked[water] .= NaN32

    grid  = native_grid(metadata, CPU())
    field = Field{Center, Center, Nothing}(grid)
    mask  = Field{Center, Center, Nothing}(grid, Bool)
    interior(field, :, :, 1) .= blanked
    interior(mask, :, :, 1) .= isnan.(blanked)

    inpaint_mask!(field, mask)

    filled = Array(interior(field, :, :, 1))
    data[land] .= filled[land]

    return data
end

#####
##### Download
#####

function Downloads.download(metadata::ASTERGEDMetadatum)
    require_bounded_region(metadata)
    path = metadata_path(metadata)
    @root if !isfile(path)
        asterged_tiles_to_netcdf(metadata, path)
    end
    return path
end

# Implemented in the ArchGDAL extension.
asterged_tiles_to_netcdf(metadata, path) =
    error("Reading ASTER GED HDF5 tiles requires ArchGDAL (built with the HDF5 driver) " *
          "and NASA Earthdata credentials. Load ArchGDAL with `using ArchGDAL`, and provide " *
          "credentials via EARTHDATA_USERNAME / EARTHDATA_PASSWORD (register free at " *
          "https://urs.earthdata.nasa.gov).")

end # module ASTERGED
