module GloBFP3D

export GlobalBuildingFootprints3D, building_morphometry

using Downloads: Downloads
using Oceananigans: Center, Face
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: LatitudeLongitudeGrid, λnodes, φnodes
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum,
                       metadata_path, BoundingBox

import Oceananigans

download_GloBFP3D_cache::String = ""
function __init__()
    global download_GloBFP3D_cache = DataWrangling.download_cache("GloBFP3D")
end

#####
##### Dataset type
#####

"""
    GlobalBuildingFootprints3D(; resolution = 3)

3D-GloBFP building footprints (Che et al. 2024): ~1.3 billion building polygons (LoD1), each
carrying an estimated height, distributed globally as per-tile shapefiles in EPSG:4326.

The adapter rasterizes the footprint heights onto a fine grid of `resolution` meters, giving a
single variable `:building_height` (m, `0` where unbuilt). [`building_morphometry`](@ref) reduces
that raster onto a coarser target grid.

Because it is a global vector product, it is read in regional windows only: construct the
`Metadatum` with a longitude/latitude `BoundingBox`. The tile download and rasterization require
`using ArchGDAL`.

Heights are ML-estimated (RMSE 1.9–14.6 m) and biased low. They are licensed CC BY 4.0; the
footprint geometry derives from Microsoft Building Footprints and OpenStreetMap (ODbL) and
Google–Microsoft Open Buildings (CC BY 4.0).

Reference: Che, Y. et al. (2024), *3D-GloBFP: the first global three-dimensional building
footprint dataset*, Earth Syst. Sci. Data 16:5357–5374, doi:10.5194/essd-16-5357-2024
(Zenodo 10.5281/zenodo.11319913).

```jldoctest
julia> using NumericalEarth

julia> GlobalBuildingFootprints3D()
GlobalBuildingFootprints3D(resolution = 3 m)
```
"""
struct GlobalBuildingFootprints3D <: AbstractStaticDataset
    resolution :: Int
end

function GlobalBuildingFootprints3D(; resolution = 3)
    resolution > 0 ||
        throw(ArgumentError("GlobalBuildingFootprints3D resolution must be a positive number of meters, got $resolution."))
    return GlobalBuildingFootprints3D(resolution)
end

Base.summary(dataset::GlobalBuildingFootprints3D) =
    string("GlobalBuildingFootprints3D(resolution = ", dataset.resolution, " m)")
Base.show(io::IO, dataset::GlobalBuildingFootprints3D) = print(io, summary(dataset))

const GlobalBuildingFootprints3DMetadatum = Metadatum{<:GlobalBuildingFootprints3D}

#####
##### Variables
#####

GloBFP3D_variable_names = Dict(:building_height => "building_height")

DataWrangling.available_variables(::GlobalBuildingFootprints3D) = GloBFP3D_variable_names
DataWrangling.dataset_variable_name(data::GlobalBuildingFootprints3DMetadatum) = GloBFP3D_variable_names[data.name]

#####
##### Dataset interface
#####

DataWrangling.default_download_directory(::GlobalBuildingFootprints3D) = download_GloBFP3D_cache

# The native hull is global; the shared regrid restricts the read to the metadatum's BoundingBox.
DataWrangling.longitude_interfaces(::GlobalBuildingFootprints3D) = (-180, 180)
DataWrangling.latitude_interfaces(::GlobalBuildingFootprints3D)  = (-90, 90)

native_resolution(dataset::GlobalBuildingFootprints3D) = dataset.resolution

# Degree step of a `resolution`-meter arc, used in BOTH longitude and latitude so the raster stays
# a sub-window of the global lattice the shared `Field(::Metadatum)` read path assumes; a
# latitude-dependent Δλ would misalign that read. Cells are ~`resolution` m N–S, less E–W.
native_cell_size(dataset::GlobalBuildingFootprints3D) =
    rad2deg(native_resolution(dataset) / Oceananigans.defaults.planet_radius)

# Nominal global native size in EPSG:4326 (only the windowed portion is materialized).
function Base.size(dataset::GlobalBuildingFootprints3D, variable)
    Δ = native_cell_size(dataset)
    Nx = round(Int, 360 / Δ)
    Ny = round(Int, 180 / Δ)
    return (Nx, Ny, 1)
end

DataWrangling.metadata_filename(dataset::GlobalBuildingFootprints3D, name, date, region) =
    string("GlobalBuildingFootprints3D_", dataset.resolution, "m_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::GlobalBuildingFootprints3DMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("$(summary(metadata.dataset)) must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:building_height; dataset = GlobalBuildingFootprints3D(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    west, east   = region.longitude
    south, north = region.latitude
    if !(west < east) || !(south < north)
        error("$(summary(metadata.dataset)) needs a region with strictly increasing bounds " *
              "(longitude[1] < longitude[2] and latitude[1] < latitude[2]); got $(summary(region)). " *
              "A region crossing the antimeridian (west > east, e.g. (179, -179)) is not supported — " *
              "split it into two regions on either side of 180°.")
    end
    return nothing
end

#####
##### Metadatum interface
#####

DataWrangling.is_three_dimensional(::GlobalBuildingFootprints3DMetadatum) = false

# The regional NetCDF we materialize stores coordinates as "lon"/"lat".
DataWrangling.longitude_name(::GlobalBuildingFootprints3DMetadatum) = "lon"
DataWrangling.latitude_name(::GlobalBuildingFootprints3DMetadatum)  = "lat"

# A building height of 0 over unbuilt land is physical, not a gap: never inpaint.
DataWrangling.default_inpainting(::GlobalBuildingFootprints3DMetadatum) = nothing

Oceananigans.Fields.location(::GlobalBuildingFootprints3DMetadatum) = (Center, Center, Center)

#####
##### Native aggregation grid (used by the rasterizing extension)
#####

"""
    native_region_grid(region::BoundingBox, Δλ, Δφ; pad = 2)

Regular lat/lon raster of longitude/latitude cell steps `Δλ`/`Δφ` (degrees) covering `region`,
snapped to the global lattice anchored at `(-180, -90)` and padded by `pad` cells on each side.
Returns `(; west, south, Δλ, Δφ, Nx, Ny)`.
"""
function native_region_grid(region::BoundingBox, Δλ, Δφ; pad = 2)
    west, east   = region.longitude
    south, north = region.latitude
    i₀ = floor(Int, (west  + 180) / Δλ) - pad
    j₀ = floor(Int, (south +  90) / Δφ) - pad
    i₁ = ceil(Int,  (east  + 180) / Δλ) + pad
    j₁ = ceil(Int,  (north +  90) / Δφ) + pad
    return (; west = -180 + i₀ * Δλ, south = -90 + j₀ * Δφ, Δλ, Δφ, Nx = i₁ - i₀, Ny = j₁ - j₀)
end

#####
##### Per-cell morphometry from the fine building-height raster
#####

# Target cell each fine coordinate falls in, resolved once per fine row/column rather than per
# cell. `searchsortedlast` returns 0 or `N+1` outside the target hull, filtered by the bounds
# check in the loop below.
target_index_map(faces, coordinates) = Int[searchsortedlast(faces, c) for c in coordinates]

"""
    reduce_morphometry(height, longitudes, latitudes, target_grid)

Reduce the fine building-height raster `height` (m), on the regular grid of cell-center
`longitudes`/`latitudes` (degrees), onto `target_grid`. Returns the arrays that
[`building_morphometry`](@ref) wraps in `Field`s; empty target cells are `0`.

Each fine cell is placed by the target cell faces, so a latitude/longitude-stretched
`target_grid` works too, with the latitude-varying cell size carried through `dx`/`dy` in `λf`.
"""
function reduce_morphometry(height, longitudes, latitudes, target_grid::LatitudeLongitudeGrid)
    Nx = size(target_grid, 1)
    Ny = size(target_grid, 2)
    λfaces   = λnodes(target_grid, Face())
    φfaces   = φnodes(target_grid, Face())
    φcenters = φnodes(target_grid, Center())

    nx, ny = size(height)
    Δλ_fine = nx > 1 ? longitudes[2] - longitudes[1] : zero(eltype(longitudes))
    Δφ_fine = ny > 1 ? latitudes[2]  - latitudes[1]  : zero(eltype(latitudes))
    Imap = target_index_map(λfaces, longitudes)
    Jmap = target_index_map(φfaces, latitudes)

    count_total = zeros(Int, Nx, Ny)
    count_built = zeros(Int, Nx, Ny)
    Σheight     = zeros(Float64, Nx, Ny)
    Σheight²    = zeros(Float64, Nx, Ny)
    running_max = zeros(Float64, Nx, Ny)
    Σstep_x     = zeros(Float64, Nx, Ny)
    Σstep_y     = zeros(Float64, Nx, Ny)

    @inbounds for j in 1:ny, i in 1:nx
        I = Imap[i]
        J = Jmap[j]
        (1 <= I <= Nx && 1 <= J <= Ny) || continue
        h = height[i, j]
        count_total[I, J] += 1
        if h > 0
            count_built[I, J] += 1
            Σheight[I, J]  += h
            Σheight²[I, J] += h * h
            running_max[I, J] = max(running_max[I, J], h)
        end
        i < nx && (Σstep_x[I, J] += abs(height[i + 1, j] - h))
        j < ny && (Σstep_y[I, J] += abs(height[i, j + 1] - h))
    end

    built_up_fraction       = zeros(Float64, Nx, Ny)
    mean_building_height    = zeros(Float64, Nx, Ny)
    building_height_std     = zeros(Float64, Nx, Ny)
    maximum_building_height = running_max
    gross_building_height   = zeros(Float64, Nx, Ny)
    frontal_area_index      = zeros(Float64, Nx, Ny)

    # Fine-cell sizes on the target grid's sphere, as Oceananigans measures them.
    R = target_grid.radius
    for J in 1:Ny
        φc = φcenters[J]
        dy = R * deg2rad(Δφ_fine)
        dx = R * deg2rad(Δλ_fine) * cosd(φc)
        for I in 1:Nx
            nt = count_total[I, J]
            nb = count_built[I, J]
            if nt > 0
                built_up_fraction[I, J]     = nb / nt
                gross_building_height[I, J] = Σheight[I, J] / nt
                cell_area = nt * dx * dy
                frontal_area_index[I, J] = cell_area > 0 ?
                    (Σstep_x[I, J] * dy + Σstep_y[I, J] * dx) / (4 * cell_area) : 0.0
            end
            if nb > 0
                m = Σheight[I, J] / nb
                mean_building_height[I, J] = m
                building_height_std[I, J]  = sqrt(max(Σheight²[I, J] / nb - m^2, 0.0))
            end
        end
    end

    return (; mean_building_height, building_height_std, maximum_building_height,
              built_up_fraction, frontal_area_index, gross_building_height)
end

"""
    building_morphometry(target_grid; dataset = GlobalBuildingFootprints3D(), region)

Per-cell building morphometry on `target_grid` (a `LatitudeLongitudeGrid`, coarser than the
`dataset` rasterization resolution), aggregated from the fine 3D-GloBFP building-height raster
over `region`. Returns a NamedTuple of `Field`s:

- `built_up_fraction` `λp` — fraction of fine cells that are built.
- `mean_building_height` `H` — mean height over the built fine cells.
- `building_height_std` `σH` — standard deviation of height over the built fine cells.
- `maximum_building_height` `Hmax` — maximum height.
- `gross_building_height` — mean height over all fine cells (`= λp·H`), the digital surface lift.
- `frontal_area_index` `λf` — windward wall area from height steps, direction-averaged:
  `(Σₓ|ΔH|·dy + Σᵧ|ΔH|·dx) / (4·A)`.

Downloading and rasterizing the footprints requires `using ArchGDAL`.
"""
function building_morphometry(target_grid::LatitudeLongitudeGrid; dataset = GlobalBuildingFootprints3D(), region)
    metadatum = Metadatum(:building_height; dataset, region)
    Downloads.download(metadatum)
    height = DataWrangling.retrieve_data(metadatum)
    longitudes, latitudes = DataWrangling.read_file_coords(metadatum)

    reduced = reduce_morphometry(height, longitudes, latitudes, target_grid)

    arch = architecture(target_grid)
    return map(reduced) do array
        field = Field{Center, Center, Nothing}(target_grid)
        interior(field) .= on_architecture(arch, reshape(array, size(array, 1), size(array, 2), 1))
        field
    end
end

#####
##### Tile discovery — 3D-GloBFP ships one `.zip` per grid tile, named
##### `gridID_lon1_lat1_lon2_lat2_region.zip` (SW/NE corners), across ten figshare parts.
#####

# figshare article ids of the ten dataset parts (see the Zenodo record's data_links.txt).
const FIGSHARE_ARTICLE_IDS = (28879733, 28881749, 28882700, 28889813, 28890593,
                              28891631, 28903454, 28903853, 28904453, 28906499)

figshare_article_url(id) = string("https://api.figshare.com/v2/articles/", id)

"""
    parse_tile_bounds(name)

Parse a 3D-GloBFP tile filename `gridID_lon1_lat1_lon2_lat2_…` into
`(; gid, west, south, east, north)`, or `nothing` if it does not match.
"""
function parse_tile_bounds(name)
    m = match(r"^(\d+)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_", name)
    isnothing(m) && return nothing
    W, S, E, N = parse.(Float64, (m[2], m[3], m[4], m[5]))
    return (; gid = parse(Int, m[1]), west = W, south = S, east = E, north = N)
end

function tile_intersects(bounds, region::BoundingBox)
    λ₁, λ₂ = region.longitude
    φ₁, φ₂ = region.latitude
    return !(bounds.east < λ₁ || bounds.west > λ₂ || bounds.north < φ₁ || bounds.south > φ₂)
end

#####
##### Download (regional footprint tiles → rasterized NetCDF via the ArchGDAL ext)
#####

function Downloads.download(metadatum::GlobalBuildingFootprints3DMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        globfp3d_rasterize_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

# Implemented in ext/NumericalEarthArchGDALExt/globfp3d.jl; this fallback fires when it is not active.
globfp3d_rasterize_to_netcdf(metadatum, nc_path) =
    error("Reading the 3D-GloBFP footprint shapefiles requires the ArchGDAL package " *
          "(for the OGR vector read + rasterization). Load it with `using ArchGDAL`.")

end # module GloBFP3D
