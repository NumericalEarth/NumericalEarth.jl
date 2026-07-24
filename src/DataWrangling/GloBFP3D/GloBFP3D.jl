module GloBFP3D

export BuildingFootprints3D, building_morphometry

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
    BuildingFootprints3D(; resolution = 3)

3D-GloBFP building footprints (Che et al. 2024): ~1.3 billion individual building footprint
polygons (LoD1), each carrying an estimated `Height` (m), distributed globally as per-tile
shapefiles in EPSG:4326.

The adapter **rasterizes** the footprint heights onto a fine grid of `resolution` meters
(default 3 m), producing a single decoded field. `:building_height` (the height of the
building covering each cell, `0` where unbuilt). This fine building-height raster is the
accurate common source for per-cell morphometry: [`building_morphometry`](@ref) reduces it
onto any (coarser) target grid, computing the mean/maximum height, height standard deviation,
built-up fraction, frontal-area index, and gross building lift, each with the estimator
appropriate to it.

Because it is a global vector product, it is read in regional windows only: construct the
`Metadatum` with a longitude/latitude `BoundingBox`. The windowed tile download + rasterization
is performed by `ext/NumericalEarthArchGDALExt.jl` and requires `using ArchGDAL`.

The heights are ML-estimated (XGBoost, RMSE 1.9–14.6 m) and biased low. The `Height`
attributes are licensed CC BY 4.0; the footprint geometry derives from Microsoft Building
Footprints (ODbL), OpenStreetMap (ODbL) and Google–Microsoft Open Buildings (CC BY 4.0),
whose share-alike/attribution terms may propagate to a derived product.

Reference: Che, Y. et al. (2024), *3D-GloBFP: the first global three-dimensional building
footprint dataset*, Earth Syst. Sci. Data 16:5357–5374, doi:10.5194/essd-16-5357-2024
(Zenodo 10.5281/zenodo.11319913).

```jldoctest
julia> using NumericalEarth.DataWrangling.GloBFP3D

julia> BuildingFootprints3D()
BuildingFootprints3D(resolution = 3 m)
```
"""
struct BuildingFootprints3D <: AbstractStaticDataset
    resolution :: Int
end

function BuildingFootprints3D(; resolution = 3)
    resolution > 0 ||
        throw(ArgumentError("BuildingFootprints3D resolution must be a positive number of meters, got $resolution."))
    return BuildingFootprints3D(resolution)
end

Base.summary(dataset::BuildingFootprints3D) =
    string("BuildingFootprints3D(resolution = ", dataset.resolution, " m)")
Base.show(io::IO, dataset::BuildingFootprints3D) = print(io, summary(dataset))

const BuildingFootprints3DMetadatum = Metadatum{<:BuildingFootprints3D}

#####
##### Variables — the adapter decodes a single fine building-height raster
#####

GloBFP3D_variable_names = Dict(:building_height => "building_height")

DataWrangling.available_variables(::BuildingFootprints3D) = GloBFP3D_variable_names
DataWrangling.dataset_variable_name(data::BuildingFootprints3DMetadatum) = GloBFP3D_variable_names[data.name]

#####
##### Dataset interface
#####

DataWrangling.default_download_directory(::BuildingFootprints3D) = download_GloBFP3D_cache

# The rasterized regional raster is on a plain lat/lon grid; the native hull is the global
# lat/lon extent and the shared regrid restricts it to the BoundingBox.
DataWrangling.longitude_interfaces(::BuildingFootprints3D) = (-180, 180)
DataWrangling.latitude_interfaces(::BuildingFootprints3D)  = (-90, 90)

native_resolution(dataset::BuildingFootprints3D) = dataset.resolution

# Degree step of a `resolution`-meter arc on the default planet, using the same metric a
# LatitudeLongitudeGrid does (`meters = radius · deg2rad(degrees)`). The raster uses this step in
# BOTH longitude and latitude — a uniform lat/lon grid — so it is a sub-window of the global
# lattice the shared `Field(::Metadatum)` read path assumes (from `longitude_interfaces` + `size`);
# a latitude-dependent Δλ would misalign that integer-offset read. Cells are thus ~`resolution` m
# N–S and ~`resolution`·cos φ m E–W. Ingest has no target grid, so we use the global default
# radius (a custom-radius grid shifts this by < 0.1%).
native_cell_size(dataset::BuildingFootprints3D) =
    rad2deg(native_resolution(dataset) / Oceananigans.defaults.planet_radius)

# Nominal global native size in EPSG:4326 (only the windowed portion is materialized).
function Base.size(dataset::BuildingFootprints3D, variable)
    Δ = native_cell_size(dataset)
    Nx = round(Int, 360 / Δ)
    Ny = round(Int, 180 / Δ)
    return (Nx, Ny, 1)
end

DataWrangling.metadata_filename(dataset::BuildingFootprints3D, name, date, region) =
    string("BuildingFootprints3D_", dataset.resolution, "m_", region_suffix(region), ".nc")

region_suffix(::Nothing) = "global"

function region_suffix(region::BoundingBox)
    λ = region.longitude
    φ = region.latitude
    return string("lon_", bound_str(λ), "_lat_", bound_str(φ))
end

bound_str(::Nothing) = "nothing"
bound_str(bounds) = string(bounds[1], "_", bounds[2])

function DataWrangling.validate_dataset_coverage(grid, metadata::BuildingFootprints3DMetadatum)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("$(summary(metadata.dataset)) must be used with a bounded region. " *
              "Build the metadatum with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadatum = Metadatum(:building_height; dataset = BuildingFootprints3D(),\n" *
              "                          region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))\n" *
              "    Field(metadatum, grid)")
    end
    return nothing
end

#####
##### Metadatum interface
#####

DataWrangling.is_three_dimensional(::BuildingFootprints3DMetadatum) = false

# The regional NetCDF we materialize stores coordinates as "lon"/"lat".
DataWrangling.longitude_name(::BuildingFootprints3DMetadatum) = "lon"
DataWrangling.latitude_name(::BuildingFootprints3DMetadatum)  = "lat"

# NEVER inpaint: a building height of 0 over non-built land is physical, not a gap.
DataWrangling.default_inpainting(::BuildingFootprints3DMetadatum) = nothing

Oceananigans.Fields.location(::BuildingFootprints3DMetadatum) = (Center, Center, Center)

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
    west, east   = extrema(region.longitude)
    south, north = extrema(region.latitude)
    i₀ = floor(Int, (west  + 180) / Δλ) - pad
    j₀ = floor(Int, (south +  90) / Δφ) - pad
    i₁ = ceil(Int,  (east  + 180) / Δλ) + pad
    j₁ = ceil(Int,  (north +  90) / Δφ) + pad
    return (; west = -180 + i₀ * Δλ, south = -90 + j₀ * Δφ, Δλ, Δφ, Nx = i₁ - i₀, Ny = j₁ - j₀)
end

#####
##### Per-cell morphometry from the fine building-height raster
#####
#####
##### Each variable is reduced from the fine cells falling in a target cell with the
##### estimator appropriate to it: a mean over built cells for the height statistics, the
##### coverage fraction for λp, the whole-cell mean for the gross lift, a running max for
##### Hmax, and the windward wall area (from height steps) for the frontal-area index λf.
#####

# Target cell each fine coordinate falls in, from the target cell `faces`. Resolved once per fine
# row/column (not per cell), so the hot loop stays O(1) per fine cell for regular and stretched
# grids alike. `searchsortedlast` returns 0 or `N+1` outside the target hull (filtered by the
# bounds check in the loop).
target_index_map(faces, coordinates) = Int[searchsortedlast(faces, c) for c in coordinates]

"""
    reduce_morphometry(height, longitudes, latitudes, target_grid)

Reduce a fine building-height raster — `height` (m, `0` where unbuilt) on the regular grid of
cell-center `longitudes`/`latitudes` (degrees) — onto `target_grid` (a coarser
`LatitudeLongitudeGrid`, regular or latitude/longitude-stretched). Returns a NamedTuple of
`(Nx, Ny)` arrays:

- `built_up_fraction` `λp` — fraction of fine cells that are built.
- `mean_building_height` `H` — mean height over the **built** fine cells (area-weighted).
- `building_height_std` `σH` — standard deviation of height over the built fine cells.
- `maximum_building_height` `Hmax` — maximum height (max-pooled).
- `gross_building_height` — mean height over **all** fine cells (`= λp·H`), the DSM lift.
- `frontal_area_index` `λf` — windward wall area from height steps, direction-averaged:
  `(Σₓ|ΔH|·dy + Σᵧ|ΔH|·dx) / (4·A)`.

Empty target cells are `0`. The fine raster is geographic (EPSG:4326), so `target_grid` must be a
`LatitudeLongitudeGrid` (coarser than the raster); each fine cell is placed by the target cell
faces, so a latitude/longitude-stretched grid works too, and the latitude-varying cell size is
carried through the metric `dx`/`dy` in `λf`.
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

    # Metric fine-cell sizes on the target grid's sphere, as Oceananigans measures them
    # (Δx = radius·cosd(φ)·deg2rad(Δλ), Δy = radius·deg2rad(Δφ)).
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
    building_morphometry(target_grid; dataset = BuildingFootprints3D(), region)

Per-cell building morphometry on `target_grid` (a `LatitudeLongitudeGrid`, coarser than the
`dataset` rasterization resolution), aggregated from the fine 3D-GloBFP building-height raster
over `region`. Returns a NamedTuple of `Field`s (`mean_building_height`, `building_height_std`,
`maximum_building_height`, `built_up_fraction`, `frontal_area_index`, `gross_building_height`)
via [`reduce_morphometry`](@ref); see there for the per-variable estimators. Downloading and
rasterizing the footprints requires `using ArchGDAL`.
"""
function building_morphometry(target_grid::LatitudeLongitudeGrid; dataset = BuildingFootprints3D(), region)
    metadatum = Metadatum(:building_height; dataset, region)
    Downloads.download(metadatum)
    height = DataWrangling.retrieve_data(metadatum)
    longitudes, latitudes = DataWrangling.read_file_coords(metadatum)

    reduced = reduce_morphometry(height, longitudes, latitudes, target_grid)

    # The reduction runs on the host; move each result onto the target architecture (CPU or GPU).
    arch = architecture(target_grid)
    return map(reduced) do array
        field = Field{Center, Center, Nothing}(target_grid)
        interior(field) .= on_architecture(arch, reshape(array, size(array, 1), size(array, 2), 1))
        field
    end
end

#####
##### Tile discovery — the 3D-GloBFP shapefiles are hosted on figshare in ten parts, one
##### `.zip` per grid tile named `gridID_lon1_lat1_lon2_lat2_region.zip` (SW/NE corners).
##### The tile bbox is parsed from the filename; the figshare file index + download live in
##### the ArchGDAL extension.
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

"""
    tile_intersects(bounds, region::BoundingBox)

Whether a tile's `bounds` (from [`parse_tile_bounds`](@ref)) overlaps `region`.
"""
function tile_intersects(bounds, region::BoundingBox)
    λ₁, λ₂ = extrema(region.longitude)
    φ₁, φ₂ = extrema(region.latitude)
    return !(bounds.east < λ₁ || bounds.west > λ₂ || bounds.north < φ₁ || bounds.south > φ₂)
end

#####
##### Download (regional footprint tiles → rasterized NetCDF via the ArchGDAL ext)
#####

function Downloads.download(metadatum::BuildingFootprints3DMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)
    nc_path = metadata_path(metadatum)
    @root if !isfile(nc_path)
        globfp3d_rasterize_to_netcdf(metadatum, nc_path)
    end
    return nc_path
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded. This fallback
# fires only when the extension is not active.
globfp3d_rasterize_to_netcdf(metadatum, nc_path) =
    error("Reading the 3D-GloBFP footprint shapefiles requires the ArchGDAL package " *
          "(for the OGR vector read + rasterization). Load it with `using ArchGDAL`.")

end # module GloBFP3D
