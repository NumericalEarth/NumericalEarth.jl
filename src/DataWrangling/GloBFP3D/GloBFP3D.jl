module GloBFP3D

export GlobalBuildingFootprints3D, building_morphometry

using Downloads: Downloads
using Oceananigans: Center, Face
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: LatitudeLongitudeGrid, λnodes, φnodes
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum, metadata_path,
                       BoundingBox, bounding_box_suffix, latitude_summary, native_region_grid

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

3D-GloBFP building footprints: ~1.3 billion building polygons (LoD1), each carrying an
estimated height, distributed globally as per-tile shapefiles in EPSG:4326.

The heights are rasterized onto a fine grid of `resolution` meters, giving a single variable
`:building_height` (m, `0` where unbuilt). [`building_morphometry`](@ref) reduces that raster
onto a coarser target grid.

Being a global vector product, it is read in regional windows only: build the `Metadatum` with
a longitude/latitude `BoundingBox`. The tile download and rasterization require `using ArchGDAL`.

Heights are machine-learning estimates (RMSE 1.9–14.6 m) and biased low. The product is licensed
CC BY 4.0 over footprint geometry from Microsoft Building Footprints, OpenStreetMap (ODbL), and
Google–Microsoft Open Buildings.

Reference: Che et al. (2024), https://doi.org/10.5194/essd-16-5357-2024

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

DataWrangling.longitude_interfaces(::GlobalBuildingFootprints3D) = (-180, 180)
DataWrangling.latitude_interfaces(::GlobalBuildingFootprints3D)  = (-90, 90)

globfp3d_native_resolution(dataset::GlobalBuildingFootprints3D) = dataset.resolution

# The same degree step in longitude and latitude, so the raster is a sub-window of the global
# lattice `Field(::Metadatum)` assumes; a latitude-dependent Δλ would misalign that read. Cells
# are therefore ~`resolution` m N–S and shorter E–W.
globfp3d_native_cell_size(dataset::GlobalBuildingFootprints3D) =
    rad2deg(globfp3d_native_resolution(dataset) / Oceananigans.defaults.planet_radius)

# Nominal global size; only the windowed portion is ever materialized.
function Base.size(dataset::GlobalBuildingFootprints3D, variable)
    Δ = globfp3d_native_cell_size(dataset)
    Nx = round(Int, 360 / Δ)
    Ny = round(Int, 180 / Δ)
    return (Nx, Ny, 1)
end

DataWrangling.metadata_filename(dataset::GlobalBuildingFootprints3D, name, date, region) =
    string("GlobalBuildingFootprints3D_", dataset.resolution, "m_", bounding_box_suffix(region), ".nc")

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

DataWrangling.longitude_name(::GlobalBuildingFootprints3DMetadatum) = "lon"
DataWrangling.latitude_name(::GlobalBuildingFootprints3DMetadatum)  = "lat"

# A building height of 0 over unbuilt land is physical, not a gap: never inpaint.
DataWrangling.default_inpainting(::GlobalBuildingFootprints3DMetadatum) = nothing

Oceananigans.Fields.location(::GlobalBuildingFootprints3DMetadatum) = (Center, Center, Center)

#####
##### Per-cell morphometry from the fine building-height raster
#####

# Target cell each fine coordinate falls in, resolved once per fine row/column rather than per
# cell. Coordinates outside the target hull come back as 0 or `Nx+1`, dropped by the bounds check
# in the loop below.
target_index_map(faces, coordinates) = Int[searchsortedlast(faces, c) for c in coordinates]

"""
    reduce_morphometry(height, longitudes, latitudes, target_grid)

Reduce the fine building-height raster `height` (m), on the regular grid of cell-center
`longitudes`/`latitudes` (degrees), onto `target_grid`. Returns the arrays that
[`building_morphometry`](@ref) wraps in `Field`s; empty target cells are `0`.

Fine cells are binned by the target cell faces, so a stretched `target_grid` works too.
"""
function reduce_morphometry(height, longitudes, latitudes, target_grid::LatitudeLongitudeGrid)
    Nx = size(target_grid, 1)
    Ny = size(target_grid, 2)
    λfaces   = λnodes(target_grid, Face())
    φfaces   = φnodes(target_grid, Face())
    φcenters = φnodes(target_grid, Center())

    nx, ny = size(height)
    δλ = nx > 1 ? longitudes[2] - longitudes[1] : zero(eltype(longitudes))
    δφ = ny > 1 ? latitudes[2]  - latitudes[1]  : zero(eltype(latitudes))
    Imap = target_index_map(λfaces, longitudes)
    Jmap = target_index_map(φfaces, latitudes)

    count_total     = zeros(Int, Nx, Ny)
    count_built     = zeros(Int, Nx, Ny)
    Σh              = zeros(Float64, Nx, Ny)
    Σh²             = zeros(Float64, Nx, Ny)
    running_maximum = zeros(Float64, Nx, Ny)
    Σδhˣ            = zeros(Float64, Nx, Ny)
    Σδhʸ            = zeros(Float64, Nx, Ny)

    @inbounds for j in 1:ny, i in 1:nx
        I = Imap[i]
        J = Jmap[j]
        (1 <= I <= Nx && 1 <= J <= Ny) || continue
        h = height[i, j]
        count_total[I, J] += 1
        if h > 0
            count_built[I, J] += 1
            Σh[I, J]  += h
            Σh²[I, J] += h * h
            running_maximum[I, J] = max(running_maximum[I, J], h)
        end
        i < nx && (Σδhˣ[I, J] += abs(height[i + 1, j] - h))
        j < ny && (Σδhʸ[I, J] += abs(height[i, j + 1] - h))
    end

    plan_area_index           = zeros(Float64, Nx, Ny)
    mean_building_height      = zeros(Float64, Nx, Ny)
    building_height_deviation = zeros(Float64, Nx, Ny)
    maximum_building_height   = running_maximum
    gross_building_height     = zeros(Float64, Nx, Ny)
    frontal_area_index        = zeros(Float64, Nx, Ny)

    # Fine-cell side lengths in meters, at each target row's latitude.
    R = target_grid.radius
    for J in 1:Ny
        φc = φcenters[J]
        dy = R * deg2rad(δφ)
        dx = R * deg2rad(δλ) * cosd(φc)
        for I in 1:Nx
            nt = count_total[I, J]
            nb = count_built[I, J]
            if nt > 0
                plan_area_index[I, J]       = nb / nt
                gross_building_height[I, J] = Σh[I, J] / nt
                cell_area = nt * dx * dy
                frontal_area_index[I, J] = cell_area > 0 ?
                    (Σδhˣ[I, J] * dy + Σδhʸ[I, J] * dx) / (4 * cell_area) : 0.0
            end
            if nb > 0
                h̄ = Σh[I, J] / nb
                mean_building_height[I, J]      = h̄
                building_height_deviation[I, J] = sqrt(max(Σh²[I, J] / nb - h̄^2, 0.0))
            end
        end
    end

    return (; mean_building_height, building_height_deviation, maximum_building_height,
              plan_area_index, frontal_area_index, gross_building_height)
end

const morphometry_names = (:mean_building_height, :building_height_deviation, :maximum_building_height,
                           :plan_area_index, :frontal_area_index, :gross_building_height)

"""
    NoIntersectingTilesError(region)

No 3D-GloBFP tile intersects `region`: the dataset has no building footprints there.
"""
struct NoIntersectingTilesError{R} <: Exception
    region :: R
end

Base.showerror(io::IO, error::NoIntersectingTilesError) =
    print(io, "No 3D-GloBFP tiles intersect the requested region ", summary(error.region), ".")

# Download, read, and reduce the fine raster of one metadatum onto (the covered rows of) `target_grid`.
function reduced_morphometry(target_grid, metadatum)
    Downloads.download(metadatum)
    height = DataWrangling.retrieve_data(metadatum)
    longitudes, latitudes = DataWrangling.read_file_coords(metadatum)
    return reduce_morphometry(height, longitudes, latitudes, target_grid)
end

function morphometry_fields(reduced, target_grid)
    arch = architecture(target_grid)
    return map(reduced) do array
        field = Field{Center, Center, Nothing}(target_grid)
        interior(field) .= on_architecture(arch, reshape(array, size(array, 1), size(array, 2), 1))
        field
    end
end

"""
    morphometry_latitude_bands(target_grid, region, Δ, maximum_raster_cells, padding)

Split the rows of `target_grid` covered by `region` into contiguous latitude bands whose native
rasters (cell size `Δ` degrees) hold at most `maximum_raster_cells` cells each (a band never
narrower than one row). Returns a vector of `(; rows, region)`: the target-grid row range and the
`BoundingBox` to rasterize for it — the full `region` longitude, and the rows' latitude interval
widened by `padding` native cells but clamped to the `region` latitude, so every band raster is a
sub-window of the single-pass raster and band rows reduce identically to a single pass.
"""
function morphometry_latitude_bands(target_grid, region, Δ, maximum_raster_cells, padding)
    Ny = size(target_grid, 2)
    φfaces = φnodes(target_grid, Face())
    south, north = region.latitude

    band_region(j₁, j₂) = BoundingBox(longitude = region.longitude,
                                      latitude = (max(φfaces[j₁] - padding * Δ, south),
                                                  min(φfaces[j₂ + 1] + padding * Δ, north)))
    band_cells(j₁, j₂) = let raster = native_region_grid(band_region(j₁, j₂), Δ, Δ)
        raster.Nx * raster.Ny
    end

    # Rows the single-pass raster reaches, including its 2-cell snap pad beyond the region;
    # rows farther out reduce to zero in a single pass too and are left out of every band.
    j₁ = findfirst(j -> φfaces[j + 1] > south - 2Δ, 1:Ny)
    j₊ = findlast(j -> φfaces[j] < north + 2Δ, 1:Ny)

    ranges = UnitRange{Int}[]
    while !isnothing(j₁) && !isnothing(j₊) && j₁ <= j₊
        j₂ = j₁
        while j₂ < j₊ && band_cells(j₁, j₂ + 1) <= maximum_raster_cells
            j₂ += 1
        end
        push!(ranges, j₁:j₂)
        j₁ = j₂ + 1
    end

    return [(rows = rows, region = band_region(first(rows), last(rows))) for rows in ranges]
end

"""
    building_morphometry(target_grid; dataset = GlobalBuildingFootprints3D(), region,
                         maximum_raster_cells = 400_000_000)

Per-cell building morphometry on `target_grid` (a `LatitudeLongitudeGrid`, coarser than the
`dataset` rasterization resolution), aggregated from the fine 3D-GloBFP building-height raster
over `region`. Returns a NamedTuple of `Field`s:

- `plan_area_index` `λᵖ` — fraction of fine cells that are built.
- `mean_building_height` `h` — mean height over the built fine cells.
- `building_height_deviation` `σʰ` — standard deviation of height over the built fine cells.
- `maximum_building_height` `hᵐᵃˣ` — maximum height.
- `gross_building_height` — mean height over all fine cells (`= λᵖ·h`), the digital surface lift.
- `frontal_area_index` `λᶠ` — windward wall area from height steps, direction-averaged:
  `(Σₓ|δh|·dy + Σᵧ|δh|·dx) / (4·A)`, with `A` the cell area.

A `region` whose native raster exceeds `maximum_raster_cells` (default `400_000_000` cells,
3.2 GB of `Float64`) is processed in latitude bands sized to the limit, so memory stays bounded
regardless of the region size. Each band's raster file is deleted once reduced (the downloaded
footprint tiles stay cached), so disk usage stays bounded too.

Downloading and rasterizing the footprints requires `using ArchGDAL`.
"""
function building_morphometry(target_grid::LatitudeLongitudeGrid; dataset = GlobalBuildingFootprints3D(),
                              region, maximum_raster_cells = 400_000_000)
    metadatum = Metadatum(:building_height; dataset, region)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)

    Δ = globfp3d_native_cell_size(dataset)
    raster = native_region_grid(region, Δ, Δ)
    raster.Nx * raster.Ny <= maximum_raster_cells &&
        return morphometry_fields(reduced_morphometry(target_grid, metadatum), target_grid)

    # A band bounding box also selects the footprint tiles to burn: pad it by ~200 m of native
    # cells so buildings overhanging a tile just outside the band still land in the band's rows.
    padding = cld(200, globfp3d_native_resolution(dataset))
    bands = morphometry_latitude_bands(target_grid, region, Δ, maximum_raster_cells, padding)
    @info string(summary(dataset), ": the raster over ", summary(region), " has ",
                 raster.Nx, " × ", raster.Ny, " cells, above maximum_raster_cells = ",
                 maximum_raster_cells, "; reducing in ", length(bands), " latitude bands.")

    Nx = size(target_grid, 1)
    Ny = size(target_grid, 2)
    accumulated = NamedTuple{morphometry_names}(ntuple(_ -> zeros(Float64, Nx, Ny), length(morphometry_names)))
    any_tiles = false
    for (b, band) in enumerate(bands)
        @info string("Reducing latitude band ", b, " of ", length(bands),
                     " (target rows ", band.rows, ", ", latitude_summary(band.region.latitude), ")...")
        band_metadatum = Metadatum(:building_height; dataset, region = band.region)
        reduced = try
            reduced_morphometry(target_grid, band_metadatum)
        catch exception
            # An all-water/unbuilt band intersects no tiles and reduces to zero, as in a single pass.
            exception isa NoIntersectingTilesError || rethrow()
            continue
        end
        any_tiles = true
        @root rm(metadata_path(band_metadatum); force = true)
        for name in morphometry_names
            accumulated[name][:, band.rows] .= reduced[name][:, band.rows]
        end
    end
    any_tiles || throw(NoIntersectingTilesError(region))

    return morphometry_fields(accumulated, target_grid)
end

#####
##### Tile discovery — 3D-GloBFP ships one `.zip` per grid tile across ten figshare parts.
#####

# figshare article ids of the ten parts, from the Zenodo record's data_links.txt.
const GLOBFP3D_FIGSHARE_ARTICLE_IDS = (28879733, 28881749, 28882700, 28889813, 28890593,
                                       28891631, 28903454, 28903853, 28904453, 28906499)

"""
    globfp3d_parse_tile_bounds(name)

Parse a 3D-GloBFP tile filename `gridID_lon1_lat1_lon2_lat2_…`, whose coordinates are the SW and
NE corners, into `(; gid, west, south, east, north)`. Returns `nothing` if it does not match.
"""
function globfp3d_parse_tile_bounds(name)
    m = match(r"^(\d+)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_", name)
    isnothing(m) && return nothing
    W, S, E, N = parse.(Float64, (m[2], m[3], m[4], m[5]))
    return (; gid = parse(Int, m[1]), west = W, south = S, east = E, north = N)
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

# Implemented in ext/NumericalEarthArchGDALExt/globfp3d.jl.
globfp3d_rasterize_to_netcdf(metadatum, nc_path) =
    error("Reading the 3D-GloBFP footprint shapefiles requires the ArchGDAL package " *
          "(for the OGR vector read + rasterization). Load it with `using ArchGDAL`.")

end # module GloBFP3D
