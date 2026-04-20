using Oceananigans.Grids: λnode, φnode, on_architecture
using ImageMorphology

#####
##### Basin struct
#####

"""
    Basin{M, G, S}

A connected water region (ocean basin) identified on a grid, together with the
boolean mask that labels cells belonging to it and the seed points used to pick
the connected component.

Fields
======
- `mask`: a 2D `Field{Center, Center, Nothing}` with `Bool` values — `true` for
          cells belonging to the basin, `false` otherwise.
- `grid`: the grid on which the basin is defined.
- `seed_points`: `(λ, φ)` coordinate pairs used to identify which connected
                 component corresponds to the desired basin. The algorithm finds
                 the label at each seed point and builds the mask from that label.
"""
struct Basin{M, G, S}
    mask :: M
    grid :: G
    seed_points :: S
end

Base.summary(basin::Basin) = "Basin"

function Base.show(io::IO, basin::Basin)
    print(io, summary(basin), " on ", summary(basin.grid))
end

# Forward getindex to the underlying mask
Base.getindex(basin::Basin, i, j, k) = basin.mask[i, j, k]

#####
##### Basin union
#####

"""
    Base.:|(basin1::Basin, basin2::Basin)

Combine two basins into a new one whose mask is the union of both. Cells that
belong to either input basin are `true` in the combined mask. This mirrors the
elementwise `.|` applied to the underlying boolean masks.
"""
function Base.:|(basin1::Basin, basin2::Basin)
    grid = basin1.grid
    combined_mask = Field{Center, Center, Nothing}(grid, Bool)

    # Union: a cell is in the combined mask if it is in either input mask
    parent(combined_mask) .= parent(basin1.mask) .| parent(basin2.mask)
    fill_halo_regions!(combined_mask)

    # Combine seed points
    combined_seeds = (basin1.seed_points..., basin2.seed_points...)

    return Basin(combined_mask, grid, combined_seeds)
end

#####
##### Connected component utilities
#####

"""
    find_label_at_point(labels, grid, λs, φs; radius = 2)

Find the connected component label at the given longitude/latitude seed point.
Returns the label value, or 0 if the point is on land or outside the domain.
Checks in a circular cap of radius `radius` degrees around the seed point
"""
function find_label_at_point(labels, grid, λs, φs; radius = 2)
    Nx, Ny, _ = size(grid)

    # Find grid cell containing the seed point
    for j in 1:Ny
        for i in 1:Nx
            λ = λnode(i, j, 1, grid, Center(), Center(), Center())
            φ = φnode(i, j, 1, grid, Center(), Center(), Center())

            λ  = convert_to_0_360(λ)

            Δλ = if isnothing(λs)
                zero(λ)
            else
                λs = convert_to_0_360(λs)
                λ - λs
            end

            # Check if this cell contains the seed point (within a certain radius)
            if Δλ^2 + (φ - φs)^2 < radius^2
                return labels[i, j]
            end
        end
    end

    return 0  # Seed point not found
end

"""
    create_basin_mask_from_label(grid, labels, basin_label)

Create a mask field for all cells with the given label.
"""
function create_basin_mask_from_label(grid, labels, basin_label)
    Nx, Ny, _ = size(grid)

    mask = Field{Center, Center, Nothing}(grid, Bool)

    launch!(CPU(), grid, :xy, _compute_basin_mask!,
            mask, grid, labels, basin_label)

    fill_halo_regions!(mask)

    return mask
end

@kernel function _compute_basin_mask!(mask, grid, labels, basin_label)
    i, j = @index(Global, NTuple)
    correct_basin = @inbounds labels[i, j] == basin_label
    @inbounds mask[i, j, 1] = correct_basin
end

#####
##### Some useful Basin seeds and barriers
#####

const SOUTHERN_OCEAN_SEPARATION_BARRIER = BoundingBox(longitude=(-180.0, 180.0), latitude=(-56.0, -54.0))

const ATLANTIC_OCEAN_BARRIERS = [
    meridional_barrier(20.0, -90.0, -30.0),   # Cape Agulhas
    meridional_barrier(289.0, -90.0, -30.0),  # Drake Passage
]

const INDIAN_OCEAN_BARRIERS = [
    meridional_barrier(141.0, -90.0, -3.0),                              # Indonesian side
    meridional_barrier(20.0,  -90.0, -30.0),                             # Cape Agulhas
    BoundingBox(longitude=(105.0, 141.0), latitude=(-4.0, -3.0)),        # Indonesian/Asian seas (zonal barrier at 3.5ᵒ S)
]

const SOUTHERN_OCEAN_BARRIERS = [SOUTHERN_OCEAN_SEPARATION_BARRIER]

const PACIFIC_OCEAN_BARRIERS = [
    meridional_barrier(141.0, -90.0, -3.0),                              # Indonesian side
    meridional_barrier(20.0,  -90.0, -30.0),                             # Cape Agulhas
    BoundingBox(longitude=(105.0, 141.0), latitude=(-4.0, -3.0)),        # Indonesian/Asian seas (zonal barrier at 3.5ᵒ S)
]

# Seed points for Atlantic Ocean (definitely in the Atlantic)
const ATLANTIC_SEED_POINTS = [
    (-30.0, 0.0),    # Central equatorial Atlantic
    (-40.0, 30.0),   # North Atlantic
    (-25.0, -20.0),  # South Atlantic
]

# Seed points for Indian Ocean
const INDIAN_SEED_POINTS = [
    (70.0, -10.0),   # Central Indian Ocean
    (60.0, 10.0),    # Arabian Sea region
    (90.0, -20.0),   # Eastern Indian Ocean
]

# Seed points for Southern Ocean
const SOUTHERN_SEED_POINTS = [
    (0.0, -60.0),     # South Atlantic sector
    (90.0, -60.0),    # Indian Ocean sector
    (180.0, -60.0),   # Pacific sector (date line)
    (-90.0, -60.0),   # South Pacific sector
]

# Seed points for Pacific Ocean
const PACIFIC_SEED_POINTS = [
    (180.0, 0.0),     # Central equatorial Pacific (dateline)
    (-150.0, 20.0),   # North Pacific (Hawaii region)
    (-120.0, -20.0),  # South Pacific
    # Same values but from 0 to 360
    (180.0, 0.0),             # Central equatorial Pacific
    (-150.0 + 360, 20.0),     # North Pacific
    (-120.0 + 360, -20.0),    # South Pacific
]

#####
##### Basin constructor
#####

add_barrier(v::AbstractVector, b::BoundingBox) = [v..., b]
add_barrier(v::BoundingBox,    b::BoundingBox) = [v, b]
add_barrier(::Nothing,         b::BoundingBox) = b

"""
    Basin(grid;
          south_boundary = nothing,
          north_boundary = nothing,
          seed_points = [(0, 0)],
          barriers = nothing)

Build a `Basin` — a single connected water region on `grid` together with its
boolean mask.

The algorithm first labels every connected water region (using
`ImageMorphology.label_components`), optionally splitting regions with
`barriers`. It then retrieves the basin whose label matches the first
`seed_point` that falls on water.

Multiple `seed_points` are tried in order as fallbacks (the first hit wins);
multiple `barriers` are applied simultaneously before labeling so that each
barrier independently blocks water connectivity.

Arguments
=========
- `grid`: An `ImmersedBoundaryGrid` whose immersed boundary defines the coastlines.

Keyword Arguments
=================
- `south_boundary`: Southern latitude limit — cells south of this become land. Default: `nothing`.
- `north_boundary`: Northern latitude limit — cells north of this become land. Default: `nothing`.
- `seed_points`: `(λ, φ)` pairs identifying the target basin. The first seed that lands
                 on water determines which connected component becomes `true` in the mask.
                 Multiple seeds are tried as fallbacks for grids where a single point
                 may fall on land. Default: `[(0, 0)]`.
- `barriers`: A `BoundingBox` (or a `Vector` of them) applied before labeling.
              Each barrier temporarily marks its horizontal rectangle as land,
              preventing the flood-fill from crossing it (e.g., closing Drake Passage
              separates the Atlantic from the Pacific). The `z` field of the
              `BoundingBox` is ignored. Default: `nothing`.
"""
function Basin(grid;
               south_boundary = nothing,
               north_boundary = nothing,
               seed_points = [(0, 0)],
               barriers = nothing)

    # The computations are 2D and require serial algorithms, so
    # we perform the computation on the CPU then move the output
    # to the GPU if the initial grid was a GPU grid
    cpu_grid = Oceananigans.on_architecture(CPU(), grid)

    # Enforce north and south boundaries
    if !isnothing(south_boundary)
        barriers = add_barrier(barriers, BoundingBox(longitude=nothing, latitude=(-90, south_boundary)))
    end

    if !isnothing(north_boundary)
        barriers = add_barrier(barriers, BoundingBox(longitude=nothing, latitude=(north_boundary, 90)))
    end

    # Compute connected component labels for all ocean cells
    # Barriers are applied to temporarily separate connected basins
    labels = label_ocean_basins(cpu_grid; barriers)

    # Find the basin label using seed points
    basin_label = 0
    for (λs, φs) in seed_points
        label = find_label_at_point(labels, cpu_grid, λs, φs)
        if label > 0
            basin_label = label
            break
        end
    end

    if basin_label == 0
        @warn "Could not find the basin in grid. Returning empty mask."
        mask = Field{Center, Center, Nothing}(grid, Bool)
        return Basin(mask, grid, seed_points)
    end

    # Create mask from label with latitude bounds
    mask = create_basin_mask_from_label(cpu_grid, labels, basin_label)
    mask = Oceananigans.on_architecture(architecture(grid), mask)

    return Basin(mask, grid, seed_points)
end

#####
##### Convenience functions for Earth's ocean basins
#####

"""
    atlantic_ocean_basin(grid; include_southern_ocean=false, kw...)

Build a `Basin` for Earth's Atlantic Ocean with predefined barriers and seed points.

Keyword Arguments
=================
- `include_southern_ocean`: If `true`, extends the Atlantic basin into the Southern Ocean
                            sector below the standard separation latitude (~55°S). Default: `false`.
- `south_boundary`: Southern latitude limit. Default: -50.0 (or -90.0 if `include_southern_ocean=true`)
- `north_boundary`: Northern latitude limit. Default: 65.0
- Other keyword arguments are passed to `Basin`.
"""
function atlantic_ocean_basin(grid;
                              include_southern_ocean = true,
                              south_boundary = include_southern_ocean ? -90.0 : -50.0,
                              north_boundary = 65.0,
                              barriers = ATLANTIC_OCEAN_BARRIERS,
                              seed_points = ATLANTIC_SEED_POINTS,
                              kw...)

    if !include_southern_ocean
        barriers = [barriers..., SOUTHERN_OCEAN_SEPARATION_BARRIER]
    end

    return Basin(grid; south_boundary, north_boundary, barriers, seed_points, kw...)
end

"""
    indian_ocean_basin(grid; include_southern_ocean=false, kw...)

Build a `Basin` for Earth's Indian Ocean with predefined barriers and seed points.

Keyword Arguments
=================
- `include_southern_ocean`: If `true`, extends the Indian basin into the Southern Ocean
                            sector below the standard separation latitude (~55°S). Default: `false`.
- `south_boundary`: Southern latitude limit. Default: -50.0 (or -90.0 if `include_southern_ocean=true`)
- `north_boundary`: Northern latitude limit. Default: 30.0
- Other keyword arguments are passed to `Basin`.
"""
function indian_ocean_basin(grid;
                            include_southern_ocean = true,
                            south_boundary = include_southern_ocean ? -90.0 : -50.0,
                            north_boundary = 30.0,
                            barriers = INDIAN_OCEAN_BARRIERS,
                            seed_points = INDIAN_SEED_POINTS,
                            kw...)

    if !include_southern_ocean
        barriers = [barriers..., SOUTHERN_OCEAN_SEPARATION_BARRIER]
    end

    return Basin(grid; south_boundary, north_boundary, barriers, seed_points, kw...)
end

"""
    southern_ocean_basin(grid; kw...)

Build a `Basin` for Earth's Southern Ocean with predefined barriers and seed points.
Default boundaries: south=-90.0, north=-35.0
"""
function southern_ocean_basin(grid;
                              south_boundary = -90.0,
                              north_boundary = -35.0,
                              barriers = SOUTHERN_OCEAN_BARRIERS,
                              seed_points = SOUTHERN_SEED_POINTS,
                              kw...)
    return Basin(grid; south_boundary, north_boundary, barriers, seed_points, kw...)
end

"""
    pacific_ocean_basin(grid; include_southern_ocean=false, kw...)

Build a `Basin` for Earth's Pacific Ocean with predefined barriers and seed points.

Keyword Arguments
=================
- `include_southern_ocean`: If `true`, extends the Pacific basin into the Southern Ocean
                            sector below the standard separation latitude (~55°S). Default: `false`.
- `south_boundary`: Southern latitude limit. Default: -50.0 (or -90.0 if `include_southern_ocean=true`)
- `north_boundary`: Northern latitude limit. Default: 65.0
- Other keyword arguments are passed to `Basin`.
"""
function pacific_ocean_basin(grid;
                             include_southern_ocean = true,
                             south_boundary = include_southern_ocean ? -90.0 : -50.0,
                             north_boundary = 65.0,
                             barriers = PACIFIC_OCEAN_BARRIERS,
                             seed_points = PACIFIC_SEED_POINTS,
                             kw...)

    if !include_southern_ocean
        barriers = [barriers..., SOUTHERN_OCEAN_SEPARATION_BARRIER]
    end

    return Basin(grid; south_boundary, north_boundary, barriers, seed_points, kw...)
end

"""
    arctic_ocean_basin(grid; kw...)

Build a `Basin` for Earth's Arctic Ocean with predefined seed points.
Default boundaries: south=65.0, north=91.0
"""
function arctic_ocean_basin(grid;
                            include_southern_ocean = true,
                            south_boundary = 65.0,
                            north_boundary = 91.0,
                            barriers = nothing,
                            seed_points = [(nothing, 90.0)],
                            kw...)

    return Basin(grid; south_boundary, north_boundary, barriers, seed_points, kw...)
end
