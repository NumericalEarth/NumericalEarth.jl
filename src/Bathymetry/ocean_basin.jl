using Oceananigans.Grids: λnode, φnode

#####
##### Basin
#####

"""
    Basin{M, G}

A connected water region identified on a grid, together with the boolean mask labeling the cells that belong
to it.

Fields
======
- `mask`: a 2D `Field{Center, Center, Nothing}` of `Bool`s, `true` on the cells belonging to the basin.
- `grid`: the grid on which the basin is defined.
"""
struct Basin{M, G}
    mask :: M
    grid :: G
end

Base.summary(basin::Basin) = "Basin"
Base.show(io::IO, basin::Basin) = print(io, summary(basin), " on ", summary(basin.grid))

"""
$(TYPEDSIGNATURES)

Return the connected component label at the longitude/latitude seed point `(λs, φs)`, searching within a cap
of radius `radius` degrees, or zero if the seed point falls outside the domain.
"""
function find_label_at_point(labels, grid, λs, φs; radius = 2)
    Nx, Ny, _ = size(grid)

    for j in 1:Ny, i in 1:Nx
        λ = convert_to_0_360(λnode(i, j, 1, grid, Center(), Center(), Center()))
        φ = φnode(i, j, 1, grid, Center(), Center(), Center())
        Δλ = isnothing(λs) ? zero(λ) : λ - convert_to_0_360(λs)

        if Δλ^2 + (φ - φs)^2 < radius^2
            return labels[i, j]
        end
    end

    return 0
end

#####
##### Barriers and seed points of Earth's ocean basins
#####

const atlantic_ocean_barriers = [
    meridional_barrier(20,  -90, -30),   # Cape Agulhas
    meridional_barrier(289, -90, -30),   # Drake Passage
]

# The same barriers bound the Indian and the Pacific: 141ᵒ E separates them from each other, Cape Agulhas
# separates both from the Atlantic, and the zonal barrier closes the Indonesian and Asian seas.
const indo_pacific_barriers = [
    meridional_barrier(141, -90, -3),
    meridional_barrier(20,  -90, -30),
    BoundingBox(longitude=(105, 141), latitude=(-4, -3)),
]

const southern_ocean_barriers = [BoundingBox(longitude=(-180, 180), latitude=(-56, -54))]

const atlantic_seed_points = [
    (-30, 0),    # Central equatorial Atlantic
    (-40, 30),   # North Atlantic
    (-25, -20),  # South Atlantic
]

const indian_seed_points = [
    (70, -10),   # Central Indian Ocean
    (60, 10),    # Arabian Sea
    (90, -20),   # Eastern Indian Ocean
]

const southern_seed_points = [
    (0,   -60),   # South Atlantic sector
    (90,  -60),   # Indian Ocean sector
    (180, -60),   # Pacific sector (date line)
    (-90, -60),   # South Pacific sector
]

const pacific_seed_points = [
    (180,  0),    # Central equatorial Pacific (dateline)
    (-150, 20),   # North Pacific (Hawaii)
    (-120, -20),  # South Pacific
]

#####
##### Basin constructors
#####

"""
$(TYPEDSIGNATURES)

Build a `Basin` — a single connected water region on `grid` together with its boolean mask.

Every connected water region is labeled with [`label_ocean_basins`](@ref), and the region containing the first
`seed_point` that falls on water becomes the basin.

Arguments
=========
- `grid`: an `ImmersedBoundaryGrid` whose immersed boundary defines the coastlines.

Keyword Arguments
=================
- `south_boundary`: southern latitude limit; cells south of it become land. Default: `nothing`.
- `north_boundary`: northern latitude limit; cells north of it become land. Default: `nothing`.
- `seed_points`: `(λ, φ)` pairs identifying the basin, tried in order. Default: `[(0, 0)]`.
- `barriers`: a vector of `BoundingBox`es marked as land before labeling. Default: `nothing`.
"""
function Basin(grid;
               south_boundary = nothing,
               north_boundary = nothing,
               seed_points = [(0, 0)],
               barriers = nothing)

    # The labeling is two-dimensional and serial, so it is performed on the CPU
    cpu_grid = on_architecture(CPU(), grid)

    if !isnothing(south_boundary)
        barriers = add_barrier(barriers, BoundingBox(longitude=nothing, latitude=(-90, south_boundary)))
    end

    if !isnothing(north_boundary)
        barriers = add_barrier(barriers, BoundingBox(longitude=nothing, latitude=(north_boundary, 90)))
    end

    labels = label_ocean_basins(cpu_grid; barriers)

    basin_label = 0
    for (λs, φs) in seed_points
        basin_label = find_label_at_point(labels, cpu_grid, λs, φs)
        basin_label > 0 && break
    end

    if basin_label == 0
        @warn "Could not find the basin in grid. Returning empty mask."
        return Basin(Field{Center, Center, Nothing}(grid, Bool), grid)
    end

    mask = Field{Center, Center, Nothing}(cpu_grid, Bool)
    interior(mask, :, :, 1) .= labels .== basin_label
    fill_halo_regions!(mask)

    return Basin(on_architecture(architecture(grid), mask), grid)
end

"""
$(TYPEDSIGNATURES)

Build a [`Basin`](@ref) from a basin's predefined `barriers` and `seed_points`.

`include_southern_ocean = false` closes the basin at 50ᵒ S instead of extending it to the pole. Remaining
keyword arguments go to `Basin`.
"""
function ocean_basin(grid, barriers, seed_points;
                     include_southern_ocean = true,
                     south_boundary = include_southern_ocean ? -90 : -50,
                     north_boundary,
                     kw...)

    return Basin(grid; south_boundary, north_boundary, barriers, seed_points, kw...)
end

"""
$(TYPEDSIGNATURES)

Earth's Atlantic Ocean. Keyword arguments go to `ocean_basin`.
"""
atlantic_ocean_basin(grid; kw...) = ocean_basin(grid, atlantic_ocean_barriers, atlantic_seed_points; north_boundary = 65, kw...)

"""
$(TYPEDSIGNATURES)

Earth's Indian Ocean. Keyword arguments go to `ocean_basin`.
"""
indian_ocean_basin(grid; kw...) = ocean_basin(grid, indo_pacific_barriers, indian_seed_points; north_boundary = 30, kw...)

"""
$(TYPEDSIGNATURES)

Earth's Pacific Ocean. Keyword arguments go to `ocean_basin`.
"""
pacific_ocean_basin(grid; kw...) = ocean_basin(grid, indo_pacific_barriers, pacific_seed_points; north_boundary = 65, kw...)

"""
$(TYPEDSIGNATURES)

Earth's Southern Ocean. Keyword arguments go to `ocean_basin`.
"""
southern_ocean_basin(grid; kw...) = ocean_basin(grid, southern_ocean_barriers, southern_seed_points; south_boundary = -90, north_boundary = -35, kw...)

"""
$(TYPEDSIGNATURES)

Earth's Arctic Ocean. Keyword arguments go to `ocean_basin`.
"""
arctic_ocean_basin(grid; kw...) = ocean_basin(grid, nothing, [(nothing, 90)]; south_boundary = 65, north_boundary = 91, kw...)
