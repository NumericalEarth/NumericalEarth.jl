#####
##### Blending a nested ocean's bathymetry toward its parent's along the open sides
#####
#
# The barotropic transport through an open side is ∫u dz over the child's own water column, so where the
# child's seafloor differs from the parent's, the parent transport it is driven with does not fit the
# geometry it enters. Blending the child bathymetry toward the parent's over the outermost cells of each
# open side removes that mismatch while leaving the interior at the child's own resolution.

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid, bottom_height_field,
                                       compute_numerical_bottom_height!, has_active_cells_map
using KernelAbstractions: @kernel, @index

const GridFittedBottomGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

"""
    blend_parent_bathymetry!(grid, parent_bottom_height, sides; width, minimum_depth)

Blend the immersed bottom of `grid` in place toward `parent_bottom_height` (a bottom height `Field`, interpolated
onto `grid`) within `width` cells of each side in `sides`. The weight of the parent's seafloor is a raised cosine,
1 in the boundary cell and 0 `width` cells inward. Columns that are land in either bathymetry keep the child's
seafloor, so the blend moves no coastline, and the blended seafloor is at least `minimum_depth` deep. A grid
without a grid-fitted bottom, or a parent without a known seafloor (`nothing`), is left as it is.
"""
blend_parent_bathymetry!(grid, parent_bottom_height, sides; kw...) = grid
blend_parent_bathymetry!(grid::GridFittedBottomGrid, ::Nothing, sides; kw...) = grid

function blend_parent_bathymetry!(grid::GridFittedBottomGrid, parent_bottom_height, sides; width, minimum_depth)
    width > 0 || return grid

    # TODO: Oceananigans could provide a public way to reset the bottom of an existing ImmersedBoundaryGrid,
    # rebuilding its active cells map with it.
    has_active_cells_map(grid) &&
        throw(ArgumentError("blending the bathymetry of a grid with an active cells map would leave the map " *
                            "stale; build the grid without one, or pass `bathymetry_blend_width = 0`."))

    underlying_grid = grid.underlying_grid
    bottom_height = bottom_height_field(grid)   # shares the grid's bottom height data

    parent_height = Field{Center, Center, Nothing}(underlying_grid)
    interpolate!(parent_height, parent_bottom_height)

    Nx, Ny, _ = size(underlying_grid)
    open = map(side -> side in sides, lateral_sides)
    FT = eltype(underlying_grid)

    launch!(architecture(grid), underlying_grid, :xy, _blend_parent_bathymetry!,
            bottom_height, parent_height, width, Nx, Ny, open, convert(FT, minimum_depth))

    fill_halo_regions!(bottom_height)
    immersed_boundary = GridFittedBottom(bottom_height, grid.immersed_boundary.immersed_condition)
    compute_numerical_bottom_height!(bottom_height, underlying_grid, immersed_boundary)

    return grid
end

@kernel function _blend_parent_bathymetry!(bottom_height, parent_height, width, Nx, Ny, open, minimum_depth)
    i, j = @index(Global, NTuple)
    west, east, south, north = open

    # Distance in cells from the nearest open side; `width` (zero weight) away from them
    d = min(ifelse(west, i - 1, width), ifelse(east, Nx - i, width),
            ifelse(south, j - 1, width), ifelse(north, Ny - j, width), width)

    α = (1 + cos(π * d / width)) / 2

    @inbounds begin
        hᶜ = bottom_height[i, j, 1]
        hᵖ = parent_height[i, j, 1]
        wet = (hᶜ < 0) & (hᵖ < 0)
        blended = min(-minimum_depth, (1 - α) * hᶜ + α * hᵖ)
        bottom_height[i, j, 1] = ifelse(wet & (d < width), blended, hᶜ)
    end
end
