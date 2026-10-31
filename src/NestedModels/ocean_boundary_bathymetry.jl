#####
##### Blending a nested ocean's bathymetry toward its parent's along the open sides
#####
#
# The barotropic transport through an open side is ∫u dz over the child's own water column, so where the
# child's seafloor differs from the parent's, the parent transport it is driven with does not fit the
# geometry it enters. Blending the child bathymetry toward the parent's over the outermost cells of each
# open side removes that mismatch while leaving the interior at the child's own resolution.

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: AbstractGridFittedBottom, GridFittedBottom, PartialCellBottom, ImmersedBoundaryGrid,
                                       bottom_height_field, has_active_cells_map, has_active_z_columns
using KernelAbstractions: @kernel, @index

const GridFittedBottomGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

"""
    blend_parent_bathymetry(grid, parent_bottom_height, sides; width, minimum_depth)

Return a copy of `grid` whose immersed bottom is blended toward `parent_bottom_height` (a bottom height `Field`,
interpolated onto `grid`) within `width` cells of each side in `sides`, with a minimum `width` of 2. The weight of
the parent's seafloor is 1 in the boundary cell and the first cell inward, and falls as a raised cosine to 0 `width`
cells inward. The boundary cell then takes the seafloor of the first cell inward, so that the boundary does not sit
in a trench or behind a sill: a boundary cell deeper than its neighbor would be a pocket closed on its inward side.
Columns that are land in either bathymetry keep the child's seafloor, so the blend moves no coastline, and the
blended seafloor is at least `minimum_depth` deep. The copy keeps the immersed boundary of `grid` (a
`PartialCellBottom` its minimum fractional cell height) and
rebuilds any active cells map for the blended seafloor; `grid` itself is not changed. A grid without a grid-fitted
bottom, or a parent without a known seafloor (`nothing`), is returned as it is.
"""
blend_parent_bathymetry(grid, parent_bottom_height, sides; kw...) = grid
blend_parent_bathymetry(grid::GridFittedBottomGrid, ::Nothing, sides; kw...) = grid

function blend_parent_bathymetry(grid::GridFittedBottomGrid, parent_bottom_height, sides; width, minimum_depth)
    width > 0 || return grid
    width = max(width, 2)
    validate_band_width(grid, sides, width, "bathymetry_blend_width")

    underlying_grid = grid.underlying_grid
    bottom_height = Field{Center, Center, Nothing}(underlying_grid)
    set!(bottom_height, bottom_height_field(grid))

    parent_height = Field{Center, Center, Nothing}(underlying_grid)
    interpolate!(parent_height, parent_bottom_height)

    Nx, Ny, _ = size(underlying_grid)
    open = map(side -> side in sides, lateral_sides)
    FT = eltype(underlying_grid)

    launch!(architecture(grid), underlying_grid, :xy, _blend_parent_bathymetry!,
            bottom_height, parent_height, width, Nx, Ny, open, convert(FT, minimum_depth))

    # South and north first, so that the corners end up level with both of their neighbors
    launch!(architecture(grid), underlying_grid, :xy, _level_south_north_boundary_cells!, bottom_height, Ny, open)
    launch!(architecture(grid), underlying_grid, :xy, _level_west_east_boundary_cells!, bottom_height, Nx, open)

    immersed_boundary = with_bottom_height(grid.immersed_boundary, bottom_height)

    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary;
                                active_cells_map = has_active_cells_map(grid),
                                active_z_columns = has_active_z_columns(grid))
end

with_bottom_height(ib::GridFittedBottom, bottom_height) = GridFittedBottom(bottom_height, ib.immersed_condition)
with_bottom_height(ib::PartialCellBottom, bottom_height) = PartialCellBottom(bottom_height, ib.minimum_fractional_cell_height)

@kernel function _blend_parent_bathymetry!(bottom_height, parent_height, width, Nx, Ny, open, minimum_depth)
    i, j = @index(Global, NTuple)
    west, east, south, north = open

    # Distance in cells from the nearest open side; `width` (zero weight) away from them
    d = min(ifelse(west, i - 1, width), ifelse(east, Nx - i, width),
            ifelse(south, j - 1, width), ifelse(north, Ny - j, width), width)

    α = (1 + cos(π * max(d - 1, 0) / (width - 1))) / 2

    @inbounds begin
        hᶜ = bottom_height[i, j, 1]
        hᵖ = parent_height[i, j, 1]
        wet = (hᶜ < 0) & (hᵖ < 0)
        blended = min(-minimum_depth, (1 - α) * hᶜ + α * hᵖ)
        bottom_height[i, j, 1] = ifelse(wet & (d < width), blended, hᶜ)
    end
end

@inline level_with(hᵇ, hⁱ, boundary_cell) = ifelse(boundary_cell & (hᵇ < 0) & (hⁱ < 0), hⁱ, hᵇ)

@kernel function _level_south_north_boundary_cells!(bottom_height, Ny, open)
    i, j = @index(Global, NTuple)
    _, _, south, north = open

    @inbounds begin
        inward = ifelse(j == 1, 2, Ny - 1)
        boundary_cell = (south & (j == 1)) | (north & (j == Ny))
        bottom_height[i, j, 1] = level_with(bottom_height[i, j, 1], bottom_height[i, inward, 1], boundary_cell)
    end
end

@kernel function _level_west_east_boundary_cells!(bottom_height, Nx, open)
    i, j = @index(Global, NTuple)
    west, east, _, _ = open

    @inbounds begin
        inward = ifelse(i == 1, 2, Nx - 1)
        boundary_cell = (west & (i == 1)) | (east & (i == Nx))
        bottom_height[i, j, 1] = level_with(bottom_height[i, j, 1], bottom_height[inward, j, 1], boundary_cell)
    end
end
