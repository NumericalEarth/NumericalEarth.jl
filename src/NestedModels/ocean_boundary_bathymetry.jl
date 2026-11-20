#####
##### Blending a nested ocean's bathymetry toward its parent's along the open sides
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: AbstractGridFittedBottom, GridFittedBottom, PartialCellBottom, ImmersedBoundaryGrid,
                                       bottom_height_field, has_active_cells_map, has_active_z_columns
using KernelAbstractions: @kernel, @index

const GridFittedBottomGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

# The child's bottom blended toward the parent's with a raised cosine within `width` cells (at least 2) of `sides`
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
    west, east, south, north = open
    FT = eltype(underlying_grid)

    launch!(architecture(grid), underlying_grid, :xy, _blend_parent_bathymetry!,
            bottom_height, parent_height, width, Nx, Ny, open, convert(FT, minimum_depth))

    # South and north first, so that the corners end up level with both of their neighbors
    launch!(architecture(grid), underlying_grid, :xy, _level_boundary_cells!, bottom_height, Val(2), Ny, south, north)
    launch!(architecture(grid), underlying_grid, :xy, _level_boundary_cells!, bottom_height, Val(1), Nx, west, east)

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

# Level the boundary cells of the sides normal to dimension `dim`, `N` cells apart, with their inward neighbors
@kernel function _level_boundary_cells!(bottom_height, dim, N, left_open, right_open)
    i, j = @index(Global, NTuple)
    n, s = horizontal_indices(dim, i, j)

    @inbounds begin
        inward = ifelse(n == 1, 2, N - 1)
        boundary_cell = (left_open & (n == 1)) | (right_open & (n == N))
        hᵇ = bottom_height[i, j, 1]
        hⁱ = bottom_height[horizontal_indices(dim, inward, s)..., 1]
        bottom_height[i, j, 1] = ifelse(boundary_cell & (hᵇ < 0) & (hⁱ < 0), hⁱ, hᵇ)
    end
end
