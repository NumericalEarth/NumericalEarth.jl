#####
##### query_node: the coordinate at which a child samples its parent
#####

using Oceananigans: instantiated_location
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: interior_indices, node
using Oceananigans.Utils: launch!, KernelParameters
using KernelAbstractions: @kernel, @index

"""
    query_node(source_grid, i, j, k, grid, ℓx, ℓy, ℓz)

Coordinate at which a child on `grid` samples a parent on `source_grid`, for the child node
`(i, j, k)` at location `(ℓx, ℓy, ℓz)`. `interpolate` locates the vertical through the source
grid's `znodes`, so the query must be in the same vertical coordinate: the child's physical
`node` by default, and the reference node for a terrain-following source, whose `znodes` are its
reference levels.
"""
@inline query_node(source_grid, i, j, k, grid, ℓx, ℓy, ℓz) = node(i, j, k, grid, ℓx, ℓy, ℓz)

@kernel function _interpolate_from_parent!(child_field, grid, loc, source, source_loc, source_grid)
    i, j, k = @index(Global, NTuple)
    X = query_node(source_grid, i, j, k, grid, loc...)
    @inbounds child_field[i, j, k] = interpolate(X, source, source_loc, source_grid)
end

"""
    interpolate_from_parent!(child_field, source)

Fill `child_field` by interpolating the parent field `source` at each child node, queried at
[`query_node`](@ref), then fill its halos.
"""
function interpolate_from_parent!(child_field, source)
    grid = child_field.grid
    launch!(architecture(child_field), grid, KernelParameters(interior_indices(child_field)),
            _interpolate_from_parent!, child_field, grid, instantiated_location(child_field),
            source, instantiated_location(source), source.grid)
    fill_halo_regions!(child_field)
    return child_field
end
