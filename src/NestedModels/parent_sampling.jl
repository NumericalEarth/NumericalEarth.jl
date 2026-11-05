#####
##### interpolation_node: the coordinate at which a child samples its parent
#####

using Oceananigans: instantiated_location
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: interior_indices, node
using Oceananigans.Utils: launch!, KernelParameters
using KernelAbstractions: @kernel, @index

"""
    interpolation_node(i, j, k, grid, ℓx, ℓy, ℓz, from_grid)

Coordinate at which a child on `grid` samples a parent on `from_grid`, for the child node
`(i, j, k)` at location `(ℓx, ℓy, ℓz)`. `interpolate` locates the vertical through
`from_grid`'s `znodes`, so the node must be in the same vertical coordinate: the child's physical
`node` by default, and the reference node for a terrain-following source, whose `znodes` are its
reference levels.
"""
@inline interpolation_node(i, j, k, grid, ℓx, ℓy, ℓz, from_grid) = node(i, j, k, grid, ℓx, ℓy, ℓz)

@kernel function _interpolate_from_parent!(child_field, grid, loc, source, source_loc, source_grid)
    i, j, k = @index(Global, NTuple)
    X = interpolation_node(i, j, k, grid, loc..., source_grid)
    @inbounds child_field[i, j, k] = interpolate(X, source, source_loc, source_grid)
end

"""
    interpolate_from_parent!(child_field, source)

Fill `child_field` by interpolating the parent field `source` at each child node, queried at
[`interpolation_node`](@ref), then fill its halos.
"""
function interpolate_from_parent!(child_field, source)
    grid = child_field.grid
    launch!(architecture(child_field), grid, KernelParameters(interior_indices(child_field)),
            _interpolate_from_parent!, child_field, grid, instantiated_location(child_field),
            source, instantiated_location(source), source.grid)
    fill_halo_regions!(child_field)
    return child_field
end
