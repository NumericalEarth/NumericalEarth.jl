#####
##### query_node: the coordinate at which a child samples its parent
#####
#
# `interpolate` locates the vertical through the SOURCE grid's 1-D `znodes`, so the query must be
# expressed in the coordinate those nodes span: physical height for a geopotential parent (what `node`
# returns, the default below), but the reference levels for a terrain-following one, whose
# `znodes(grid, ℓz) = rnodes(grid, ℓz)` while `node` returns the terrain-deformed height. Dispatching
# on the source grid keeps both in the same coordinate; the terrain-following method lives in the
# Breeze extension. Sampling in the terrain-following coordinate is also what nesting wants — no child
# node maps below the parent's ground.

"""
    query_node(source_grid, i, j, k, grid, ℓx, ℓy, ℓz)

Coordinate at which a child on `grid` samples a parent living on `source_grid`, for the child node
`(i, j, k)` at location `(ℓx, ℓy, ℓz)`. Defaults to the child's physical `node`; a source grid whose
`znodes` are not physical heights (e.g. a terrain-following grid) overrides this to return the
matching coordinate.

An override reads that coordinate off `grid`, so it assumes the child shares the parent's kind of
vertical coordinate — the case for nested grids. A child without one (a flat-bottom, fixed-height LES
under a terrain-following parent) has none to offer; that case needs the parent's column mapping
inverted, not a different query.
"""
@inline query_node(source_grid, i, j, k, grid, ℓx, ℓy, ℓz) = node(i, j, k, grid, ℓx, ℓy, ℓz)

@kernel function _interpolate_from_parent!(child_field, grid, loc, source, source_grid)
    i, j, k = @index(Global, NTuple)
    X = query_node(source_grid, i, j, k, grid, loc...)
    @inbounds child_field[i, j, k] = interpolate(X, source, loc, source_grid)
end

"""
$(TYPEDSIGNATURES)

Fill `child_field` by interpolating the parent field `source` at each child node, sampling in the
parent's own vertical coordinate ([`query_node`](@ref)). `Oceananigans.interpolate!` with that
convention: identical for a parent whose vertical is physical height, terrain-offset-free for a
terrain-following one.
"""
function interpolate_from_parent!(child_field, source)
    grid = child_field.grid
    loc = instantiated_location(child_field)
    launch!(architecture(child_field), grid, KernelParameters(interior_indices(child_field)),
            _interpolate_from_parent!, child_field, grid, loc, source, source.grid)
    fill_halo_regions!(child_field)
    return child_field
end
