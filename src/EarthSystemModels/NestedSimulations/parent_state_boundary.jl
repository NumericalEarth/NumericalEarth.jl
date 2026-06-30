#####
##### ParentBoundary: a lateral boundary value sampled on the fly from the parent state. A thin carrier
##### that pairs Oceananigans' boundary-regularization tags (Dim / Side / location) with a concrete
##### `value` sampler — a callable `value(X, t) -> child prognostic boundary value` supplied by the
##### consuming model (Breeze, in `NumericalEarthBreezeExt`). The sampler owns the parent fields, grid,
##### and physics (interpolating each raw parent field and combining); this carrier only knows how to
##### locate the boundary-face node and call it. Mirrors the discrete per-variable relaxation forcings:
##### the same per-variable samplers drive both the lateral BCs (here) and the interior Davies relaxation.
#####
#
# Reuses the node / boundary-index / clock helpers from `interpolated_fts_boundary.jl`.

# Clamp the z-component of a position `X` into `grid`'s center-z range (constant extrapolation in z
# beyond it), so a child node below (or above) the parent's vertical extent samples the parent's edge
# value rather than the halo-dependent / linearly-extrapolated value `interpolate` gives outside the
# grid. `validate_parent_bracket` intentionally lets the child exceed the parent in z (e.g. ERA5
# pressure-level data doesn't reach the surface); this makes the resulting extrapolation well-defined.
@inline function z_clamp(X, grid)
    Nz = size(grid, 3)
    z₁ = znode(1, 1, 1,  grid, Center(), Center(), Center())
    zₙ = znode(1, 1, Nz, grid, Center(), Center(), Center())
    return (X[1], X[2], clamp(X[3], min(z₁, zₙ), max(z₁, zₙ)))
end

# The parent must bracket the child *horizontally* (a too-small parent region there is a real error);
# the vertical is intentionally NOT required to bracket (`z_clamp` handles the overshoot).
@inline function validate_parent_bracket(source_grid, grid, ::Type{LX}, ::Type{LY}, ::Type{LZ}) where {LX, LY, LZ}
    sim_loc = (LX(), LY(), LZ())
    src_loc = (Center(), Center(), Center())
    for (label, nodes_fn) in (("x", Oceananigans.Grids.xnodes), ("y", Oceananigans.Grids.ynodes))
        sim_lo, sim_hi = extrema(nodes_fn(grid, sim_loc...))
        src_lo, src_hi = extrema(nodes_fn(source_grid, src_loc...))
        (src_lo ≤ sim_lo && sim_hi ≤ src_hi) || throw(ArgumentError(
            "Parent $(label)-extent [$src_lo, $src_hi] does not bracket child $(label)-extent [$sim_lo, $sim_hi]"))
    end
    return nothing
end

struct ParentBoundary{Dim, SideType, LX, LY, LZ, V, G}
    value       :: V   # concrete sampler: (X, t) -> child prognostic boundary value
    source_grid :: G   # parent grid, for the horizontal bracket check at regularization time
end

# User-facing-ish constructor — pre-regularization, dim/side/location tags are `Nothing`.
ParentBoundary(value, source_grid) =
    ParentBoundary{Nothing, Nothing, Nothing, Nothing, Nothing, typeof(value), typeof(source_grid)}(value, source_grid)

@inline ParentBoundary{Dim, SideType, LX, LY, LZ}(value::V, source_grid::G) where {Dim, SideType, LX, LY, LZ, V, G} =
    ParentBoundary{Dim, SideType, LX, LY, LZ, V, G}(value, source_grid)

Adapt.adapt_structure(to, c::ParentBoundary{D, S, LX, LY, LZ}) where {D, S, LX, LY, LZ} =
    ParentBoundary{D, S, LX, LY, LZ}(adapt(to, c.value), adapt(to, c.source_grid))

function regularize_boundary_condition(c::ParentBoundary{Nothing}, grid, loc, dim, SideType, args...)
    LX = typeof(loc[1]); LY = typeof(loc[2]); LZ = typeof(loc[3])
    validate_parent_bracket(c.source_grid, grid, LX, LY, LZ)
    return ParentBoundary{dim, SideType, LX, LY, LZ}(c.value, c.source_grid)
end

# `getbc` evaluates the sampler at the boundary-face node in the normal direction. The sampler
# `z_clamp`s and interpolates the parent internally, so this carrier is dataset/prognostic-agnostic.
@inline function getbc(bc::ParentBoundary{1, S, LX, LY, LZ},
                       j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    i = _boundary_index(S, grid.Nx)
    X = node(i, j, k, grid, Face(), LY(), LZ())
    return bc.value(X, clock_time(clock))
end

@inline function getbc(bc::ParentBoundary{2, S, LX, LY, LZ},
                       i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    j = _boundary_index(S, grid.Ny)
    X = node(i, j, k, grid, LX(), Face(), LZ())
    return bc.value(X, clock_time(clock))
end

@inline function getbc(bc::ParentBoundary{3, S, LX, LY, LZ},
                       i::Integer, j::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    k = _boundary_index(S, grid.Nz)
    X = node(i, j, k, grid, LX(), LY(), Face())
    return bc.value(X, clock_time(clock))
end
