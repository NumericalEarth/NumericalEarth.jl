#####
##### ParentStateBoundary: a boundary value computed on the fly from several interpolated parent
##### state fields. Internal — not a user-facing abstraction. The simple single-source case is
##### `interpolated_fts_boundary.jl`; this is the multi-field generalization the nesting path uses
##### to derive a child prognostic (e.g. Breeze's ρ, ρu, ρθˡⁱ, ρqᵉ) from raw parent state.
#####
#
# At each boundary-face node it interpolates each source in `sources` (a `NamedTuple` of parent
# `FieldTimeSeries`) — each with its own `interpolations` transform (e.g. `log` for pressure and
# temperature, `identity` otherwise) — and passes the resulting `NamedTuple` of scalar state values
# to `transform`, a pointwise map to the child prognostic. `transform` and the per-source
# interpolation transforms are supplied by the consuming model (Breeze, in `NumericalEarthBreezeExt`);
# this carrier knows nothing about which prognostic or which dataset.
#
# Reuses the node / boundary-index / clock helpers from `interpolated_fts_boundary.jl`.

# `interpolate(func, …)` (Oceananigans #5726) returns the blend in func-space and applies no inverse —
# the caller owns it. The carrier wants the *actual* state value (to feed the prognostic map), so
# `_query_source` un-maps with `inverse_transform`: `log`-space interpolation of pressure becomes
# `exp(blend of log p)` (faithful ln-p interpolation); `identity` is a plain interpolation.
@inline inverse_transform(::typeof(identity)) = identity
@inline inverse_transform(::typeof(log))      = exp
@inline inverse_transform(::typeof(log2))     = exp2
@inline inverse_transform(::typeof(log10))    = exp10

# Source query applying interpolation transform `f` per source value, then un-mapping. A
# `FieldTimeSeries` is interpolated in space + time; an `AbstractField` (or GPU-adapted source) in space.
@inline _query_source(fts::FlavorOfFTS, source_grid, X, loc, t, f) =
    inverse_transform(f)(Oceananigans.Fields.interpolate(f, X, Time(t), fts, Oceananigans.instantiated_location(fts), source_grid))

@inline _query_source(source, source_grid, X, loc, t, f) =
    inverse_transform(f)(Oceananigans.Fields.interpolate(f, X, source, loc, source_grid))

# Clamp the query's z-coordinate into the source grid's center-z range, so a child node below (or
# above) the parent's vertical extent returns the parent's edge value (constant extrapolation in z)
# rather than the halo-dependent / linearly-extrapolated value `interpolate` gives outside the grid.
# `validate_source_bracket` intentionally lets the child exceed the parent in z (e.g. ERA5
# pressure-level data doesn't reach the surface); this makes the resulting extrapolation well-defined.
@inline function clamp_to_source_z(X, source_grid)
    Nz = size(source_grid, 3)
    z₁ = znode(1, 1, 1,  source_grid, Center(), Center(), Center())
    zₙ = znode(1, 1, Nz, source_grid, Center(), Center(), Center())
    return (X[1], X[2], clamp(X[3], min(z₁, zₙ), max(z₁, zₙ)))
end

struct ParentStateBoundary{Dim, SideType, LX, LY, LZ, S, G, I, F}
    sources        :: S   # NamedTuple of parent FieldTimeSeries (keyed by physical variable)
    source_grid    :: G   # one shared parent grid (all sources live on it)
    interpolations :: I   # NamedTuple (same keys) of per-source interp transforms (log / identity)
    transform      :: F   # (state::NamedTuple) -> scalar child prognostic
end

# User-facing-ish constructor — pre-regularization, location/side/dim tags are `Nothing`. We capture
# one shared `source_grid` (mirroring `Interpolated`), essential for GPU since adapted FTS drop `.grid`.
function ParentStateBoundary(sources::NamedTuple, interpolations::NamedTuple, transform)
    grid = first(sources).grid
    return ParentStateBoundary{Nothing, Nothing, Nothing, Nothing, Nothing,
                               typeof(sources), typeof(grid), typeof(interpolations), typeof(transform)}(
        sources, grid, interpolations, transform)
end

@inline ParentStateBoundary{Dim, SideType, LX, LY, LZ}(sources::S, source_grid::G, interpolations::I, transform::F) where {Dim, SideType, LX, LY, LZ, S, G, I, F} =
    ParentStateBoundary{Dim, SideType, LX, LY, LZ, S, G, I, F}(sources, source_grid, interpolations, transform)

Adapt.adapt_structure(to, c::ParentStateBoundary{D, S, LX, LY, LZ}) where {D, S, LX, LY, LZ} =
    ParentStateBoundary{D, S, LX, LY, LZ}(map(s -> adapt(to, s), c.sources),
                                          adapt(to, c.source_grid),
                                          c.interpolations,
                                          adapt(to, c.transform))

function regularize_boundary_condition(c::ParentStateBoundary{Nothing}, grid, loc, dim, SideType, args...)
    LX = typeof(loc[1]); LY = typeof(loc[2]); LZ = typeof(loc[3])
    for s in values(c.sources)
        validate_source_bracket(s, grid, LX, LY, LZ)
    end
    return ParentStateBoundary{dim, SideType, LX, LY, LZ}(c.sources, c.source_grid, c.interpolations, c.transform)
end

# Interpolate every source at the boundary-face node `X` (each via its own interpolation transform —
# `log` for fields interpolated in log space) and apply the prognostic `transform` to the resulting
# state. Sources are Center-located parent fields; `X` is the boundary face in the normal direction.
@inline function _parent_state(c::ParentStateBoundary{<:Any, <:Any, LX, LY, LZ}, X, t) where {LX, LY, LZ}
    loc   = (LX(), LY(), LZ())
    Xc    = clamp_to_source_z(X, c.source_grid)
    state = map((src, itp) -> _query_source(src, c.source_grid, Xc, loc, t, itp), c.sources, c.interpolations)
    return c.transform(state)
end

@inline function getbc(bc::ParentStateBoundary{1, S, LX, LY, LZ},
                       j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    i = _boundary_index(S, grid.Nx)
    X = node(i, j, k, grid, Face(), LY(), LZ())
    return _parent_state(bc, X, clock_time(clock))
end

@inline function getbc(bc::ParentStateBoundary{2, S, LX, LY, LZ},
                       i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    j = _boundary_index(S, grid.Ny)
    X = node(i, j, k, grid, LX(), Face(), LZ())
    return _parent_state(bc, X, clock_time(clock))
end

@inline function getbc(bc::ParentStateBoundary{3, S, LX, LY, LZ},
                       i::Integer, j::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    k = _boundary_index(S, grid.Nz)
    X = node(i, j, k, grid, LX(), LY(), Face())
    return _parent_state(bc, X, clock_time(clock))
end

#####
##### ParentStateTarget: the interior-relaxation analogue of `ParentStateBoundary`. A callable
##### `target(x, y, z, t)` for an Oceananigans `Relaxation` forcing that interpolates each parent
##### source at the interior node `(x, y, z)` (each via its `interpolations` transform, un-mapped) and
##### applies the prognostic `transform` — same `_query_source` machinery as the boundary carrier, but
##### at an arbitrary node rather than a boundary face. Used for on-the-fly Davies relaxation.
#####

struct ParentStateTarget{S, G, I, F}
    sources        :: S
    source_grid    :: G
    interpolations :: I
    transform      :: F
end

function ParentStateTarget(sources::NamedTuple, interpolations::NamedTuple, transform)
    grid = first(sources).grid
    return ParentStateTarget(sources, grid, interpolations, transform)
end

@inline function (t::ParentStateTarget)(x, y, z, time)
    Xc = clamp_to_source_z((x, y, z), t.source_grid)
    return t.transform(map((src, itp) -> _query_source(src, t.source_grid, Xc, nothing, time, itp),
                           t.sources, t.interpolations))
end

Adapt.adapt_structure(to, t::ParentStateTarget) =
    ParentStateTarget(map(s -> adapt(to, s), t.sources),
                      adapt(to, t.source_grid),
                      t.interpolations,
                      adapt(to, t.transform))
