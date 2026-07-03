#####
##### Interpolated: a user-facing `NormalFlowBoundaryCondition` wrapper around a
##### parent state source (a `FieldTimeSeries` or an `AbstractField`).
##### Mirrors `Oceananigans.Forcings.FieldTimeSeriesTarget` (the FTS-`Relaxation`
##### machinery) but for boundary conditions.
#####
#
# Usage: `NormalFlowBoundaryCondition(Interpolated(source))` where `source` is
# either a `FieldTimeSeries` (prescribed parents — interpolates in space + time)
# or an `AbstractField` (prognostic parents — interpolates in space; the
# parent's `time_step!` advances the field state).
#
# The user-facing constructor leaves the dim/side/location type parameters as
# `Nothing`; Oceananigans' standard regularization runs through this module's
# `regularize_boundary_condition(::Interpolated{Nothing}, …)` method and
# returns a fully-tagged `Interpolated{Dim, Side, LX, LY, LZ, S}` that carries
# enough type info for `getbc` to compute the right boundary-face node.
#
# This whole file collapses to a deprecation shim once an analogous native
# path lands upstream in Oceananigans (parallel to PR #5575 for `Relaxation`).

import Oceananigans.BoundaryConditions: regularize_boundary_condition, getbc,
                                        LeftBoundary, RightBoundary
import Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Grids: node, Face
using Adapt: Adapt, adapt

const InterpolatedSource = Union{FlavorOfFTS, Oceananigans.Fields.AbstractField}

"""
    Interpolated(source)

Wrap a parent state source for use as the condition value in an
`NormalFlowBoundaryCondition`. `source` can be:

- a `FieldTimeSeries` (prescribed parent — interpolated in space + time), or
- an `AbstractField` (prognostic parent — interpolated in space at the
  parent's current state, which advances via `time_step!`).

During model construction Oceananigans' standard boundary-condition
regularization tags this with the boundary's dimension / side / field
location; afterward `getbc` evaluates the source at the appropriate
boundary-face node.

`NormalFlowBoundaryCondition(Interpolated(source); scheme = …)` works with any
scheme that consults `getbc` for its exterior value (e.g.
`PerturbationAdvection`).
"""
struct Interpolated{Dim, SideType, LX, LY, LZ, S, G}
    source       :: S
    source_grid  :: G
end

# User-facing constructor — pre-regularization, all tags are `Nothing`.
# We capture `source.grid` at construction time and carry it as a separate
# field, mirroring Oceananigans' `FieldTimeSeriesTarget`: this is essential
# for GPU compilation because `GPUAdaptedFieldTimeSeries` drops the `.grid`
# field during Adapt, and `_query_source` (below) needs the grid to evaluate
# the interpolation.
Interpolated(source::S) where S<:InterpolatedSource =
    Interpolated{Nothing, Nothing, Nothing, Nothing, Nothing, S, typeof(source.grid)}(source, source.grid)

# Type-stable post-regularization constructor.
@inline Interpolated{Dim, SideType, LX, LY, LZ}(source::S, source_grid::G) where {Dim, SideType, LX, LY, LZ, S, G} =
    Interpolated{Dim, SideType, LX, LY, LZ, S, G}(source, source_grid)

Adapt.adapt_structure(to, w::Interpolated{D, S, LX, LY, LZ}) where {D, S, LX, LY, LZ} =
    Interpolated{D, S, LX, LY, LZ}(adapt(to, w.source), adapt(to, w.source_grid))

#####
##### Regularization: convert the location-less wrapper to the tagged form.
#####

function regularize_boundary_condition(c::Interpolated{Nothing}, grid, loc, dim, SideType, args...)
    LX = typeof(loc[1])
    LY = typeof(loc[2])
    LZ = typeof(loc[3])
    validate_source_bracket(c.source, grid, LX, LY, LZ)
    return Interpolated{dim, SideType, LX, LY, LZ, typeof(c.source), typeof(c.source_grid)}(c.source, c.source_grid)
end

# The source must bracket the child *horizontally* (a too-small parent region there is a real
# error). The vertical is intentionally NOT required to bracket: a child legitimately extends below
# the parent's lowest level (e.g. ERA5 pressure-level data doesn't reach the surface), and the
# vertical interpolation clamps to the parent's edge value there rather than extrapolating. (The
# horizontal check mirrors the one Oceananigans uses for `Relaxation`-on-FTS.)
function validate_source_bracket(source, grid, ::Type{LX}, ::Type{LY}, ::Type{LZ}) where {LX, LY, LZ}
    sim_loc    = (LX(), LY(), LZ())
    source_loc = Oceananigans.instantiated_location(source)
    source_grid = source.grid
    for (label, nodes_fn) in (("x", Oceananigans.Grids.xnodes),
                              ("y", Oceananigans.Grids.ynodes))
        sim_lo, sim_hi = extrema(nodes_fn(grid, sim_loc...))
        src_lo, src_hi = extrema(nodes_fn(source_grid, source_loc...))
        (src_lo ≤ sim_lo && sim_hi ≤ src_hi) || throw(ArgumentError(
            "Interpolated boundary source $(label)-extent [$src_lo, $src_hi] does not " *
            "bracket model grid $(label)-extent [$sim_lo, $sim_hi]"))
    end
    return nothing
end

#####
##### getbc: evaluate the source at the boundary-face node.
#####
#
# Dispatches via `_query_source` so a `FieldTimeSeries` source is interpolated
# in space + time, while an `AbstractField` source is interpolated in space at
# the parent's current state.

# `loc` is the regularized boundary location passed in from `getbc`; for these
# same-variable nesting BCs it equals the source field's location. Passing it
# explicitly is essential on GPU: `Adapt` reduces an `AbstractField` source to its bare
# data array (an `OffsetArray`, no longer `<:AbstractField` and with no
# `instantiated_location`). So the prognostic-parent method must be generically typed —
# otherwise `_query_source(::AbstractField, …)` fails to match the adapted source in the
# halo-fill kernel (dynamic dispatch + device allocation → `InvalidIRError`) — and it
# must NOT call `instantiated_location(source)` in-kernel. The FTS source survives
# `Adapt` as a `FlavorOfFTS`, so its method is unchanged.
@inline _query_source(fts::FlavorOfFTS, source_grid, X, loc, t) =
    Oceananigans.Fields.interpolate(X, Time(t), fts,
                                    Oceananigans.instantiated_location(fts),
                                    source_grid)

@inline _query_source(source, source_grid, X, loc, t) =
    Oceananigans.Fields.interpolate(X, source, loc, source_grid)

@inline _boundary_index(::Type{LeftBoundary},  N) = 1
@inline _boundary_index(::Type{RightBoundary}, N) = N + 1

# `fill_halo_regions!` is sometimes invoked without a clock — most notably
# during `set!`-time IC setup, where the kernel is launched with an empty
# `args` tuple. Default `time = 0` in that case so dispatch succeeds.
@inline clock_time(clock) = clock.time
@inline clock_time(::Nothing) = 0.0

@inline function getbc(bc::Interpolated{1, S, LX, LY, LZ},
                       j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    i = _boundary_index(S, grid.Nx)
    # Sample the source at the boundary FACE in the normal direction (not the field's
    # center node). A Value BC's value is the value AT the boundary, so for a Center
    # field this must be the face, not the half-cell-inside center; for a Face-normal
    # field (the normal velocity) this is unchanged (its location is already Face).
    X = node(i, j, k, grid, Face(), LY(), LZ())
    return _query_source(bc.source, bc.source_grid, X, (LX(), LY(), LZ()), clock_time(clock))
end

@inline function getbc(bc::Interpolated{2, S, LX, LY, LZ},
                       i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    j = _boundary_index(S, grid.Ny)
    X = node(i, j, k, grid, LX(), Face(), LZ())   # boundary face in the y-normal direction
    return _query_source(bc.source, bc.source_grid, X, (LX(), LY(), LZ()), clock_time(clock))
end

@inline function getbc(bc::Interpolated{3, S, LX, LY, LZ},
                       i::Integer, j::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    k = _boundary_index(S, grid.Nz)
    X = node(i, j, k, grid, LX(), LY(), Face())   # boundary face in the z-normal direction
    return _query_source(bc.source, bc.source_grid, X, (LX(), LY(), LZ()), clock_time(clock))
end
