#####
##### InterpolatedFTSBoundary: a BoundaryCondition condition wrapping a
##### `FieldTimeSeries` that interpolates onto the child boundary face.
#####
#
# Mirrors `Oceananigans.Forcings.FieldTimeSeriesTarget` but for boundary
# conditions: lets the user write `OpenBoundaryCondition(fts)` and have
# Oceananigans regularize the FTS into a side-tagged condition with `getbc`
# methods that handle space/time interpolation directly.
#
# Type parameters:
#   Dim          :: 1, 2, 3 — the boundary-normal dimension.
#   SideType     :: LeftBoundary or RightBoundary.
#   LX, LY, LZ   :: child field locations (Center, Face, Nothing) at the
#                   boundary face — needed to compute boundary-node coords.
#   FTS          :: the parent `FieldTimeSeries` (or GPU-adapted equivalent).
#
# Implementation lives here (in NestedSimulations) until an analogous
# `FieldTimeSeries`-aware `OpenBoundaryCondition` lands upstream in
# Oceananigans, after which this file collapses to a deprecation shim.

import Oceananigans.BoundaryConditions: regularize_boundary_condition, getbc,
                                        LeftBoundary, RightBoundary
import Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Grids: node
using Adapt: Adapt, adapt

struct InterpolatedFTSBoundary{Dim, SideType, LX, LY, LZ, FTS}
    fts :: FTS
end

# Type-stable constructor — all the tags live in the type parameters so the
# `getbc` dispatch picks the right boundary-node formula without runtime checks.
@inline InterpolatedFTSBoundary{Dim, SideType, LX, LY, LZ}(fts::FTS) where {Dim, SideType, LX, LY, LZ, FTS} =
    InterpolatedFTSBoundary{Dim, SideType, LX, LY, LZ, FTS}(fts)

Adapt.adapt_structure(to, b::InterpolatedFTSBoundary{D, S, LX, LY, LZ}) where {D, S, LX, LY, LZ} =
    InterpolatedFTSBoundary{D, S, LX, LY, LZ}(adapt(to, b.fts))

#####
##### Regularization: convert `OpenBoundaryCondition(fts)` into an
##### `InterpolatedFTSBoundary` tagged with the boundary's dim/side/location.
#####
#
# The side-specific `regularize_{west,east,...}_boundary_condition` chain
# forwards to `regularize_boundary_condition(condition, grid, loc, dim, Side, prognostic_names)`.
# We dispatch on `condition::FlavorOfFTS` here.

function regularize_boundary_condition(fts::FlavorOfFTS, grid, loc, dim, SideType, args...)
    LX = typeof(loc[1])
    LY = typeof(loc[2])
    LZ = typeof(loc[3])
    _validate_fts_bracket(fts, grid, LX, LY, LZ)
    return InterpolatedFTSBoundary{dim, SideType, LX, LY, LZ, typeof(fts)}(fts)
end

# Match the strict "FTS must bracket every child sampling node" check that
# Oceananigans uses for `Relaxation`-on-FTS.
function _validate_fts_bracket(fts, grid, ::Type{LX}, ::Type{LY}, ::Type{LZ}) where {LX, LY, LZ}
    sim_loc = (LX(), LY(), LZ())
    fts_loc = Oceananigans.Fields.instantiated_location(fts)
    for (label, nodes_fn) in (("x", Oceananigans.Grids.xnodes),
                              ("y", Oceananigans.Grids.ynodes),
                              ("z", Oceananigans.Grids.znodes))
        sim_lo, sim_hi = extrema(nodes_fn(grid, sim_loc...))
        fts_lo, fts_hi = extrema(nodes_fn(fts.grid, fts_loc...))
        (fts_lo ≤ sim_lo && sim_hi ≤ fts_hi) || throw(ArgumentError(
            "FieldTimeSeries boundary-condition $(label)-extent [$fts_lo, $fts_hi] " *
            "does not bracket model grid $(label)-extent [$sim_lo, $sim_hi]"))
    end
    return nothing
end

#####
##### getbc: interpolate the FTS at the boundary face node and clock time.
#####
#
# For an X-boundary (Dim = 1, condition shape sampled at `i = 1` for west,
# `i = grid.Nx + 1` for east), the boundary call signature passes (j, k); we
# build the full node (x_face, y, z) via `Oceananigans.Grids.node`. Similarly
# for Y- (Dim = 2) and Z-boundaries (Dim = 3).

@inline _boundary_index(::Type{LeftBoundary},  N) = 1
@inline _boundary_index(::Type{RightBoundary}, N) = N + 1

@inline function getbc(bc::InterpolatedFTSBoundary{1, S, LX, LY, LZ},
                       j::Integer, k::Integer, grid::AbstractGrid, clock, args...) where {S, LX, LY, LZ}
    i = _boundary_index(S, grid.Nx)
    X = node(i, j, k, grid, LX(), LY(), LZ())
    return Oceananigans.Fields.interpolate(X, Time(clock.time), bc.fts,
                                           Oceananigans.Fields.instantiated_location(bc.fts),
                                           bc.fts.grid)
end

@inline function getbc(bc::InterpolatedFTSBoundary{2, S, LX, LY, LZ},
                       i::Integer, k::Integer, grid::AbstractGrid, clock, args...) where {S, LX, LY, LZ}
    j = _boundary_index(S, grid.Ny)
    X = node(i, j, k, grid, LX(), LY(), LZ())
    return Oceananigans.Fields.interpolate(X, Time(clock.time), bc.fts,
                                           Oceananigans.Fields.instantiated_location(bc.fts),
                                           bc.fts.grid)
end

@inline function getbc(bc::InterpolatedFTSBoundary{3, S, LX, LY, LZ},
                       i::Integer, j::Integer, grid::AbstractGrid, clock, args...) where {S, LX, LY, LZ}
    k = _boundary_index(S, grid.Nz)
    X = node(i, j, k, grid, LX(), LY(), LZ())
    return Oceananigans.Fields.interpolate(X, Time(clock.time), bc.fts,
                                           Oceananigans.Fields.instantiated_location(bc.fts),
                                           bc.fts.grid)
end
