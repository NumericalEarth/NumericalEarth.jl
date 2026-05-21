#####
##### Interpolated: a user-facing `OpenBoundaryCondition` wrapper around a
##### `FieldTimeSeries`. Mirrors `Oceananigans.Forcings.FieldTimeSeriesTarget`
##### (the FTS-`Relaxation` machinery) but for boundary conditions.
#####
#
# Usage: `OpenBoundaryCondition(Interpolated(fts))`.
#
# The user-facing constructor leaves the dim/side/location type parameters as
# `Nothing`; Oceananigans' standard regularization runs through this module's
# `regularize_boundary_condition(::Interpolated{Nothing}, …)` method and
# returns an `Interpolated{Dim, Side, LX, LY, LZ, FTS}` that carries enough
# type info for `getbc` to compute the right boundary-face node.
#
# This whole file collapses to a deprecation shim once an analogous native
# path lands upstream in Oceananigans (parallel to PR #5575 for `Relaxation`).

import Oceananigans.BoundaryConditions: regularize_boundary_condition, getbc,
                                        LeftBoundary, RightBoundary
import Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Grids: node
using Adapt: Adapt, adapt

"""
    Interpolated(fts)

Wrap a `FieldTimeSeries` for use as the condition value in an
`OpenBoundaryCondition`. During model construction, Oceananigans' standard
boundary-condition regularization tags this with the boundary's
dimension / side / field location, after which `getbc` interpolates the FTS
at the appropriate boundary-face node and clock time.

`OpenBoundaryCondition(Interpolated(fts); scheme = …)` works with any scheme
that consults `getbc` for its exterior value (e.g. `PerturbationAdvection`).
"""
struct Interpolated{Dim, SideType, LX, LY, LZ, FTS}
    fts :: FTS
end

# User-facing constructor — pre-regularization, all tags are `Nothing`.
Interpolated(fts::FTS) where FTS<:FlavorOfFTS =
    Interpolated{Nothing, Nothing, Nothing, Nothing, Nothing, FTS}(fts)

# Type-stable post-regularization constructor.
@inline Interpolated{Dim, SideType, LX, LY, LZ}(fts::FTS) where {Dim, SideType, LX, LY, LZ, FTS} =
    Interpolated{Dim, SideType, LX, LY, LZ, FTS}(fts)

Adapt.adapt_structure(to, w::Interpolated{D, S, LX, LY, LZ}) where {D, S, LX, LY, LZ} =
    Interpolated{D, S, LX, LY, LZ}(adapt(to, w.fts))

#####
##### Regularization: convert the location-less wrapper to the tagged form.
#####

function regularize_boundary_condition(c::Interpolated{Nothing}, grid, loc, dim, SideType, args...)
    LX = typeof(loc[1])
    LY = typeof(loc[2])
    LZ = typeof(loc[3])
    _validate_fts_bracket(c.fts, grid, LX, LY, LZ)
    return Interpolated{dim, SideType, LX, LY, LZ, typeof(c.fts)}(c.fts)
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
            "Interpolated boundary $(label)-extent [$fts_lo, $fts_hi] does not " *
            "bracket model grid $(label)-extent [$sim_lo, $sim_hi]"))
    end
    return nothing
end

#####
##### getbc: interpolate the FTS at the boundary-face node and clock time.
#####

@inline _boundary_index(::Type{LeftBoundary},  N) = 1
@inline _boundary_index(::Type{RightBoundary}, N) = N + 1

@inline function getbc(bc::Interpolated{1, S, LX, LY, LZ},
                       j::Integer, k::Integer, grid::AbstractGrid, clock, args...) where {S, LX, LY, LZ}
    i = _boundary_index(S, grid.Nx)
    X = node(i, j, k, grid, LX(), LY(), LZ())
    return Oceananigans.Fields.interpolate(X, Time(clock.time), bc.fts,
                                           Oceananigans.Fields.instantiated_location(bc.fts),
                                           bc.fts.grid)
end

@inline function getbc(bc::Interpolated{2, S, LX, LY, LZ},
                       i::Integer, k::Integer, grid::AbstractGrid, clock, args...) where {S, LX, LY, LZ}
    j = _boundary_index(S, grid.Ny)
    X = node(i, j, k, grid, LX(), LY(), LZ())
    return Oceananigans.Fields.interpolate(X, Time(clock.time), bc.fts,
                                           Oceananigans.Fields.instantiated_location(bc.fts),
                                           bc.fts.grid)
end

@inline function getbc(bc::Interpolated{3, S, LX, LY, LZ},
                       i::Integer, j::Integer, grid::AbstractGrid, clock, args...) where {S, LX, LY, LZ}
    k = _boundary_index(S, grid.Nz)
    X = node(i, j, k, grid, LX(), LY(), LZ())
    return Oceananigans.Fields.interpolate(X, Time(clock.time), bc.fts,
                                           Oceananigans.Fields.instantiated_location(bc.fts),
                                           bc.fts.grid)
end
