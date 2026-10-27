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
using Oceananigans.Grids: node, ξnode, ηnode, ξnodes, ηnodes, topology, Face, Center,
                          OrthogonalSphericalShellGrid, FullyConnected, LeftConnected, RightConnected
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Field, interior, compute!
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
    on_physical_boundary(topology(grid, dim), SideType) &&
        validate_source_bracket(c.source, grid, dim, SideType, LX, LY, LZ)
    return Interpolated{dim, SideType, LX, LY, LZ}(c.source, c.source_grid)
end

# Every rank regularizes the user's boundary conditions on its own local grid, but a rank whose side
# borders another rank replaces this condition with halo communication — its local edge is interior to
# the domain, and a source spanning only the physical boundary need not bracket it. A `LeftConnected`
# direction borders a rank on its left, a `RightConnected` one on its right.
on_physical_boundary(topology, SideType) = true
on_physical_boundary(::Type{FullyConnected}, SideType) = false
on_physical_boundary(::Type{LeftConnected}, ::Type{LeftBoundary}) = false
on_physical_boundary(::Type{RightConnected}, ::Type{RightBoundary}) = false

# `extrema` needs a real array: a view of an operation is a `SubArray`, not a `Field`, and reduces on
# the host, while reducing the operation as a `Field` would `MPI.Allreduce` across a partitioned grid,
# a collective only the ranks owning this side would reach.
function boundary_coordinate_extent(coordinate, grid, dim, SideType, ::Type{LX}, ::Type{LY}) where {LX, LY}
    operation = KernelFunctionOperation{LX, LY, Center}(coordinate, grid, LX(), LY(), Center())
    # A `Face` extent exceeds the grid's by one only where the axis is bounded, not where it is connected
    Nx, Ny, _ = size(operation)
    i = dim == 1 ? (SideType === LeftBoundary ? (1:1) : (Nx:Nx)) : 1:Nx
    j = dim == 2 ? (SideType === LeftBoundary ? (1:1) : (Ny:Ny)) : 1:Ny
    line = Field(operation; indices = (i, j, 1:1))
    compute!(line)
    return extrema(interior(line))
end

function validate_axis_bracket(label, (src_lo, src_hi), (sim_lo, sim_hi))
    (src_lo ≤ sim_lo && sim_hi ≤ src_hi) || throw(ArgumentError(
        "Interpolated boundary source $(label)-extent [$src_lo, $src_hi] does not " *
        "bracket model grid $(label)-extent [$sim_lo, $sim_hi]"))
    return nothing
end

# The source must bracket the boundary horizontally: along the boundary, the child's whole extent;
# across it, only the boundary itself, which leaves a per-side source free to span only its own side.
function validate_bracket(source, grid, dim, SideType, ::Type{LX}, ::Type{LY}) where {LX, LY}
    source_loc  = Oceananigans.instantiated_location(source)
    source_grid = source.grid

    validate_axis_bracket("x", extrema(ξnodes(source_grid, source_loc...)),
                          boundary_coordinate_extent(ξnode, grid, dim, SideType, LX, LY))

    validate_axis_bracket("y", extrema(ηnodes(source_grid, source_loc...)),
                          boundary_coordinate_extent(ηnode, grid, dim, SideType, LX, LY))

    return nothing
end

# A curvilinear grid's coordinates vary along both axes, so its extent comes from the cell corners.
validate_source_bracket(source, grid::OrthogonalSphericalShellGrid, dim, SideType,
                        ::Type{LX}, ::Type{LY}, ::Type{LZ}) where {LX, LY, LZ} =
    validate_bracket(source, grid, dim, SideType, Face, Face)

# A separable grid's comes from the field's own location, with `Face` across the boundary.
validate_source_bracket(source, grid, dim, SideType,
                        ::Type{LX}, ::Type{LY}, ::Type{LZ}) where {LX, LY, LZ} =
    validate_bracket(source, grid, dim, SideType, dim == 1 ? Face : LX, dim == 2 ? Face : LY)

#####
##### getbc: evaluate the source at the boundary-face node.
#####
#
# Dispatches via `_query_source` so a `FieldTimeSeries` source is interpolated
# in space + time, while an `AbstractField` source is interpolated in space at
# the parent's current state.

# `loc` is passed explicitly because `Adapt` reduces an `AbstractField` source to a bare
# `OffsetArray` on GPU, where `instantiated_location` is unavailable.
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
# Return the boundary-evaluation time as the grid's float type so `getbc` is type-stable regardless of
# the model precision (`clock.time` may be Float32; a bare `0.0` literal would force a Union). The
# `Nothing` branch (no clock — `set!`-time IC halo fills) evaluates the source at t = 0.
@inline clock_time(clock, grid) = convert(eltype(grid), clock.time)
@inline clock_time(::Nothing, grid) = zero(eltype(grid))

@inline function getbc(bc::Interpolated{1, S, LX, LY, LZ},
                       j::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    i = _boundary_index(S, grid.Nx)
    # Sample the source at the boundary FACE in the normal direction (not the field's
    # center node). A Value BC's value is the value AT the boundary, so for a Center
    # field this must be the face, not the half-cell-inside center; for a Face-normal
    # field (the normal velocity) this is unchanged (its location is already Face).
    X = node(i, j, k, grid, Face(), LY(), LZ())
    return _query_source(bc.source, bc.source_grid, X, (LX(), LY(), LZ()), clock_time(clock, grid))
end

@inline function getbc(bc::Interpolated{2, S, LX, LY, LZ},
                       i::Integer, k::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    j = _boundary_index(S, grid.Ny)
    X = node(i, j, k, grid, LX(), Face(), LZ())   # boundary face in the y-normal direction
    return _query_source(bc.source, bc.source_grid, X, (LX(), LY(), LZ()), clock_time(clock, grid))
end

@inline function getbc(bc::Interpolated{3, S, LX, LY, LZ},
                       i::Integer, j::Integer, grid::AbstractGrid, clock=nothing, args...) where {S, LX, LY, LZ}
    k = _boundary_index(S, grid.Nz)
    X = node(i, j, k, grid, LX(), LY(), Face())   # boundary face in the z-normal direction
    return _query_source(bc.source, bc.source_grid, X, (LX(), LY(), LZ()), clock_time(clock, grid))
end
