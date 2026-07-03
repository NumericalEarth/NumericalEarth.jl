#####
##### parent_boundary_conditions: derive lateral BCs from a parent's FieldTimeSeries
#####
#
# Each side BC wraps the parent FTS in `Interpolated` (interpolated_fts_boundary.jl).
# Oceananigans regularization tags the wrapper with the boundary's dim/side/loc;
# afterward `getbc` interpolates the FTS at the boundary-face node. No closures,
# no edge-coord bookkeeping, no Val(side) dispatch in user code.
#
# The BC kind defaults to `NormalFlowBoundaryCondition` per child variable; pass
# `bc_types = (name = ValueBoundaryCondition, ...)` to override for cell-centered
# scalars where NormalFlowBC writes asymmetrically into the interior.

"""
    parent_boundary_conditions(grid;
                               variables,
                               sides    = (:west, :east, :south, :north),
                               schemes  = NamedTuple(),
                               bc_types = NamedTuple())

Build a `NamedTuple` of `FieldBoundaryConditions` driving the child's lateral
boundaries directly from a parent's `FieldTimeSeries`.

Arguments
=========

- `grid`: the **child** grid. (Passed for symmetry with the rest of the API;
  not used directly — Oceananigans picks up the boundary node coordinates
  during regularization.)

- `variables`: a `NamedTuple` mapping child field names to the parent
  `FieldTimeSeries` that should drive them, e.g.
  `(u = parent.velocities.u, v = parent.velocities.v)`.

- `sides`: a tuple of `Symbol`s naming which boundaries to drive. Choices are
  `:west, :east, :south, :north, :bottom, :top`.

- `schemes`: optional `NamedTuple` keyed by child field name giving the
  `NormalFlowBoundaryCondition` scheme (e.g. a `PerturbationAdvection`). Fields not
  listed default to `scheme = nothing` (stiff Dirichlet). Only consulted when
  the BC type for that field is `NormalFlowBoundaryCondition`.

- `bc_types`: optional `NamedTuple` keyed by child field name giving the BC
  constructor to use for that field. Each entry is either a single BC constructor
  applied to every side, or a per-side `NamedTuple` (keyed by `:west/:east/:south/:north`)
  selecting the BC type per side. Defaults to `NormalFlowBoundaryCondition`. For
  cell-centered scalars (e.g. ρ, ρθ in a compressible LAM), pass `ValueBoundaryCondition`
  to avoid the `NormalFlowBoundaryCondition`-on-Center asymmetric-halo behavior. For a
  momentum component, use per-side types — `NormalFlowBoundaryCondition` on the
  wall-normal side and `ValueBoundaryCondition` on the tangential side — so the
  tangential momentum is set (Dirichlet) rather than left to the normal-flow halo fill.

- `schemes`: applies only to sides whose BC type is `NormalFlowBoundaryCondition`.
"""
function parent_boundary_conditions(grid;
                                    variables,
                                    sides    = (:west, :east, :south, :north),
                                    schemes  = NamedTuple(),
                                    bc_types = NamedTuple())

    field_pairs = []
    for (child_name, fts) in pairs(variables)
        spec      = haskey(bc_types, child_name) ? getproperty(bc_types, child_name) : NormalFlowBoundaryCondition
        scheme    = haskey(schemes, child_name) ? getproperty(schemes, child_name) : nothing
        condition = Interpolated(fts)

        # `spec` is either one BC constructor for all sides or a per-side NamedTuple; `scheme`
        # (e.g. PerturbationAdvection) is consulted only where the chosen type is NormalFlow.
        bc_type(side) = spec isa NamedTuple ? getproperty(spec, side) : spec
        bc_at(side)   = bc_type(side) === NormalFlowBoundaryCondition ?
                            NormalFlowBoundaryCondition(condition; scheme) : bc_type(side)(condition)

        # A `scheme` only takes effect on NormalFlow sides; requesting one for a field that is
        # NormalFlow on no side means it would be silently ignored — flag that as a user error.
        if haskey(schemes, child_name) && !any(side -> bc_type(side) === NormalFlowBoundaryCondition, sides)
            throw(ArgumentError("`schemes` was given for :$child_name, but none of its sides use a " *
                                "NormalFlowBoundaryCondition, so the scheme would be ignored."))
        end

        side_pairs = [side => bc_at(side) for side in sides]
        push!(field_pairs, child_name => FieldBoundaryConditions(; side_pairs...))
    end

    return (; field_pairs...)
end
