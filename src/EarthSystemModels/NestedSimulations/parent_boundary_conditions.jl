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
  constructor to use for that field. Defaults to `NormalFlowBoundaryCondition` for
  Face-located prognostics. For cell-centered scalars (e.g. ρ, ρθ in a
  compressible LAM), pass `ValueBoundaryCondition` to avoid the
  `NormalFlowBoundaryCondition`-on-Center asymmetric-halo behavior.
"""
function parent_boundary_conditions(grid;
                                    variables,
                                    sides    = (:west, :east, :south, :north),
                                    schemes  = NamedTuple(),
                                    bc_types = NamedTuple())

    field_pairs = []
    for (child_name, fts) in pairs(variables)
        BCType = haskey(bc_types, child_name) ? getproperty(bc_types, child_name) : NormalFlowBoundaryCondition
        condition = Interpolated(fts)
        if BCType === NormalFlowBoundaryCondition
            scheme = haskey(schemes, child_name) ? getproperty(schemes, child_name) : nothing
            side_pairs = [side => NormalFlowBoundaryCondition(condition; scheme) for side in sides]
        else
            haskey(schemes, child_name) && throw(ArgumentError(
                "`schemes` entry provided for $(child_name) but `bc_types[$(child_name)] = $BCType` " *
                "is not `NormalFlowBoundaryCondition`. Schemes apply only to `NormalFlowBoundaryCondition`."))
            side_pairs = [side => BCType(condition) for side in sides]
        end
        push!(field_pairs, child_name => FieldBoundaryConditions(; side_pairs...))
    end

    return (; field_pairs...)
end
