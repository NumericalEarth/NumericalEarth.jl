#####
##### parent_boundary_conditions: derive Open BCs from a parent's FieldTimeSeries
#####
#
# `OpenBoundaryCondition(fts)` is regularized by Oceananigans into an
# `InterpolatedFTSBoundary` (defined in interpolated_fts_boundary.jl) — a
# side-tagged condition whose `getbc` interpolates the FTS at the boundary
# face. No closures, no edge-coord bookkeeping, no Val(side) dispatch in
# user code.

"""
    parent_boundary_conditions(grid;
                               variables,
                               sides   = (:west, :east, :south, :north),
                               schemes = NamedTuple())

Build a `NamedTuple` of `FieldBoundaryConditions` driving the child's open
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
  `OpenBoundaryCondition` scheme (e.g. a `PerturbationAdvection`). Fields not
  listed default to `scheme = nothing` (stiff Dirichlet).
"""
function parent_boundary_conditions(grid;
                                    variables,
                                    sides   = (:west, :east, :south, :north),
                                    schemes = NamedTuple())

    field_pairs = []
    for (child_name, fts) in pairs(variables)
        scheme = haskey(schemes, child_name) ? getproperty(schemes, child_name) : nothing
        side_pairs = [side => OpenBoundaryCondition(Interpolated(fts); scheme) for side in sides]
        push!(field_pairs, child_name => FieldBoundaryConditions(; side_pairs...))
    end

    return (; field_pairs...)
end
