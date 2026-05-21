#####
##### parent_boundary_conditions: derive Open BCs from a parent's FieldTimeSeries
#####
#
# Oceananigans' `OpenBoundaryCondition(f)` calls `f(X..., clock.time, args...)`
# at each child boundary node, with the boundary-normal coordinate dropped from
# `X` (so a west BC sees `(y, z, t)`, a south BC sees `(x, z, t)`, etc.). The
# side dispatch is handled by the standard `ContinuousBoundaryFunction`
# regularization that Oceananigans runs at model-construction time.
#
# We therefore just need to produce per-side closures that:
#  - know the child-grid boundary-edge coordinate in the normal direction, and
#  - interpolate the parent's `FieldTimeSeries` at `(x, y, z, t)`.
#
# No new BC types or wrappers are needed.

@inline function _fts_interpolate(fts::FieldTimeSeries, X, t)
    return Oceananigans.Fields.interpolate(X, Time(t), fts,
                                           instantiated_location(fts),
                                           fts.grid)
end

# Build the per-side closure. Returns a function with the side-appropriate arity:
# west/east → (y, z, t), south/north → (x, z, t), bottom/top → (x, y, t).
function _parent_boundary_function(grid, side::Symbol, fts::FieldTimeSeries)
    if side === :west
        x_edge = xnode(1, 1, 1, grid, Face(), Center(), Center())
        return (y, z, t) -> _fts_interpolate(fts, (x_edge, y, z), t)
    elseif side === :east
        x_edge = xnode(size(grid, 1) + 1, 1, 1, grid, Face(), Center(), Center())
        return (y, z, t) -> _fts_interpolate(fts, (x_edge, y, z), t)
    elseif side === :south
        y_edge = ynode(1, 1, 1, grid, Center(), Face(), Center())
        return (x, z, t) -> _fts_interpolate(fts, (x, y_edge, z), t)
    elseif side === :north
        y_edge = ynode(1, size(grid, 2) + 1, 1, grid, Center(), Face(), Center())
        return (x, z, t) -> _fts_interpolate(fts, (x, y_edge, z), t)
    elseif side === :bottom
        z_edge = znode(1, 1, 1, grid, Center(), Center(), Face())
        return (x, y, t) -> _fts_interpolate(fts, (x, y, z_edge), t)
    elseif side === :top
        z_edge = znode(1, 1, size(grid, 3) + 1, grid, Center(), Center(), Face())
        return (x, y, t) -> _fts_interpolate(fts, (x, y, z_edge), t)
    else
        throw(ArgumentError("Unknown side $side; expected one of " *
                            "(:west, :east, :south, :north, :bottom, :top)"))
    end
end

"""
    parent_boundary_conditions(grid;
                               variables,
                               sides   = (:west, :east, :south, :north),
                               schemes = NamedTuple())

Build a `NamedTuple` of `FieldBoundaryConditions` driving the child's open
boundaries directly from a parent's `FieldTimeSeries`.

Arguments
=========

- `grid`: the **child** grid. Boundary-edge coordinates are read from it.

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
        side_pairs = []
        for side in sides
            f = _parent_boundary_function(grid, side, fts)
            push!(side_pairs, side => OpenBoundaryCondition(f; scheme))
        end
        push!(field_pairs, child_name => FieldBoundaryConditions(; side_pairs...))
    end

    return (; field_pairs...)
end
