#####
##### parent_boundary_conditions: derive child OpenBoundaryConditions from a parent
#####

# OBC's ContinuousBoundaryFunction calls `f(X..., clock.time)` at each boundary
# node, with the boundary-normal coordinate dropped from `X`. We pre-compute the
# child-grid edge coordinate for that normal direction and close over it; the
# returned function then has the side-appropriate arity:
#
#   west, east:  f(y, z, t)
#   south, north: f(x, z, t)
#   bottom, top:  f(x, y, t)

const _LATERAL_SIDES = (:west, :east, :south, :north, :bottom, :top)

struct ParentBoundaryFunction{P, V, S, E} <: Function
    parent   :: P
    var_name :: V
    side     :: S
    edge     :: E   # boundary-normal coordinate, in physical units
end

@inline (f::ParentBoundaryFunction{P, V, Val{:west}})(y, z, t)  where {P, V} =
    parent_interpolate(parent_field(f.parent, f.var_name), (f.edge, y, z), t)
@inline (f::ParentBoundaryFunction{P, V, Val{:east}})(y, z, t)  where {P, V} =
    parent_interpolate(parent_field(f.parent, f.var_name), (f.edge, y, z), t)
@inline (f::ParentBoundaryFunction{P, V, Val{:south}})(x, z, t) where {P, V} =
    parent_interpolate(parent_field(f.parent, f.var_name), (x, f.edge, z), t)
@inline (f::ParentBoundaryFunction{P, V, Val{:north}})(x, z, t) where {P, V} =
    parent_interpolate(parent_field(f.parent, f.var_name), (x, f.edge, z), t)
@inline (f::ParentBoundaryFunction{P, V, Val{:bottom}})(x, y, t) where {P, V} =
    parent_interpolate(parent_field(f.parent, f.var_name), (x, y, f.edge), t)
@inline (f::ParentBoundaryFunction{P, V, Val{:top}})(x, y, t) where {P, V} =
    parent_interpolate(parent_field(f.parent, f.var_name), (x, y, f.edge), t)

# Edge coordinate of the child grid on each side.
# Uses Face() in the boundary-normal direction so the value sits exactly on the
# domain boundary (Center cells in that direction would be a half-cell inside).
function _edge_coordinate(grid, side::Symbol)
    side === :west   && return xnode(1,                1, 1, grid, Face(),   Center(), Center())
    side === :east   && return xnode(size(grid, 1) + 1, 1, 1, grid, Face(),   Center(), Center())
    side === :south  && return ynode(1, 1,                1, grid, Center(), Face(),   Center())
    side === :north  && return ynode(1, size(grid, 2) + 1, 1, grid, Center(), Face(),   Center())
    side === :bottom && return znode(1, 1, 1,                grid, Center(), Center(), Face())
    side === :top    && return znode(1, 1, size(grid, 3) + 1, grid, Center(), Center(), Face())
    throw(ArgumentError("Unknown side $side; expected one of $(_LATERAL_SIDES)"))
end

"""
    parent_boundary_conditions(parent;
                               variables,
                               sides = (:west, :east, :south, :north),
                               grid,
                               schemes = NamedTuple())

Build a `NamedTuple` of `FieldBoundaryConditions`-shaped specs that drive the
child's open boundaries from the `parent`.

Arguments
=========

- `parent`: any object satisfying the parent interface (e.g. a 3D
  `PrescribedAtmosphere`, or a prognostic `Simulation`).

- `variables`: a `NamedTuple` mapping child field names to parent field names,
  e.g. `(u = :u, v = :v, T = :T)`.

- `sides`: a tuple of `Symbol`s naming which boundaries to drive. Choices are
  `:west, :east, :south, :north, :bottom, :top`.

- `grid`: the **child** grid. Boundary-edge coordinates are read from it.

- `schemes`: optional `NamedTuple` keyed by child field name giving the OBC
  scheme (e.g. `(u = PerturbationAdvection(...),)`). Fields not listed default
  to a stiff Dirichlet open BC (`scheme = nothing`).

Returns
=======

A `NamedTuple` keyed by child field name, each value a `FieldBoundaryConditions`
populated with `OpenBoundaryCondition`s on the requested `sides`. Pass directly
as the `boundary_conditions` kwarg of a model constructor.
"""
function parent_boundary_conditions(parent;
                                    variables,
                                    sides = (:west, :east, :south, :north),
                                    grid,
                                    schemes = NamedTuple())

    field_pairs = []
    for (child_name, parent_name) in pairs(variables)
        scheme = haskey(schemes, child_name) ? getproperty(schemes, child_name) : nothing
        side_pairs = []
        for side in sides
            edge = _edge_coordinate(grid, side)
            f    = ParentBoundaryFunction(parent, parent_name, Val(side), edge)
            push!(side_pairs, side => OpenBoundaryCondition(f; scheme))
        end
        push!(field_pairs, child_name => FieldBoundaryConditions(; side_pairs...))
    end

    return (; field_pairs...)
end
