#####
##### Immersed-top boundary conditions applying the ice shelf fluxes
#####

using DocStringExtensions: TYPEDSIGNATURES

"""
$(TYPEDSIGNATURES)

Return a `NamedTuple` of `FieldBoundaryConditions` for `u`, `v`, `T`, and `S`
that applies quadratic drag at the immersed top boundary (the ice base). `T`
and `S` are passed through unchanged aside from any boundary conditions given
via the keyword arguments below -- the ice shelf melt/heat/salt fluxes are
applied as an explicit tracer tendency by [`ice_shelf_tracer_forcing`](@ref)
instead of an immersed top flux, so that the tendency is spread over the
boundary layer beneath the ice base rather than concentrated in the topmost
cell, which can be much thinner than a nominal cell in a `PartialCellCavity`
column.

The velocity conditions reuse the ocean's immersed quadratic drag with
`drag_coefficient` (default ``C_d = 0.0015``). The drag coefficient should normally match the one inside the interface's
`VelocityBasedFrictionVelocity`.

`bottom_drag_coefficient` adds the same quadratic drag on the immersed bottom
(the seabed) when given a number.

`implicit_drag = true` applies the drag as an implicit flux, which requires a
vertically implicit closure; `implicit_drag = false` applies it explicitly.

The `u`, `v`, `T`, `S` keyword arguments are `NamedTuple`s of additional
non-immersed boundary conditions (e.g. `(; north = ValueBoundaryCondition(0))`)
splatted into each field's `FieldBoundaryConditions`. Pass the result straight
to the model constructor as `boundary_conditions`:

```jldoctest
using Oceananigans
using NumericalEarth

underlying_grid = RectilinearGrid(size = (2, 1, 4), x = (0, 2), y = (0, 1), z = (-1, 0),
                                  topology = (Bounded, Periodic, Bounded))
bottom(x, y)  = -1
ceiling(x, y) = x < 1 ? -0.5 : 0.0
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(bottom, ceiling))

interface = IceShelfOceanInterface(grid)
boundary_conditions = ice_shelf_boundary_conditions(interface)

keys(boundary_conditions)

# output
(:T, :S, :u, :v)
```
"""
function ice_shelf_boundary_conditions(interface::IceShelfOceanInterface;
                                       drag_coefficient = 0.0015,
                                       bottom_drag_coefficient = nothing,
                                       implicit_drag = true,
                                       u = NamedTuple(), v = NamedTuple(),
                                       T = NamedTuple(), S = NamedTuple())

    FT = eltype(interface.fluxes.melt_rate)
    top_drag = (μ = convert(FT, drag_coefficient), ub = zero(FT))

    u_top = bottom_drag_bc(u_immersed_drag_coefficient, u_immersed_bottom_drag, top_drag, implicit_drag)
    v_top = bottom_drag_bc(v_immersed_drag_coefficient, v_immersed_bottom_drag, top_drag, implicit_drag)

    u_immersed = immersed_drag_condition(u_top, bottom_drag_coefficient, u_immersed_drag_coefficient, u_immersed_bottom_drag, FT, implicit_drag)
    v_immersed = immersed_drag_condition(v_top, bottom_drag_coefficient, v_immersed_drag_coefficient, v_immersed_bottom_drag, FT, implicit_drag)

    return (T = FieldBoundaryConditions(; T...),
            S = FieldBoundaryConditions(; S...),
            u = FieldBoundaryConditions(; u..., immersed = u_immersed),
            v = FieldBoundaryConditions(; v..., immersed = v_immersed))
end

immersed_drag_condition(top, ::Nothing, λ, Fₑ, FT, implicit) = ImmersedBoundaryCondition(; top)

function immersed_drag_condition(top, bottom_drag_coefficient, λ, Fₑ, FT, implicit)
    bottom_drag = (μ = convert(FT, bottom_drag_coefficient), ub = zero(FT))
    bottom = bottom_drag_bc(λ, Fₑ, bottom_drag, implicit)
    return ImmersedBoundaryCondition(; top, bottom)
end
