#####
##### Mixing over the ice shelf boundary layer that the melt fluxes are sensed and spread over
#####

using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity, VerticallyImplicitTimeDiscretization

# κ = ν = C u★ H at faces within the boundary layer beneath the ice base
@inline function ice_shelf_boundary_layer_diffusivity(i, j, k, grid, clock, model_fields, p)
    kt = @inbounds p.k_draft[i, j, 1]

    FT = eltype(grid)
    depth = zero(FT)
    for kk in k:max(kt, k)
        depth += Δzᶜᶜᶜ(i, j, kk, grid)
    end

    inside = (kt > 0) & (k <= kt) & (depth < p.boundary_layer_thickness)
    u★ = @inbounds p.friction_velocity[i, j, 1]

    return ifelse(inside, p.mixing_coefficient * u★ * p.boundary_layer_thickness, zero(grid))
end

"""
$(TYPEDSIGNATURES)

Return a `VerticalScalarDiffusivity` with viscosity and diffusivity
`κ = ν = mixing_coefficient * u★ * H` within the ice shelf boundary layer of thickness
`H = interface.properties.boundary_layer_thickness`, and zero elsewhere. `u★` is the
interface friction velocity computed by [`compute_ice_shelf_fluxes!`](@ref).

The melt fluxes are computed from the boundary-layer mean state and spread uniformly per unit
volume across the layer by [`ice_shelf_tracer_forcing`](@ref). This closure keeps that layer
well mixed, so thin partial cells at the ice base cannot decouple from the layer mean and drift
below the freezing point. Add it to a model's closure tuple:

```jldoctest
using Oceananigans
using NumericalEarth

underlying_grid = RectilinearGrid(size = (2, 1, 4), x = (0, 2), y = (0, 1), z = (-1, 0),
                                  topology = (Bounded, Periodic, Bounded))
bottom(x, y) = -1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

interface = IceShelfOceanInterface(grid)
closure = ice_shelf_boundary_layer_closure(interface)

parameters = closure.κ.parameters
(parameters.mixing_coefficient, parameters.boundary_layer_thickness)

# output
(0.1, 30.0)
```
"""
function ice_shelf_boundary_layer_closure(interface::IceShelfOceanInterface; mixing_coefficient = 0.1)
    FT = eltype(interface.k_draft.grid)
    parameters = (k_draft = interface.k_draft,
                  friction_velocity = interface.friction_velocity,
                  boundary_layer_thickness = interface.properties.boundary_layer_thickness,
                  mixing_coefficient = convert(FT, mixing_coefficient))

    return VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization();
                                     κ = ice_shelf_boundary_layer_diffusivity,
                                     ν = ice_shelf_boundary_layer_diffusivity,
                                     discrete_form = true,
                                     loc = (Center, Center, Face),
                                     parameters)
end
