module Oceans

export ocean_simulation, SlabOcean

using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Utils: with_tracers, launch!
using Oceananigans.Advection: FluxFormAdvection
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, DiscreteBoundaryFunction
using Oceananigans.Grids: architecture
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, inactive_node, MutableGridOfSomeKind
using Oceananigans.OrthogonalSphericalShellGrids
using Oceananigans.Operators

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    CATKEMixingLength,
    CATKEEquation

using SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState, θᴾ_from_Θ
using KernelAbstractions: @kernel, @index

using NumericalEarth.EarthSystemModels

import NumericalEarth.EarthSystemModels: interpolate_state!,
                                     update_net_fluxes!,
                                     reference_density,
                                     heat_capacity,
                                     exchange_grid,
                                     temperature_units,
                                     DegreesKelvin,
                                     ocean_temperature,
                                     ocean_salinity,
                                     ocean_surface_temperature,
                                     ocean_surface_salinity,
                                     ocean_surface_velocities

import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, net_fluxes

default_gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration
default_planet_rotation_rate = Oceananigans.defaults.planet_rotation_rate

struct Default{V}
    value :: V
end

"""
    default_or_override(default::Default, alternative_default=default.value) = alternative_default
    default_or_override(override, alternative_default) = override

Either return `default.value`, an `alternative_default`, or an `override`.

The purpose of this function is to help define constructors with "configuration-dependent" defaults.
For example, the default bottom drag should be 0 for a single column model, but 0.003 for a global model.
We therefore need a way to specify both the "normal" default 0.003 as well as the "alternative default" 0,
all while respecting user input and changing this to a new value if specified.
"""
default_or_override(default::Default, possibly_alternative_default=default.value) = possibly_alternative_default
default_or_override(override, alternative_default=nothing) = override

include("slab_ocean.jl")
include("barotropic_potential_forcing.jl")
include("radiative_forcing.jl")
include("multiple_surface_fluxes.jl")
include("ocean_simulation.jl")
include("assemble_net_ocean_fluxes.jl")

#####
##### Extend utility functions to grab the state of the ocean
#####

ocean_salinity(ocean::Simulation{<:HydrostaticFreeSurfaceModel})    = ocean.model.tracers.S
ocean_temperature(ocean::Simulation{<:HydrostaticFreeSurfaceModel}) = ocean.model.tracers.T

function ocean_surface_salinity(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    kᴺ = size(ocean.model.grid, 3)
    return view(ocean.model.tracers.S.data, :, :, kᴺ:kᴺ)
end

function ocean_surface_temperature(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    kᴺ = size(ocean.model.grid, 3)
    return view(ocean.model.tracers.T.data, :, :, kᴺ:kᴺ)
end

function ocean_surface_velocities(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    kᴺ = size(ocean.model.grid, 3)
    return view(ocean.model.velocities.u, :, :, kᴺ), view(ocean.model.velocities.v, :, :, kᴺ)
end

@kernel function _ocean_state_to_potential_temperature!(Tᵉˣ, Tᵒᶜ, Sᵒᶜ, kᴺ)
    i, j = @index(Global, NTuple)
    @inbounds Tᵉˣ[i, j, 1] = θᴾ_from_Θ(Sᵒᶜ[i, j, kᴺ], Tᵒᶜ[i, j, kᴺ])
end

function interpolate_state!(exchanger, grid, ocean::Simulation{<:HydrostaticFreeSurfaceModel}, coupled_model)
    Tᵉˣ = exchanger.state.T      # Potential Temperature θᴾ
    Tᵒᶜ = ocean.model.tracers.T  # Conservative Temperature Θ
    Sᵒᶜ = ocean.model.tracers.S  # Absolute Salinity Sᴬ
    kᴺ = size(ocean.model.grid, 3)
    arch = architecture(ocean.model.grid)
    launch!(arch, grid, :xy, _ocean_state_to_potential_temperature!, Tᵉˣ, Tᵒᶜ, Sᵒᶜ, kᴺ)
    return nothing
end

function ComponentExchanger(ocean::Simulation{<:HydrostaticFreeSurfaceModel}, grid)
    ocean_grid = ocean.model.grid

    if ocean_grid == grid
        u = ocean.model.velocities.u
        v = ocean.model.velocities.v
        S = ocean.model.tracers.S
    else
        u = Field{Center, Center, Nothing}(grid)
        v = Field{Center, Center, Nothing}(grid)
        S = Field{Center, Center, Nothing}(grid)
    end

    # T is in potential temperature, model T in conservative temperature
    T = Field{Center, Center, Nothing}(grid)

    return ComponentExchanger((; u, v, T, S), nothing)
end

@inline net_flux(condition) = condition
@inline net_flux(bc::MultipleFluxes) = bc.flux_field
@inline net_flux(bc::DiscreteBoundaryFunction) = net_flux(bc.func)

function net_fluxes(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    # TODO: Generalize this to work with any ocean model
    τˣ = net_flux(ocean.model.velocities.u.boundary_conditions.top.condition)
    τʸ = net_flux(ocean.model.velocities.v.boundary_conditions.top.condition)
    net_ocean_surface_fluxes = (; u=τˣ, v=τʸ)

    tracers = ocean.model.tracers
    ocean_surface_tracer_fluxes = NamedTuple(name => net_flux(tracers[name].boundary_conditions.top.condition) for name in keys(tracers))
    return merge(ocean_surface_tracer_fluxes, net_ocean_surface_fluxes)
end

end # module
