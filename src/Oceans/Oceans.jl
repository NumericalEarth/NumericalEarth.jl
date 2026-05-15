module Oceans

export ocean_simulation, SlabOcean

import Oceananigans
using Oceananigans.Advection: WENO, WENOVectorInvariant
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, DiscreteBoundaryFunction,
                                       FieldBoundaryConditions, FluxBoundaryCondition
using Oceananigans.BuoyancyFormulations: SeawaterBuoyancy
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Fields: Field, CenterField, interior, set!
using Oceananigans.Grids: Center, Face, RectilinearGrid, xspacings, yspacings, inactive_node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, ImmersedBoundaryCondition
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: SplitExplicitFreeSurface
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids, TripolarGrid
using Oceananigans.Operators: Operators, ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ
using Oceananigans.Simulations: Simulation
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity,
                                                                     CATKEMixingLength,
                                                                     CATKEEquation
using Oceananigans.Units: Units, hours, minutes
using Oceananigans.Utils: Utils, launch!, with_tracers

using Adapt: Adapt, adapt
using Printf: Printf
using SeawaterPolynomials: SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using KernelAbstractions: @kernel, @index

using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels
using NumericalEarth.EarthSystemModels: EarthSystemModels, DegreesKelvin
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger

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
    return interior(ocean.model.tracers.S, :, :, kᴺ:kᴺ)
end

function ocean_surface_temperature(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    kᴺ = size(ocean.model.grid, 3)
    return interior(ocean.model.tracers.T, :, :, kᴺ:kᴺ)
end

function ocean_surface_velocities(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    kᴺ = size(ocean.model.grid, 3)
    return view(ocean.model.velocities.u, :, :, kᴺ), view(ocean.model.velocities.v, :, :, kᴺ)
end

# When using an Oceananigans simulation, we assume that the exchange grid is the ocean grid
# We need, however, to interpolate the surface pressure to the ocean grid
NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, grid, ::Simulation{<:HydrostaticFreeSurfaceModel}, coupled_model) = nothing

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(ocean::Simulation{<:HydrostaticFreeSurfaceModel}, grid)
    ocean_grid = ocean.model.grid

    if ocean_grid == grid
        u = ocean.model.velocities.u
        v = ocean.model.velocities.v
        T = ocean.model.tracers.T
        S = ocean.model.tracers.S
    else
        u = Field{Center, Center, Nothing}(grid)
        v = Field{Center, Center, Nothing}(grid)
        T = Field{Center, Center, Nothing}(grid)
        S = Field{Center, Center, Nothing}(grid)
    end

    return ComponentExchanger((; u, v, T, S), nothing)
end

@inline net_flux(condition) = condition
@inline net_flux(bc::MultipleFluxes) = bc.flux_field
@inline net_flux(bc::DiscreteBoundaryFunction) = net_flux(bc.func)

function NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(ocean::Simulation{<:HydrostaticFreeSurfaceModel})
    # TODO: Generalize this to work with any ocean model
    τˣ = net_flux(ocean.model.velocities.u.boundary_conditions.top.condition)
    τʸ = net_flux(ocean.model.velocities.v.boundary_conditions.top.condition)
    net_ocean_surface_fluxes = (; u=τˣ, v=τʸ)

    tracers = ocean.model.tracers
    ocean_surface_tracer_fluxes = NamedTuple(name => net_flux(tracers[name].boundary_conditions.top.condition) for name in keys(tracers))
    return merge(ocean_surface_tracer_fluxes, net_ocean_surface_fluxes)
end

end # module
