module Oceans

export ocean_simulation, SlabOcean, PrescribedOcean

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Advection: WENO, WENOVectorInvariant
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, DiscreteBoundaryFunction,
                                       FieldBoundaryConditions, FluxBoundaryCondition, getbc
using Oceananigans.BuoyancyFormulations: SeawaterBuoyancy
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Fields: Field, CenterField, set!, interior
using Oceananigans.Forcings: MultipleForcings
using Oceananigans.Grids: inactive_node, Face, Center, xspacings, yspacings, RectilinearGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, ImmersedBoundaryCondition
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: SplitExplicitFreeSurface
using Oceananigans.Models.NonhydrostaticModels: NonhydrostaticModel
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids, TripolarGrid
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.Simulations: Simulation
using Oceananigans.TurbulenceClosures: κzᶜᶜᶠ
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity,
                                                                     CATKEMixingLength,
                                                                     CATKEEquation
using Oceananigans.Units: minutes, hours
using Oceananigans.Utils: with_tracers, launch!
using SeawaterPolynomials: SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

using ..EarthSystemModels: EarthSystemModels,
                           ocean_surface_velocities,
                           ocean_surface_salinity,
                           DegreesKelvin,
                           heat_capacity
using ..EarthSystemModels.InterfaceComputations: ComponentExchanger

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
include("prescribed_ocean.jl")
include("barotropic_potential_forcing.jl")
include("radiative_forcing.jl")
include("multiple_surface_fluxes.jl")
include("ocean_simulation.jl")
include("nonhydrostatic_ocean_simulation.jl")
include("assemble_net_ocean_fluxes.jl")

#####
##### Extend utility functions to grab the state of the ocean
#####

# An ocean simulation is not a sea ice model — used to catch swapped positional
# args in the convenience constructors (e.g. `OceanSeaIceModel`).
EarthSystemModels.is_sea_ice_component(::OceananigansModelSimulations) = false

EarthSystemModels.ocean_salinity(ocean::OceananigansModelSimulations)    = ocean.model.tracers.S
EarthSystemModels.ocean_temperature(ocean::OceananigansModelSimulations) = ocean.model.tracers.T

function EarthSystemModels.ocean_surface_salinity(ocean::OceananigansModelSimulations)
    kᴺ = size(ocean.model.grid, 3)
    return interior(ocean.model.tracers.S, :, :, kᴺ:kᴺ)
end

function EarthSystemModels.ocean_surface_temperature(ocean::OceananigansModelSimulations)
    kᴺ = size(ocean.model.grid, 3)
    return interior(ocean.model.tracers.T, :, :, kᴺ:kᴺ)
end

function EarthSystemModels.ocean_surface_velocities(ocean::OceananigansModelSimulations)
    kᴺ = size(ocean.model.grid, 3)
    return view(ocean.model.velocities.u, :, :, kᴺ), view(ocean.model.velocities.v, :, :, kᴺ)
end

# When using an Oceananigans simulation, we assume that the exchange grid is the ocean grid
# We need, however, to interpolate the surface pressure to the ocean grid
EarthSystemModels.interpolate_state!(exchanger, grid, ::OceananigansModelSimulations, coupled_model) = nothing

function EarthSystemModels.InterfaceComputations.ComponentExchanger(ocean::OceananigansModelSimulations, grid)
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

    # Near-surface vertical tracer diffusivity, evaluated lazily inside the
    # interface flux kernel by formulations that consume it (`InteriorDiffusivity`).
    model = ocean.model
    temperature_index = findfirst(name -> name === :T, collect(keys(model.tracers)))
    κ = KernelFunctionOperation{Center, Center, Nothing}(Σκzᴺ,
                                                         ocean_grid,
                                                         model.closure,
                                                         model.closure_fields,
                                                         Val(temperature_index),
                                                         model.clock,
                                                         Oceananigans.fields(model))

    return ComponentExchanger((; u, v, T, S, κ), nothing)
end

#####
##### Near-surface vertical diffusivity assessment
#####

# Total vertical tracer diffusivity at the surface. Falls back to zero for closures without vertical diffusivity
@inline Σκzᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, model_fields) = κzᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, model_fields)

@inline Σκzᶜᶜᶠ(i, j, k, grid, closure::Tuple{}, K::Tuple{}, id, clock, model_fields) = zero(grid)

@inline Σκzᶜᶜᶠ(i, j, k, grid, closure::Tuple{<:Any}, K::Tuple{<:Any}, id, clock, model_fields) =
    κzᶜᶜᶠ(i, j, k, grid, closure[1], K[1], id, clock, model_fields) 

@inline Σκzᶜᶜᶠ(i, j, k, grid, closure::Tuple{<:Any, <:Any}, K::Tuple{<:Any, <:Any}, id, clock, model_fields) =
    κzᶜᶜᶠ(i, j, k, grid, closure[1], K[1], id, clock, model_fields) +
    κzᶜᶜᶠ(i, j, k, grid, closure[2], K[2], id, clock, model_fields) 

@inline Σκzᶜᶜᶠ(i, j, k, grid, closure::Tuple{<:Any, <:Any, <:Any}, K::Tuple{<:Any, <:Any, <:Any}, id, clock, model_fields) =
    κzᶜᶜᶠ(i, j, k, grid, closure[1], K[1], id, clock, model_fields) +
    κzᶜᶜᶠ(i, j, k, grid, closure[2], K[2], id, clock, model_fields) +
    κzᶜᶜᶠ(i, j, k, grid, closure[3], K[3], id, clock, model_fields) 

@inline Σκzᶜᶜᶠ(i, j, k, grid, closure::Tuple{<:Any, <:Any, <:Any, <:Any}, K::Tuple{<:Any, <:Any, <:Any, <:Any}, id, clock, model_fields) =
    κzᶜᶜᶠ(i, j, k, grid, closure[1], K[1], id, clock, model_fields) +
    κzᶜᶜᶠ(i, j, k, grid, closure[2], K[2], id, clock, model_fields) +
    κzᶜᶜᶠ(i, j, k, grid, closure[3], K[3], id, clock, model_fields) +
    κzᶜᶜᶠ(i, j, k, grid, closure[4], K[4], id, clock, model_fields) 

@inline Σκzᶜᶜᶠ(i, j, k, grid, closure::Tuple, K::Tuple, id, clock, model_fields) =
    κzᶜᶜᶠ(i, j, k, grid,  closure[1],     K[1],     id, clock, model_fields) +
    Σκzᶜᶜᶠ(i, j, k, grid, closure[2:end], K[2:end], id, clock, model_fields)

@inline function Σκzᴺ(i, j, k, grid, closure, K, id, clock, model_fields)
    Nz = size(grid, 3)
    return Σκzᶜᶜᶠ(i, j, Nz, grid, closure, K, id, clock, model_fields)
end

@inline net_flux(condition) = condition
@inline net_flux(bc::MultipleFluxes) = bc.flux_field
@inline net_flux(bc::DiscreteBoundaryFunction) = net_flux(bc.func)

function EarthSystemModels.InterfaceComputations.net_fluxes(ocean::OceananigansModelSimulations)
    # TODO: Generalize this to work with any ocean model
    τˣ = net_flux(ocean.model.velocities.u.boundary_conditions.top.condition)
    τʸ = net_flux(ocean.model.velocities.v.boundary_conditions.top.condition)
    net_ocean_surface_fluxes = (; u=τˣ, v=τʸ)

    tracers = ocean.model.tracers
    ocean_surface_tracer_fluxes = NamedTuple(name => net_flux(tracers[name].boundary_conditions.top.condition) for name in keys(tracers))
    return merge(ocean_surface_tracer_fluxes, net_ocean_surface_fluxes)
end

end # module
