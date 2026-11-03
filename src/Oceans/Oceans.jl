module Oceans

export ocean_simulation, ocean_model, river_mouth_vertical_diffusivity, SlabOcean, PrescribedOcean,
       TwoColorRadiation, ChlorophyllOptics, absorption_coefficient, equivalent_chlorophyll

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Advection: WENO, WENOVectorInvariant
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, DiscreteBoundaryFunction,
                                       FieldBoundaryConditions, FluxBoundaryCondition,
                                       IMEXFluxBoundaryCondition, IMEXFlux, getbc
using Oceananigans.BuoyancyFormulations: BuoyancyForce, SeawaterBuoyancy
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Fields: Field, CenterField, compute!, set!, interior
using Oceananigans.Forcings: MultipleForcings, DiscreteForcing
using Oceananigans.Grids: Grids, inactive_node, Face, Center, xspacings, yspacings, znodes, RectilinearGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, ImmersedBoundaryCondition, MutableGridOfSomeKind
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: SplitExplicitFreeSurface
using Oceananigans.Models.NonhydrostaticModels: NonhydrostaticModel
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids, TripolarGrid
using Oceananigans.Operators: active_weighted_ℑxyᶠᶜᶜ, active_weighted_ℑxyᶜᶠᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.Simulations: Simulation
using Oceananigans.TimeSteppers: Clock
using Oceananigans.TurbulenceClosures: κzᶜᶜᶠ, VerticalScalarDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity,
                                                                     CATKEMixingLength,
                                                                     CATKEEquation
using Oceananigans.Units: minutes, hours
using Oceananigans.Utils: with_tracers, launch!
using SeawaterPolynomials: SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState, θ_from_Θ

using ..EarthSystemModels: EarthSystemModels,
                           ocean_surface_velocities,
                           ocean_surface_salinity,
                           DegreesKelvin,
                           default_stop_time,
                           heat_capacity
using ..EarthSystemModels.InterfaceComputations: InterfaceComputations, ComponentExchanger, state2dindex, interface_kernel_parameters

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
include("surface_layer_velocities.jl")
include("assemble_net_ocean_fluxes.jl")

#####
##### Extend utility functions to grab the state of the ocean
#####

# The interface is implemented on the ocean models; a `Simulation` couples like its model, and so does a
# `NestedModel` whose child is an ocean model (see `NestedModels/nested_component.jl`).

# An ocean is not a sea ice model — used to catch swapped positional
# args in the convenience constructors (e.g. `OceanSeaIceModel`).
EarthSystemModels.is_sea_ice_component(::OceananigansOceanModels) = false

EarthSystemModels.ocean_salinity(model::OceananigansOceanModels)    = model.tracers.S
EarthSystemModels.ocean_temperature(model::OceananigansOceanModels) = model.tracers.T

function EarthSystemModels.ocean_surface_salinity(model::OceananigansOceanModels)
    kᴺ = size(model.grid, 3)
    return view(model.tracers.S.data, :, :, kᴺ:kᴺ)
end

function EarthSystemModels.ocean_surface_temperature(model::OceananigansOceanModels)
    kᴺ = size(model.grid, 3)
    return view(model.tracers.T.data, :, :, kᴺ:kᴺ)
end

function EarthSystemModels.ocean_surface_velocities(model::OceananigansOceanModels)
    kᴺ = size(model.grid, 3)
    return view(model.velocities.u, :, :, kᴺ), view(model.velocities.v, :, :, kᴺ)
end

EarthSystemModels.is_sea_ice_component(ocean::OceananigansModelSimulations) = EarthSystemModels.is_sea_ice_component(ocean.model)
EarthSystemModels.ocean_salinity(ocean::OceananigansModelSimulations)            = EarthSystemModels.ocean_salinity(ocean.model)
EarthSystemModels.ocean_temperature(ocean::OceananigansModelSimulations)         = EarthSystemModels.ocean_temperature(ocean.model)
EarthSystemModels.ocean_surface_salinity(ocean::OceananigansModelSimulations)    = EarthSystemModels.ocean_surface_salinity(ocean.model)
EarthSystemModels.ocean_surface_temperature(ocean::OceananigansModelSimulations) = EarthSystemModels.ocean_surface_temperature(ocean.model)
EarthSystemModels.ocean_surface_velocities(ocean::OceananigansModelSimulations)  = EarthSystemModels.ocean_surface_velocities(ocean.model)

@kernel function _ocean_state_to_potential_temperature!(Tᵉˣ, Tᵒᶜ, Sᵒᶜ, kᴺ)
    i, j = @index(Global, NTuple)
    @inbounds Tᵉˣ[i, j, 1] = θ_from_Θ(max(0, Sᵒᶜ[i, j, kᴺ]), Tᵒᶜ[i, j, kᴺ])
end

function EarthSystemModels.interpolate_state!(exchanger, grid, model::HydrostaticFreeSurfaceModel, coupled_model)
    Tᵉˣ = exchanger.state.T
    Tᵒᶜ = model.tracers.T
    Sᵒᶜ = model.tracers.S
    kᴺ = size(model.grid, 3)
    arch = architecture(model.grid)
    # Over the halo columns too, which the interface flux kernels visit: the ocean's own tracer
    # halos are already filled, so this needs no separate fill of `Tᵉˣ`.
    launch!(arch, grid, interface_kernel_parameters(grid), _ocean_state_to_potential_temperature!, Tᵉˣ, Tᵒᶜ, Sᵒᶜ, kᴺ)
    return nothing
end

EarthSystemModels.interpolate_state!(exchanger, grid, ::NonhydrostaticModel, coupled_model) = nothing

EarthSystemModels.interpolate_state!(exchanger, grid, ocean::OceananigansModelSimulations, coupled_model) =
    EarthSystemModels.interpolate_state!(exchanger, grid, ocean.model, coupled_model)

InterfaceComputations.ComponentExchanger(ocean::OceananigansModelSimulations, grid) = ComponentExchanger(ocean.model, grid)

function EarthSystemModels.InterfaceComputations.ComponentExchanger(model::OceananigansOceanModels, grid)
    ocean_grid = model.grid

    if ocean_grid == grid
        u = model.velocities.u
        v = model.velocities.v
        S = model.tracers.S
    else
        u = Field{Center, Center, Nothing}(grid)
        v = Field{Center, Center, Nothing}(grid)
        S = Field{Center, Center, Nothing}(grid)
    end

    T = Field{Center, Center, Nothing}(grid)

    # Near-surface vertical tracer diffusivity, evaluated lazily inside the
    # interface flux kernel by formulations that consume it (`InteriorDiffusivity`).
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
@inline net_flux(bc::IMEXFlux) = net_flux(bc.explicit_flux)
@inline net_flux(f::DiscreteForcing)   = f.parameters
@inline net_flux(mf::MultipleForcings) = net_flux(mf.forcings[1])

# The implicit coefficient λ of an affine surface flux; `nothing` when the flux has no implicit part.
@inline net_flux_coefficient(condition) = nothing
@inline net_flux_coefficient(bc::IMEXFlux) = net_flux(bc.implicit_coefficient)

InterfaceComputations.net_fluxes(ocean::OceananigansModelSimulations) = InterfaceComputations.net_fluxes(ocean.model)

function EarthSystemModels.InterfaceComputations.net_fluxes(model::OceananigansOceanModels)
    # TODO: Generalize this to work with any ocean model
    u_top = model.velocities.u.boundary_conditions.top.condition
    v_top = model.velocities.v.boundary_conditions.top.condition
    net_ocean_surface_fluxes = (; u = net_flux(u_top),
                                  v = net_flux(v_top),
                                  u_coefficient = net_flux_coefficient(u_top),
                                  v_coefficient = net_flux_coefficient(v_top))

    tracers = model.tracers
    ocean_surface_tracer_fluxes = NamedTuple(name => net_flux(tracers[name].boundary_conditions.top.condition) for name in keys(tracers))

    freshwater_volume_flux = extract_freshwater_flux(model.tracers.S.boundary_conditions.top.condition)
    heat_exchange = freshwater_exchange(model.tracers.T.boundary_conditions.top.condition)

    fluxes = merge(ocean_surface_tracer_fluxes, net_ocean_surface_fluxes,
                   (; η = freshwater_volume_flux,
                      freshwater_heat_content = heat_exchange.content_flux))

    return fluxes
end

end # module
