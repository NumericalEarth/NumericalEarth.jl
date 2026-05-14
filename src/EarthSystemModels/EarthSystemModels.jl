module EarthSystemModels

export
    EarthSystemModel,
    OceanOnlyModel,
    OceanSeaIceModel,
    AtmosphereOceanModel,
    SimilarityTheoryFluxes,
    CoefficientBasedFluxes,
    FreezingLimitedOceanTemperature,
    SkinTemperature,
    BulkTemperature,
    compute_atmosphere_ocean_fluxes!,
    compute_atmosphere_sea_ice_fluxes!,
    compute_sea_ice_ocean_fluxes!,
    # Sea ice-ocean heat flux formulations
    IceBathHeatFlux,
    ThreeEquationHeatFlux,
    # Friction velocity formulations
    MomentumBasedFrictionVelocity

import Oceananigans

using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using KernelAbstractions: @kernel, @index
using Oceananigans: AbstractModel
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: Operators
using Oceananigans.Simulations: Simulation
using Oceananigans.TimeSteppers: tick!
using Oceananigans.Utils: launch!
using Printf: Printf

import Thermodynamics as AtmosphericThermodynamics

# Simulations interface
import Oceananigans: fields, initialize!, prognostic_fields, prognostic_state, restore_prognostic_state!
import Oceananigans.Architectures: architecture
import Oceananigans.Diagnostics: NaNChecker
import Oceananigans.OutputWriters: default_included_properties
import Oceananigans.Simulations: timestepper, reset_clock!, iteration
import Oceananigans.TimeSteppers: time_step!, reset!, update_state!, reconcile_state!
import Oceananigans.Utils: prettytime

include("components.jl")

#####
##### The coupled model
#####

const default_gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration
const default_freshwater_density = 1000 # kg m⁻³

include("InterfaceComputations/InterfaceComputations.jl")

using .InterfaceComputations

include("earth_system_model.jl")
include("time_step_earth_system_model.jl")

#####
#####  Fallbacks for no-interface models
#####

using .InterfaceComputations: ComponentInterfaces, AtmosphereInterface, SeaIceOceanInterface

const NoSeaIceInterface = ComponentInterfaces{<:AtmosphereInterface,  <:Nothing, <:Nothing}
const NoOceanInterface  = ComponentInterfaces{<:Nothing, <:AtmosphereInterface,  <:Nothing}
const NoAtmosInterface  = ComponentInterfaces{<:Nothing, <:Nothing, <:SeaIceOceanInterface}
const NoInterface       = ComponentInterfaces{<:Nothing, <:Nothing, <:Nothing}

const NoSeaIceInterfaceModel = EarthSystemModel{R, A, L, I, O, <:NoSeaIceInterface} where {R, A, L, I, O}
const NoAtmosInterfaceModel  = EarthSystemModel{R, A, L, I, O, <:NoAtmosInterface}  where {R, A, L, I, O}
const NoOceanInterfaceModel  = EarthSystemModel{R, A, L, I, O, <:NoOceanInterface}  where {R, A, L, I, O}
const NoInterfaceModel       = EarthSystemModel{R, A, L, I, O, <:NoInterface}       where {R, A, L, I, O}

InterfaceComputations.compute_atmosphere_sea_ice_fluxes!(::NoSeaIceInterfaceModel) = nothing
InterfaceComputations.compute_sea_ice_ocean_fluxes!(::NoSeaIceInterfaceModel) = nothing

InterfaceComputations.compute_atmosphere_ocean_fluxes!(::NoAtmosInterfaceModel) = nothing
InterfaceComputations.compute_atmosphere_sea_ice_fluxes!(::NoAtmosInterfaceModel) = nothing

InterfaceComputations.compute_atmosphere_ocean_fluxes!(::NoOceanInterfaceModel) = nothing
InterfaceComputations.compute_sea_ice_ocean_fluxes!(::NoOceanInterfaceModel) = nothing

InterfaceComputations.compute_atmosphere_ocean_fluxes!(::NoInterfaceModel) = nothing
InterfaceComputations.compute_atmosphere_sea_ice_fluxes!(::NoInterfaceModel) = nothing
InterfaceComputations.compute_sea_ice_ocean_fluxes!(::NoInterfaceModel) = nothing

end # module
