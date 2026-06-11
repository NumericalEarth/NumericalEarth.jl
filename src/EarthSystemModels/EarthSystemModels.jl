module EarthSystemModels

abstract type AbstractPrescribedComponent end

export
    EarthSystemModel,
    AbstractPrescribedComponent,
    OceanOnlyModel,
    OceanSeaIceModel,
    AtmosphereOceanModel,
    AtmosphereLandModel,
    BulkHumidity,
    SkinHumidity,
    FractionalHumidity,
    CriticalSaturation,
    DryLayerHumidity,
    StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity,
    ConstantTortuosity,
    MillingtonQuirk,
    ElevationCorrection,
    atmosphere_land_interface,
    SimilarityTheoryFluxes,
    LandRoughnessLength,
    CoefficientBasedFluxes,
    FreezingLimitedOceanTemperature,
    SkinTemperature,
    BulkTemperature,
    DiffusiveFlux,
    InteriorDiffusivity,
    compute_atmosphere_ocean_fluxes!,
    compute_atmosphere_sea_ice_fluxes!,
    compute_sea_ice_ocean_fluxes!,
    # Sea ice-ocean heat flux formulations
    IceBathHeatFlux,
    ThreeEquationHeatFlux,
    # Friction velocity formulations
    MomentumBasedFrictionVelocity

using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using KernelAbstractions: @kernel, @index
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

using Oceananigans: Oceananigans, AbstractModel, initialize!,
                    prognostic_state, restore_prognostic_state!
using Oceananigans.Architectures: architecture
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.Fields: ZeroField
using Oceananigans.Simulations: reset_clock!, Simulation
using Oceananigans.TimeSteppers: Clock, reset!, tick!, time_step!, update_state!, reconcile_state!
using Oceananigans.Utils: launch!, prettytime

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
InterfaceComputations.compute_sea_ice_ocean_fluxes!(::NoSeaIceInterfaceModel)      = nothing

InterfaceComputations.compute_atmosphere_ocean_fluxes!(::NoAtmosInterfaceModel)   = nothing
InterfaceComputations.compute_atmosphere_sea_ice_fluxes!(::NoAtmosInterfaceModel) = nothing
InterfaceComputations.compute_atmosphere_land_fluxes!(::NoAtmosInterfaceModel)    = nothing

InterfaceComputations.compute_atmosphere_ocean_fluxes!(::NoOceanInterfaceModel) = nothing
InterfaceComputations.compute_sea_ice_ocean_fluxes!(::NoOceanInterfaceModel)    = nothing

# `NoInterface` means atmosphere-ocean / atmosphere-sea-ice / sea-ice-ocean
# are all absent. The atmosphere-land interface may still be present
# (e.g. an `AtmosphereLandModel` with ocean = sea_ice = nothing), so we
# explicitly do *not* fall back `compute_atmosphere_land_fluxes!` here —
# its own 2-arg Nothing dispatch handles the missing-AL-interface case.
InterfaceComputations.compute_atmosphere_ocean_fluxes!(::NoInterfaceModel)   = nothing
InterfaceComputations.compute_atmosphere_sea_ice_fluxes!(::NoInterfaceModel) = nothing
InterfaceComputations.compute_sea_ice_ocean_fluxes!(::NoInterfaceModel)      = nothing

end # module
