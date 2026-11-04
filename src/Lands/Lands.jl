module Lands

export AbstractLand,
       PrescribedLand,
       RiverRouting,
       build_river_routing,
       coastal_outlet_indices,
       routable_grid,
       # Composable container
       SlabLand, PrognosticCanopyAir, PrognosticSkin,
       # Energy-balance closures
       SlabEnergy,
       ForceRestoreEnergy,
       WaterCoupledEnergy,
       # Hydrology closures
       BucketHydrology, DryLand, SaturatedSurface,
       # Variably saturated hydrology + sub-closures
       VanGenuchtenRetention, VanGenuchtenConductivity,
       WaterViscosity, CosbyConductivity, saturated_conductivity,
       NoDeepLiquidWaterFlux, FreeDrainageFlux, DarcyDeepLiquidWaterFlux, LinearReservoirDrainage,
       NoRunoff, InfiltrationCapacityRunoff,
       VariablySaturatedHydrology,
       # Canopy interception store
       InterceptingHydrology,
       # Pedotransfer functions + depth-layer combination
       WeynantsPedotransfer, HYPRESPedotransfer,
       soil_hydraulic_parameters, soil_hydraulic_properties,
       # Urban aerodynamic roughness closures
       AbstractUrbanRoughness, MorphometricRoughness,
       IsotropicFrontalArea, EmpiricalFrontalArea,
       UniformHeight, VariableHeight,
       urban_roughness, compute_aerodynamic_roughness!, aerodynamic_parameters, building_morphometry,
       fill_aerodynamic_roughness_gaps!,
       # Atmosphere-facing accessors
       surface_temperature, surface_saturation,
       # Land model and simulations
       land_model,
       land_simulation

"""
    abstract type AbstractLand end

Marker abstract type for *prognostic* NumericalEarth land components
([`SlabLand`](@ref) and any future multi-layer variants). `PrescribedLand`
is a prescribed-forcing component and subtypes `AbstractPrescribedComponent`
instead.
"""
abstract type AbstractLand end

using Adapt: Adapt
using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, prognostic_state, restore_prognostic_state!
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: CPU, architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField, CenterField, Field, Center, Face, ZeroField, interior, regrid!, compute!,
                           dimension_faces
using Oceananigans.Grids: grid_name, Center, Face, LatitudeLongitudeGrid, Bounded, Flat,
                          znodes, λnode, φnode, hack_cosd, hack_sind
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!, update_state!, clock_convert, kernel_time_type
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!, prettysummary, prettytime
using Base: ForwardOrdering

using ..NumericalEarth: NumericalEarth, stateindex
using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent, surface_temperature
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations,
                                                 CanopyInterface, EnergyBalanceTemperature,
                                                 ConstantTortuosity, PowerLawTortuosity,
                                                 effective_vapor_diffusivity,
                                                 interface_kernel_parameters, ComponentExchanger,
                                                 canopy_transmittance, land_surface_fluxes,
                                                 convert_if_number, maybe_write!

# Extended by external land models
function land_model end
function land_simulation end

# Closure interfaces
include("energy_balance/energy_balance.jl")
include("hydrology/hydrology.jl")
include("properties/property_providers.jl")

# Pedotransfer functions and depth-layer combination.
include("properties/pedotransfer.jl")
include("properties/soil_hydraulic_properties.jl")

# Urban aerodynamic roughness closures.
include("roughness/urban_roughness_closure.jl")
include("roughness/urban_roughness_field.jl")

# Container.
include("slab_land.jl")

# Default closures (sized so `SlabEnergy + BucketHydrology` is the canonical
# bucket slab of Manabe (1969); see `docs/src/NumericalEarth.bib`).
# `SlabEnergy` is the `τ → ∞` limit of `ForceRestoreEnergy` and lives in the
# same file as a thin constructor.
include("energy_balance/force_restore_energy.jl")
include("energy_balance/water_coupled_energy.jl")
include("hydrology/bucket_hydrology.jl")
include("hydrology/dry_land.jl")
include("hydrology/saturated_surface.jl")

# Variably saturated hydrology + sub-closures (deep liquid water flux, runoff,
# retention/conductivity). These are pure helpers and small types used by
# `VariablySaturatedHydrology`; they have no dependence on each other.
include("hydrology/hydraulic_functions.jl")
include("hydrology/deep_liquid_water_fluxes.jl")
include("hydrology/runoff_models.jl")
include("hydrology/variably_saturated_hydrology.jl")

# Canopy interception store — wraps a soil hydrology (throughfall + wet-canopy
# evaporation), depends on `merge_unique` from the container above.
include("hydrology/intercepting_hydrology.jl")

include("surface_storage.jl")

# Legacy PrescribedLand component (river / iceberg freshwater forcing).
include("prescribed_land.jl")
include("prescribed_land_regridder.jl")
include("interpolate_land_state.jl")
include("river_routing.jl")

end # module Lands
