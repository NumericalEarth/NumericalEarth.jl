module Lands

export AbstractLand,
       PrescribedLand,
       # Composable container
       SlabLand,
       # Energy-balance closures
       SlabEnergy,
       ForceRestoreEnergy,
       # Hydrology closures
       BucketHydrology, DryLand, SaturatedSurface,
       # Variably saturated hydrology + sub-closures
       VanGenuchtenRetention, VanGenuchtenConductivity,
       NoDeepLiquidFlux, FreeDrainageFlux, DarcyDeepLiquidFlux, LinearReservoirDrainage,
       NoRunoff, InfiltrationCapacityRunoff,
       VariablySaturatedBucketHydrology,
       # Atmosphere-facing accessors
       surface_temperature, surface_saturation

"""
    abstract type AbstractLand end

Marker abstract type for *prognostic* NumericalEarth land components
([`SlabLand`](@ref) and any future multi-layer variants). `PrescribedLand`
is a prescribed-forcing component and subtypes `AbstractPrescribedComponent`
instead.
"""
abstract type AbstractLand end

using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, prognostic_state, restore_prognostic_state!
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField, CenterField, Field, Center, Face, ZeroField
using Oceananigans.Grids: grid_name
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!, prettysummary, prettytime

using ..NumericalEarth: NumericalEarth, stateindex
using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters, ComponentExchanger

import ..EarthSystemModels: interpolate_state!,
                            update_net_fluxes!,
                            surface_temperature
import ..EarthSystemModels.InterfaceComputations: atmosphere_land_interface

# Closure interfaces
include("energy_balance/energy_balance.jl")
include("hydrology/hydrology.jl")
include("properties/property_providers.jl")

# Container.
include("slab_land.jl")

# Default closures (sized so `SlabEnergy + BucketHydrology` is the canonical
# bucket slab of Manabe (1969); see `docs/src/NumericalEarth.bib`).
# `SlabEnergy` is the `τ → ∞` limit of `ForceRestoreEnergy` and lives in the
# same file as a thin constructor.
include("energy_balance/force_restore_energy.jl")
include("hydrology/bucket_hydrology.jl")
include("hydrology/dry_land.jl")
include("hydrology/saturated_surface.jl")

# Variably saturated bucket hydrology + sub-closures (deep liquid flux, runoff,
# retention/conductivity). These are pure helpers and small types used by
# `VariablySaturatedBucketHydrology`; they have no dependence on each other.
include("hydrology/hydraulic_functions.jl")
include("hydrology/deep_liquid_fluxes.jl")
include("hydrology/runoff_models.jl")
include("hydrology/variably_saturated_bucket_hydrology.jl")

# Legacy PrescribedLand component (river / iceberg freshwater forcing).
include("prescribed_land.jl")
include("prescribed_land_regridder.jl")
include("interpolate_land_state.jl")

end # module Lands
