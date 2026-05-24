module Lands

export AbstractLand,
       PrescribedLand,
       # Composable container
       SlabLand,
       # Energy-balance closures
       SlabEnergy,
       # Hydrology closures
       BucketHydrology, DryLand, SaturatedSurface,
       # Surface-property closures
       ConstantSurfaceProperties,
       # Atmosphere-facing accessors
       surface_temperature, surface_wetness

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
using Oceananigans.Fields: AbstractField, CenterField, ConstantField, Field, Center, Face, ZeroField, interior
using Oceananigans.Grids: grid_name
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!, prettysummary, prettytime

using ..NumericalEarth: NumericalEarth
using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters, ComponentExchanger

import Oceananigans.TimeSteppers: time_step!
import ..EarthSystemModels: interpolate_state!,
                            update_net_fluxes!,
                            surface_temperature
import ..EarthSystemModels.InterfaceComputations: atmosphere_land_interface, initialize!

# Closure interfaces
include("energy_balance/energy_balance.jl")
include("hydrology/hydrology.jl")
include("surface_properties/surface_properties.jl")
include("properties/property_providers.jl")

# Container.
include("slab_land.jl")

# Default closures (sized so `SlabEnergy + BucketHydrology + ConstantSurfaceProperties`
# is the canonical Manabe-bucket slab).
include("energy_balance/slab_energy.jl")
include("hydrology/bucket_hydrology.jl")
include("hydrology/dry_land.jl")
include("hydrology/saturated_surface.jl")
include("surface_properties/constant_surface_properties.jl")

# Legacy PrescribedLand component (river / iceberg freshwater forcing).
include("prescribed_land.jl")
include("prescribed_land_regridder.jl")
include("interpolate_land_state.jl")

end # module Lands
