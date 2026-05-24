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

Top-level abstract type for NumericalEarth land components. Concrete
subtypes (e.g. [`SlabLand`](@ref), [`PrescribedLand`](@ref)) participate
in the `EarthSystemModel` coupling by implementing
`time_step!`, `update_state!`, `surface_temperature`, and the
component-exchanger / atmosphere-land flux entry points.
"""
abstract type AbstractLand end

using Oceananigans
using Oceananigans.Utils: launch!, prettytime
using Oceananigans.Fields: AbstractField, Center, Face, CenterField, ZeroField, interior
using Oceananigans.Grids: grid_name, architecture, prettysummary
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units: Time
using Oceananigans.BoundaryConditions: fill_halo_regions!

using KernelAbstractions: @kernel, @index
using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters

import Oceananigans.TimeSteppers: time_step!, update_state!

import NumericalEarth.EarthSystemModels: interpolate_state!,
                                         update_net_fluxes!,
                                         surface_temperature

import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger,
                                                              atmosphere_land_interface,
                                                              initialize!

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
