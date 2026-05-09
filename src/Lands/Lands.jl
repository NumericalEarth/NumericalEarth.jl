module Lands

export PrescribedLand,
       # Composable container
       SlabLand, SlabLandParameters,
       # Energy-balance closures
       SlabEnergy,
       # Hydrology closures
       ManabeBucket, DryLand, SaturatedSurface,
       # Surface-property closures
       ConstantSurfaceProperties,
       # RUC composite
       RucSlabLand, RucSlabLandParameters,
       RucEnergy, RucHydrology, RucSurfaceProperties,
       # Atmosphere-facing accessors
       surface_temperature, surface_wetness,
       # Land-class registry
       LandClassification, usgs_land_classifications,
       apply_land_classifications!

using Oceananigans
using Oceananigans.Utils: launch!, prettytime
using Oceananigans.Fields: Center, Face, CenterField, ConstantField, ZeroField, interior
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

# Container parameters and the SlabLand struct.
include("slab_land_parameters.jl")
include("slab_land.jl")

# Default closures (sized so `SlabEnergy + ManabeBucket + ConstantSurfaceProperties`
# is the canonical Manabe-bucket slab).
include("energy_balance/slab_energy.jl")
include("hydrology/manabe_bucket.jl")
include("hydrology/dry_land.jl")
include("hydrology/saturated_surface.jl")
include("surface_properties/constant_surface_properties.jl")

# RUC concrete closures. The parameter bag is loaded first because the
# RUC hydrology and surface-property closures dispatch on it. The
# user-facing `RucSlabLand(grid; …)` constructor and the RUC-composition
# specialisations of `update_net_fluxes!` / `ComponentExchanger` /
# checkpointing are loaded last, after the closure types they reference
# in their method signatures.
include("ruc_slab_land_parameters.jl")
include("energy_balance/ruc_energy.jl")
include("hydrology/ruc_hydrology.jl")
include("surface_properties/ruc_surface_properties.jl")
include("ruc_slab_land.jl")

# Legacy components and class registry
include("prescribed_land.jl")
include("prescribed_land_regridder.jl")
include("interpolate_land_state.jl")
include("usgs_land_classification.jl")

end # module Lands
