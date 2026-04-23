module Diagnostics

export MixedLayerDepthField, MixedLayerDepthOperand
export meridional_heat_transport
export frazil_temperature_flux, net_ocean_temperature_flux, sea_ice_ocean_temperature_flux, atmosphere_ocean_temperature_flux,
       frazil_heat_flux, net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
       net_ocean_salinity_flux, sea_ice_ocean_salinity_flux, atmosphere_ocean_salinity_flux,
       net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Grids: new_data, inactive_cell, znode
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: FieldStatus
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction
using NumericalEarth.EarthSystemModels: EarthSystemModel
using NumericalEarth.Oceans: MultipleFluxes

import Oceananigans.Fields: compute!

include("mixed_layer_depth.jl")
include("meridional_heat_transport.jl")
include("interface_fluxes.jl")

end # module
