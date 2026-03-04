module Diagnostics

export MixedLayerDepthField, MixedLayerDepthOperand
export net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
       net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Grids: new_data, inactive_cell, znode
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: FieldStatus
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel
using NumericalEarth.EarthSystemModels: EarthSystemModel

import Oceananigans.Fields: compute!

include("mixed_layer_depth.jl")
include("interface_fluxes.jl")

end # module
