module Diagnostics

export MixedLayerDepthField, MixedLayerDepthOperand
export meridional_heat_transport
export frazil_temperature_flux, net_ocean_temperature_flux, sea_ice_ocean_temperature_flux, atmosphere_ocean_temperature_flux,
       frazil_heat_flux, net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
       net_ocean_salinity_flux, sea_ice_ocean_salinity_flux, atmosphere_ocean_salinity_flux,
       net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux

import Oceananigans
using Oceananigans.Architectures: architecture
<<<<<<< ncc/clean-imports
using Oceananigans.AbstractOperations: Integral
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Grids: new_data, inactive_cell, znode, Center, Face, OrthogonalSphericalShellGrid
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!, DiscreteBoundaryFunction
using Oceananigans.Fields: FieldStatus, Field
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
=======
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: FieldStatus, ZeroField
using Oceananigans.Grids: new_data, inactive_cell, znode
using Oceananigans.Models: buoyancy_operation
>>>>>>> main
using Oceananigans.Utils: launch!

using KernelAbstractions: @index, @kernel
<<<<<<< ncc/clean-imports
using NumericalEarth.EarthSystemModels: EarthSystemModel
=======
using NumericalEarth.EarthSystemModels: EarthSystemModel,
                                        NoSeaIceInterfaceModel,
                                        NoOceanInterfaceModel,
                                        NoInterfaceModel
>>>>>>> main
using NumericalEarth.Oceans: MultipleFluxes

import Oceananigans.Fields: compute!

include("mixed_layer_depth.jl")
include("meridional_heat_transport.jl")
include("interface_fluxes.jl")

end # module
