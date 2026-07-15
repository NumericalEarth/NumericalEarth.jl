module Diagnostics

export MixedLayerDepthField, MixedLayerDepthOperand
export meridional_heat_transport, MeridionalFluxMethod, TendencyMethod
export net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
       net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux

using KernelAbstractions: @index, @kernel
import ConservativeRegridding # Load OceananigansConservativeRegriddingExt.

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: CumulativeIntegral, Integral
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: ConservativeRegriddedField, Field, FieldStatus, ZeroField
using Oceananigans.Grids: new_data, inactive_cell, znode, Face, Center,
                          LatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Utils: launch!

using ..EarthSystemModels: EarthSystemModel, NoSeaIceInterfaceModel,
                           NoOceanInterfaceModel, NoInterfaceModel
using ..Oceans: MultipleFluxes

include("mixed_layer_depth.jl")
include("meridional_heat_transport.jl")
include("interface_fluxes.jl")

end # module
