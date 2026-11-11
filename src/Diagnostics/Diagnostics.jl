module Diagnostics

export MixedLayerDepthField, MixedLayerDepthOperand
export meridional_heat_transport
export TransportSection, volume_transport, heat_transport, salt_transport, freshwater_transport
export net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux, frazil_heat_flux,
       net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux

using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: Integral, KernelFunctionOperation
using Oceananigans.Architectures: CPU, architecture, on_architecture
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FieldBoundaryConditions, fill_halo_regions!,
                                      West, East, South, North
using Oceananigans.Fields: Field, FieldStatus, Reduction, ZeroField, interior
using Oceananigans.Grids: new_data, inactive_cell, znode, λnode, φnode, Face, Center, OrthogonalSphericalShellGrid,
                         immersed_peripheral_node, immersed_inactive_node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Operators: Axᶠᶜᶜ, Ayᶜᶠᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Utils: launch!

using ..EarthSystemModels: EarthSystemModel, NoSeaIceInterfaceModel,
                           NoOceanInterfaceModel, NoInterfaceModel
using ..Oceans: MultipleFluxes

include("mixed_layer_depth.jl")
include("meridional_heat_transport.jl")
include("section_transports.jl")
include("interface_fluxes.jl")

end # module
