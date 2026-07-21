module Diagnostics

export MixedLayerDepthField, MixedLayerDepthOperand, BudgetComputation
export meridional_heat_transport, MeridionalFluxMethod, TendencyMethod
export net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux, ocean_freshwater_heat_flux,
       net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux

using KernelAbstractions: @index, @kernel
using ConservativeRegridding # Load OceananigansConservativeRegriddingExt.

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: CumulativeIntegral, Integral, KernelFunctionOperation, RegriddedOperation
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: Field, FieldStatus, ZeroField
using Oceananigans.Grids: new_data, inactive_cell, znode, Face, Center,
                          LatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, MutableGridOfSomeKind
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Simulations: Callback, Simulation, validate_schedule
using Oceananigans: TimeStepCallsite
using Oceananigans.OutputWriters: IterationInterval
using Oceananigans.Utils: launch!

using ..EarthSystemModels: EarthSystemModel, NoSeaIceInterfaceModel,
                           NoOceanInterfaceModel, NoInterfaceModel
using ..EarthSystemModels: reference_density, heat_capacity
using ..Oceans: MultipleFluxes, get_radiative_forcing

include("mixed_layer_depth.jl")
include("interface_fluxes.jl")
include("budget_diagnostic.jl")
include("meridional_heat_transport.jl")

end # module
