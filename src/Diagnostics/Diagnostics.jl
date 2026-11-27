module Diagnostics

export MixedLayerDepthField, MixedLayerDepthOperand
export meridional_heat_transport
export net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux, frazil_heat_flux,
       net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux

using NumericalEarth: SIGNATURES
using KernelAbstractions: @index, @kernel
using SeawaterPolynomials: BoussinesqEquationOfState, ρ′

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: Integral, KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.BuoyancyFormulations: BuoyancyForce, SeawaterBuoyancy, get_temperature_and_salinity, θ_and_sᴬ
using Oceananigans.Fields: Field, FieldStatus, ZeroField
using Oceananigans.Grids: new_data, inactive_cell, znode, Face, Center, OrthogonalSphericalShellGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: buoyancy_operation
using Oceananigans.Utils: launch!

using ..EarthSystemModels: EarthSystemModel, NoSeaIceInterfaceModel,
                           NoOceanInterfaceModel, NoInterfaceModel, prognostic_model
using ..Oceans: MultipleFluxes

include("mixed_layer_depth.jl")
include("meridional_heat_transport.jl")
include("interface_fluxes.jl")

end # module
