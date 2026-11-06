module NumericalEarthBreezeExt

using Breeze: Breeze
using Breeze.BoundaryConditions: EnergyFluxBoundaryConditionFunction, StaticEnergyFluxBoundaryConditionFunction,
                                 ThetaFluxBoundaryConditionFunction
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.MultiRegion: MultiRegionField, MultiRegionGrids, MultiRegionObject
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: @apply_regionally, getregion, launch!
using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels: sea_ice_concentration
using NumericalEarth.EarthSystemModels.InterfaceComputations: AtmosphereSurfaceFluxes, ComponentExchanger, computed_fluxes

# TODO: move to Breeze, next to the `Adapt` rules of these boundary condition functions
Breeze.Utils.@regional EnergyFluxBoundaryConditionFunction
Breeze.Utils.@regional StaticEnergyFluxBoundaryConditionFunction
Breeze.Utils.@regional ThetaFluxBoundaryConditionFunction

include("coupled_radiation.jl")
include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")
include("breeze_prognostic_state.jl")
include("breeze_state_exchanger.jl")
include("breeze_nested_atmosphere.jl")
include("breeze_air_land_radiation.jl")
include("breeze_surface_albedo.jl")

end # module NumericalEarthBreezeExt
