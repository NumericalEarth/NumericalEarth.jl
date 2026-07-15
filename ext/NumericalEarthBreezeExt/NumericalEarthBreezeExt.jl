module NumericalEarthBreezeExt

using Breeze: Breeze
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: launch!
using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, computed_fluxes

include("coupled_radiation.jl")
include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")
include("breeze_prognostic_state.jl")
include("breeze_state_exchanger.jl")
include("breeze_nested_atmosphere.jl")
include("breeze_air_land_radiation.jl")

end # module NumericalEarthBreezeExt
