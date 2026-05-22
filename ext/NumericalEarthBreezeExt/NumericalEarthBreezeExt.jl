module NumericalEarthBreezeExt

using Breeze: Breeze
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Utils: launch!
using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, computed_fluxes

include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")
include("breeze_child_model.jl")

end # module NumericalEarthBreezeExt
