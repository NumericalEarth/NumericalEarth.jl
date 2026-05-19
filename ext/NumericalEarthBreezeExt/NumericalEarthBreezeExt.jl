module NumericalEarthBreezeExt

import Breeze
import NumericalEarth
import Oceananigans

using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Utils: launch!

using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, computed_fluxes

include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")

end # module NumericalEarthBreezeExt
