module NumericalEarthBreezeExt

import Breeze
using NumericalEarth: NumericalEarth
import Oceananigans

using KernelAbstractions: @kernel, @index
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Utils: launch!

using NumericalEarth.EarthSystemModels.InterfaceComputations: computed_fluxes

include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")

end # module NumericalEarthBreezeExt
