module NumericalEarthBreezeExt

import Breeze
import NumericalEarth
import Oceananigans

using KernelAbstractions: @kernel, @index
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Utils: launch!

import NumericalEarth.EarthSystemModels: thermodynamics_parameters,
                                          surface_layer_height,
                                          boundary_layer_height,
                                          interpolate_state!,
                                          update_net_fluxes!
import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, net_fluxes, computed_fluxes

include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")

end # module NumericalEarthBreezeExt
