module NumericalEarthBreezeExt

import Breeze
import NumericalEarth
import Oceananigans

using KernelAbstractions: @kernel, @index
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: architecture
using Oceananigans.Utils: launch!

import NumericalEarth.Atmospheres: atmosphere_simulation
import NumericalEarth.EarthSystemModels: thermodynamics_parameters,
                                          surface_layer_height,
                                          boundary_layer_height,
                                          interpolate_state!,
                                          update_net_fluxes!
import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, net_fluxes

import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.TimeSteppers: time_step!

include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")

end # module NumericalEarthBreezeExt
