module NumericalEarthBreezeExt

import Breeze
import NumericalEarth
import Oceananigans

using Oceananigans.Utils: launch!
using Oceananigans.Grids: architecture
using Oceananigans.Fields: ZeroField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using KernelAbstractions: @kernel, @index

import NumericalEarth: AtmosphereOceanModel
import NumericalEarth.Atmospheres: atmosphere_simulation
import NumericalEarth.EarthSystemModels: thermodynamics_parameters,
                                          surface_layer_height,
                                          boundary_layer_height,
                                          interpolate_state!,
                                          update_net_fluxes!

import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, ComponentInterfaces, net_fluxes

import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Advection: cell_advection_timescale

include("breeze_atmosphere_interface.jl")
include("breeze_atmosphere_simulation.jl")

end # module NumericalEarthBreezeExt
