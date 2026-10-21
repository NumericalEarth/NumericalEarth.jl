module NumericalEarthTerrariumExt

using KernelAbstractions: @kernel, @index
using Oceananigans: Simulation, Center, architecture
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!
using Oceananigans.Simulations: conjure_time_step_wizard!

import Terrarium
import Terrarium.RingGrids
import NumericalEarth
import NumericalEarth.EarthSystemModels: interpolate_state!, update_net_fluxes!
import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger
using NumericalEarth.Lands: land_model

const TerrariumSimulation = Simulation{<:Terrarium.ModelIntegrator}

include("terrarium_land_simulations.jl")
include("terrarium_exchanger.jl")
include("terrarium_input_sources.jl")

end # module NumericalEarthTerrariumExt
