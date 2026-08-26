module NumericalEarthTerrariumExt

using KernelAbstractions
using Oceananigans: Simulation, Center, architecture
using Oceananigans.Architectures
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

import Terrarium
import Terrarium.RingGrids
import NumericalEarth
import NumericalEarth.EarthSystemModels: interpolate_state!, update_net_fluxes!
import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger
import NumericalEarth.Lands: land_simulation, land_model

const TerrariumSimulation = Simulation{<:Terrarium.ModelIntegrator}

include("terrarium_land_simulations.jl")
include("terrarium_exchanger.jl")
include("terrarium_input_sources.jl")

end # module NumericalEarthTerrariumExt
