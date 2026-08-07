module NumericalEarthTerrariumExt

using KernelAbstractions
using Oceananigans: Field

import Terrarium
import Terrarium.RingGrids
import NumericalEarth
import Oceananigans

include("terrarium_land_simulations.jl")
include("terrarium_exchanger.jl")
include("terrarium_input_sources.jl")

end # module NumericalEarthTerrariumExt
