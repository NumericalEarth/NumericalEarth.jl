module NumericalEarthTerrariumExt

using OffsetArrays
using KernelAbstractions
using Statistics

import Terrarium
import Terrarium.RingGrids
import NumericalEarth
import Oceananigans

include("terrarium_land_simulations.jl")
include("terrarium_exchanger.jl")

end # module NumericalEarthTerrariumExt
