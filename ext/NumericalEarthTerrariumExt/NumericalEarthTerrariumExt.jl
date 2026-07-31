module NumericalEarthTerrariumExt

using KernelAbstractions

import Terrarium
import NumericalEarth
import Oceananigans

include("terrarium_land_simulations.jl")
include("terrarium_exchanger.jl")
include("terrarium_input_sources.jl")

end # module NumericalEarthTerrariumExt
