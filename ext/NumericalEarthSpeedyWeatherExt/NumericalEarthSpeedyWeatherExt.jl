module NumericalEarthSpeedyWeatherExt

using OffsetArrays
using KernelAbstractions
using Statistics

import SpeedyWeather 
import NumericalEarth 
import Oceananigans 
import SpeedyWeather.RingGrids

include("speedy_atmosphere_simulations.jl")
include("speedy_regridder.jl")
include("speedy_weather_exchanger.jl")

end # module NumericalEarthSpeedyWeatherExt
