module NumericalEarthSpeedyWeatherExt

using OffsetArrays: OffsetArrays
using KernelAbstractions: KernelAbstractions, @index, @kernel
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Grids: Center
using Oceananigans.Fields: Field, interior, regrid!
using Oceananigans.Simulations: Simulations
using Statistics: Statistics
using XESMF: XESMF

import SpeedyWeather
using NumericalEarth: NumericalEarth
import Oceananigans
import SpeedyWeather.RingGrids

include("speedy_atmosphere_simulations.jl")
include("speedy_regridder.jl")
include("speedy_weather_exchanger.jl")

end # module NumericalEarthSpeedyWeatherExt
