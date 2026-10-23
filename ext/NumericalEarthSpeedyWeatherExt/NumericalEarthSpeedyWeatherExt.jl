module NumericalEarthSpeedyWeatherExt

using GPUArraysCore: @allowscalar
using KernelAbstractions: KernelAbstractions, @index, @kernel
using Oceananigans: Oceananigans
using Oceananigans.Grids: Center
using Oceananigans.Fields: Field, interior
using NumericalEarth: NumericalEarth
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.EarthSystemModels: EarthSystemModels
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations
using SpeedyWeather: SpeedyWeather
using SpeedyWeather.RingGrids: RingGrids

include("speedy_atmosphere_simulations.jl")
include("speedy_weather_exchanger.jl")

end # module NumericalEarthSpeedyWeatherExt
