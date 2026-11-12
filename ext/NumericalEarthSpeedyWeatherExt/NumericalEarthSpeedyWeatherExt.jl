module NumericalEarthSpeedyWeatherExt

using GPUArraysCore: @allowscalar
using Oceananigans: Oceananigans
using NumericalEarth: NumericalEarth
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.EarthSystemModels: EarthSystemModels
using SpeedyWeather: SpeedyWeather
using SpeedyWeather.RingGrids: RingGrids

include("speedy_atmosphere_simulations.jl")
include("speedy_weather_exchanger.jl")
include("speedy_radiation.jl")

end # module NumericalEarthSpeedyWeatherExt
