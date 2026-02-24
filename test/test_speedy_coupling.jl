using SpeedyWeather, XESMF
using NumericalEarth
using Oceananigans
using Dates
using Test

NumericalEarthSpeedyWeatherExt = Base.get_extension(NumericalEarth, :NumericalEarthSpeedyWeatherExt)
@test !isnothing(NumericalEarthSpeedyWeatherExt)

spectral_grid = SpeedyWeather.SpectralGrid(trunc=51, nlayers=3, Grid=FullClenshawGrid)
oceananigans_grid = LatitudeLongitudeGrid(Oceananigans.CPU(); size=(200, 100, 1), latitude=(-80, 80), longitude=(0, 360), z = (0, 1))

ocean = NumericalEarth.Oceans.ocean_simulation(oceananigans_grid; momentum_advection=nothing, tracer_advection=nothing, closure=nothing)
Oceananigans.set!(ocean.model, T=EN4Metadatum(:temperature), S=EN4Metadatum(:salinity))

atmos = NumericalEarth.atmosphere_simulation(spectral_grid)

radiation   = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)
earth_model = EarthSystemModel(atmos, ocean, default_sea_ice(); radiation)

Qca = atmos.prognostic_variables.ocean.sensible_heat_flux.data
Mva = atmos.prognostic_variables.ocean.surface_humidity_flux.data

@test !(all(Qca .== 0.0))
@test !(all(Mva .== 0.0))
