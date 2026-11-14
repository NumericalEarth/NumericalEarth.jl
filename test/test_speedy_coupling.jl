include("runtests_setup.jl")

using ConservativeRegridding
using SpeedyWeather

NumericalEarthSpeedyWeatherExt = Base.get_extension(NumericalEarth, :NumericalEarthSpeedyWeatherExt)
@test !isnothing(NumericalEarthSpeedyWeatherExt)

spectral_grid = SpeedyWeather.SpectralGrid(truncation=52, nlayers=3, Grid=FullClenshawGrid)
oceananigans_grid = LatitudeLongitudeGrid(Oceananigans.CPU(); size=(200, 100, 1), latitude=(-80, 80), longitude=(0, 360), z = (0, 1))

ocean = NumericalEarth.Oceans.ocean_simulation(oceananigans_grid; momentum_advection=nothing, tracer_advection=nothing, closure=nothing)
Oceananigans.set!(ocean.model, T=Metadatum(:temperature; dataset=SyntheticOcean()), S=Metadatum(:salinity; dataset=SyntheticOcean()))

atmos = NumericalEarth.atmosphere_simulation(spectral_grid)

earth_model = EarthSystemModel(; atmosphere=atmos, sea_ice=default_sea_ice(), ocean)

Qca = atmos.variables.parameterizations.ocean.sensible_heat_flux.data
Mva = atmos.variables.parameterizations.ocean.surface_humidity_flux.data

@test !(all(Qca .== 0.0))
@test !(all(Mva .== 0.0))

@testset "SpeedyWeather feels the sea ice" begin
    grid = LatitudeLongitudeGrid(Oceananigans.CPU(); size=(40, 20, 1), latitude=(-80, 80), longitude=(0, 360), z=(-10, 0))
    ocean = NumericalEarth.Oceans.ocean_simulation(grid; momentum_advection=nothing, tracer_advection=nothing, closure=nothing)
    Oceananigans.set!(ocean.model, T=20, S=35)

    sea_ice = NumericalEarth.SeaIces.sea_ice_simulation(grid, ocean; dynamics=nothing, advection=nothing)
    Oceananigans.set!(sea_ice.model, h=1, ℵ=1)

    atmos = NumericalEarth.atmosphere_simulation(spectral_grid)
    EarthSystemModel(; atmosphere=atmos, sea_ice, ocean)

    @test maximum(atmos.variables.prognostic.ocean.sea_surface_temperature.data) < 273.15
end
