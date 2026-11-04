include("runtests_setup.jl")

using ConservativeRegridding
using SpeedyWeather

NumericalEarthSpeedyWeatherExt = Base.get_extension(NumericalEarth, :NumericalEarthSpeedyWeatherExt)
@test !isnothing(NumericalEarthSpeedyWeatherExt)

spectral_grid = SpeedyWeather.SpectralGrid(truncation=32, nlayers=3, Grid=FullClenshawGrid)
oceananigans_grid = LatitudeLongitudeGrid(Oceananigans.CPU(); size=(200, 100, 1), latitude=(-80, 80), longitude=(0, 360), z = (0, 1))

ocean = NumericalEarth.Oceans.ocean_simulation(oceananigans_grid; momentum_advection=nothing, tracer_advection=nothing, closure=nothing)
Oceananigans.set!(ocean.model, T=Metadatum(:temperature; dataset=SyntheticOcean()), S=Metadatum(:salinity; dataset=SyntheticOcean()))

atmos = NumericalEarth.atmosphere_simulation(spectral_grid)

earth_model = EarthSystemModel(; atmosphere=atmos, sea_ice=default_sea_ice(), ocean)

Qca = atmos.variables.prognostic.ocean.sensible_heat_flux.data
Mva = atmos.variables.prognostic.ocean.surface_humidity_flux.data

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
    SpeedyWeather.time_step!(atmos)

    σ = atmos.model.atmosphere.stefan_boltzmann
    @test maximum(atmos.variables.parameterizations.ocean.surface_longwave_up.data) < σ * 274^4

    atmos = NumericalEarth.atmosphere_simulation(spectral_grid)
    sea_ice_surface = NumericalEarth.Radiations.SurfaceRadiationProperties(0.5, 1.0)
    radiation = NumericalEarth.atmosphere_radiation(atmos; sea_ice_surface)
    EarthSystemModel(; radiation, atmosphere=atmos, sea_ice, ocean)

    albedo = atmos.variables.parameterizations.ocean.albedo.data
    @test all(isapprox.(albedo[albedo .> 0], 0.5; rtol=1e-5))
end

@testset "SpeedyWeather feels the ocean" begin
    underlying_grid = LatitudeLongitudeGrid(Oceananigans.CPU(); size=(40, 20, 1), latitude=(-80, 80), longitude=(0, 360), z=(-10, 0))
    western_land(λ, φ) = ifelse(λ < 90, 0, -20)
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(western_land))

    ocean = NumericalEarth.Oceans.ocean_simulation(grid; momentum_advection=nothing, tracer_advection=nothing, closure=nothing)
    Oceananigans.set!(ocean.model, T=35, S=35)

    atmos = NumericalEarth.atmosphere_simulation(spectral_grid)
    earth_model = EarthSystemModel(; atmosphere=atmos, sea_ice=default_sea_ice(), ocean)

    albedo = atmos.variables.parameterizations.ocean.albedo.data
    covered = albedo .> 0

    # Land cells do not dilute the area average over the water below each atmospheric cell
    @test any(.!covered)
    @test all(isapprox.(albedo[covered], 0.05; rtol=1e-5))

    # The heat and moisture fluxes survive a SpeedyWeather step, which reads them from the prognostic ocean
    SpeedyWeather.time_step!(atmos)
    @test maximum(atmos.variables.parameterizations.ocean.sensible_heat_flux.data) > 0
    @test maximum(atmos.variables.parameterizations.ocean.surface_humidity_flux.data) > 0
end

@testset "SpeedyWeather and the surface share one radiative budget" begin
    grid = LatitudeLongitudeGrid(Oceananigans.CPU(); size=(80, 40, 1), latitude=(-80, 80), longitude=(0, 360), z=(-10, 0))
    ocean = NumericalEarth.Oceans.ocean_simulation(grid; momentum_advection=nothing, tracer_advection=nothing, closure=nothing)
    Oceananigans.set!(ocean.model, T=(λ, φ, z) -> max(-1.8, 28 * cosd(φ)^2), S=35)

    ice_presence(λ, φ) = abs(φ) > 65 ? 1.0 : 0.0
    sea_ice = NumericalEarth.SeaIces.sea_ice_simulation(grid, ocean; dynamics=nothing, advection=nothing)
    Oceananigans.set!(sea_ice.model, h=ice_presence, ℵ=(λ, φ) -> 0.8 * ice_presence(λ, φ))

    atmos = NumericalEarth.atmosphere_simulation(spectral_grid)
    model = EarthSystemModel(; atmosphere=atmos, sea_ice, ocean)
    SpeedyWeather.time_step!(atmos)
    update_state!(model)

    regridder = model.interfaces.exchanger.atmosphere.regridder
    fluxes = model.radiation.interface_fluxes
    parameterizations = atmos.variables.parameterizations
    ℵ = interior(model.sea_ice.model.ice_concentration)

    exchange_area = regridder.exchange.to_component.src_areas .* vec(interior(regridder.wet_mask))
    atmosphere_area = regridder.exchange.to_component.dst_areas .* vec(regridder.wet_fraction)
    over_exchange_grid(values) = sum(exchange_area .* vec(values))
    over_atmosphere_grid(values) = sum(atmosphere_area .* vec(values))

    # The ocean's absorbed fluxes are stored per cell, the ice's and all emission per unit of each surface
    absorbed_shortwave = -over_exchange_grid(interior(fluxes.ocean.downwelling_shortwave) .+ ℵ .* interior(fluxes.sea_ice.downwelling_shortwave))
    absorbed_longwave = -over_exchange_grid(interior(fluxes.ocean.downwelling_longwave) .+ ℵ .* interior(fluxes.sea_ice.downwelling_longwave))
    emitted_longwave = over_exchange_grid((1 .- ℵ) .* interior(fluxes.ocean.upwelling_longwave) .+ ℵ .* interior(fluxes.sea_ice.upwelling_longwave))

    downwelling_shortwave = copy(parameterizations.surface_shortwave_down.data)
    downwelling_longwave = copy(parameterizations.surface_longwave_down.data)
    albedo = copy(parameterizations.ocean.albedo.data)

    @test over_atmosphere_grid(downwelling_shortwave) ≈ absorbed_shortwave + over_atmosphere_grid(albedo .* downwelling_shortwave) rtol=1e-6

    SpeedyWeather.time_step!(atmos)

    upwelling_longwave = parameterizations.ocean.surface_longwave_up.data
    @test over_atmosphere_grid(upwelling_longwave) ≈ emitted_longwave + over_atmosphere_grid(downwelling_longwave) - absorbed_longwave rtol=1e-6

    sunlit = (parameterizations.surface_shortwave_down.data .> 10) .& (regridder.wet_fraction .> 0)
    used_albedo = parameterizations.ocean.surface_shortwave_up.data[sunlit] ./ parameterizations.surface_shortwave_down.data[sunlit]
    @test all(isapprox.(used_albedo, albedo[sunlit]; atol=1e-5))

    # A change in SpeedyWeather's downwelling reaches both surfaces
    update_state!(model)
    interfaces = model.interfaces
    ocean_temperature_flux = interior(interfaces.net_fluxes.ocean.T)
    sea_ice_heat_flux = interior(interfaces.net_fluxes.sea_ice.top.heat)
    initial_ocean_temperature_flux = copy(ocean_temperature_flux)
    initial_sea_ice_heat_flux = copy(sea_ice_heat_flux)

    parameterizations.surface_longwave_down.data .+= 100
    update_state!(model)

    ocean_properties = interfaces.ocean_properties
    ocean_heat_flux_change = (ocean_temperature_flux .- initial_ocean_temperature_flux) .* ocean_properties.reference_density .* ocean_properties.heat_capacity
    ϵ = model.radiation.surface_properties.ocean.emissivity
    @test all(isapprox.(ocean_heat_flux_change[ℵ .== 0], -ϵ * 100; atol=1e-2))
    @test maximum(abs, sea_ice_heat_flux .- initial_sea_ice_heat_flux) > 1

    Oceananigans.TimeSteppers.time_step!(model, convert(eltype(grid), atmos.model.time_stepping.Δt))
    @test !any(isnan, interior(ocean.model.tracers.T))
end
