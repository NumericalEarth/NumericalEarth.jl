include("runtests_setup.jl")

using Terrarium
using Oceananigans.TimeSteppers: update_state!, time_step!
using Oceananigans.Fields: interior

# Couple a Terrarium `LandModel` into NumericalEarth's `EarthSystemModel`, with the surface
# turbulent fluxes computed by NumericalEarth (Monin-Obukhov) and consumed by Terrarium.
# Requires `NumericalEarthTerrariumExt`, loaded automatically when Terrarium is present.
@testset "Terrarium land coupling" begin
    # CPU only: the Terrarium column grid is not yet GPU-backed.
    gpu_test && return

    land_grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 20))
    soil = SoilEnergyWaterCarbon(eltype(land_grid);
                                 hydrology = SoilHydrology(eltype(land_grid), RichardsEq()))
    land = NumericalEarth.land_simulation(land_grid; soil, vegetation = nothing,
        initializers = (temperature = (x, z) -> 5.0 - 0.02 * z,
                        saturation_water_ice = (x, z) -> 0.5))

    # Atmosphere on the exchange grid (= land's flattened field grid; 1:1, no regridding).
    # `land` is an Oceananigans `Simulation` wrapping the Terrarium `ModelIntegrator`.
    exchange_grid = land.model.grid
    atmosphere = NumericalEarth.PrescribedAtmosphere(exchange_grid;
                                                     surface_layer_height = 10,
                                                     boundary_layer_height = 512)
    fill!(parent(atmosphere.velocities.u), 3)
    fill!(parent(atmosphere.velocities.v), 0)
    fill!(parent(atmosphere.temperature), 288)      # K
    fill!(parent(atmosphere.specific_humidity), 0.005)
    fill!(parent(atmosphere.pressure), 101325)

    model = AtmosphereLandModel(atmosphere, land; radiation = nothing)
    update_state!(model)

    # The land exchanger publishes skin temperature (K) and surface saturation for the
    # atmosphere-land flux kernel.
    ex = model.interfaces.exchanger.land
    @test hasproperty(ex.state, :T)
    @test hasproperty(ex.state, :saturation)
    @test all(isfinite.(interior(ex.state.T)))
    @test all(interior(ex.state.saturation) .≈ 0.5)

    # The MO scheme produced finite turbulent fluxes, pushed into Terrarium's inputs.
    state = land.model.state
    @test all(isfinite.(interior(state.sensible_heat_flux)))
    @test all(isfinite.(interior(state.latent_heat_flux)))
    @test all(interior(state.windspeed) .≈ 3)

    # Advancing the coupled model integrates the soil with the coupler's ground heat flux.
    for _ in 1:3
        time_step!(model, 60.0)
    end
    @test all(isfinite.(interior(state.internal_energy)))
    @test all(isfinite.(interior(state.ground_heat_flux)))
    @test any(abs.(Array(interior(state.ground_heat_flux))) .> 0)
end

# With radiation, the coupler assembles the full surface energy budget into Terrarium's
# prescribed `ground_heat_flux` and pushes the upwelling radiation that drove it.
@testset "Terrarium land coupling with radiation" begin
    gpu_test && return

    land_grid = ColumnGrid(CPU(), ExponentialSpacing(Δz_max = 1.0, N = 20))
    land = NumericalEarth.land_simulation(land_grid; vegetation = nothing,
        initializers = (temperature = 5.0, saturation_water_ice = 0.5))

    exchange_grid = land.model.grid
    atmosphere = NumericalEarth.PrescribedAtmosphere(exchange_grid;
                                                     surface_layer_height = 10,
                                                     boundary_layer_height = 512)
    fill!(parent(atmosphere.velocities.u), 3)
    fill!(parent(atmosphere.velocities.v), 0)
    fill!(parent(atmosphere.temperature), 288)
    fill!(parent(atmosphere.specific_humidity), 0.005)
    fill!(parent(atmosphere.pressure), 101325)

    α, ϵ = 0.18, 0.95
    radiation = NumericalEarth.PrescribedRadiation(exchange_grid;
                                                   land_surface = SurfaceRadiationProperties(α, ϵ),
                                                   ocean_surface = nothing,
                                                   sea_ice_surface = nothing)
    fill!(parent(radiation.downwelling_shortwave), 400)
    fill!(parent(radiation.downwelling_longwave), 320)

    model = AtmosphereLandModel(atmosphere, land; radiation)
    update_state!(model)

    state = land.model.state
    turbulent = model.interfaces.atmosphere_land_interface.fluxes
    radiative = model.radiation.interface_fluxes.land

    # Downwelling radiation reaches Terrarium's inputs.
    @test all(interior(state.surface_shortwave_down) .≈ 400)
    @test all(interior(state.surface_longwave_down) .≈ 320)

    # Upwelling radiation is pushed from the same `α`, `ϵ`, and `Tₛ` as the coupler's budget.
    @test all(interior(state.surface_shortwave_up) .≈ α * 400)
    @test all(interior(state.surface_longwave_up) .≈ interior(radiative.upwelling_longwave))

    # `G` is the coupler's net surface energy flux (positive upward): turbulent fluxes plus
    # emitted longwave minus absorbed longwave and shortwave.
    G = interior(state.ground_heat_flux)
    expected_G = interior(turbulent.sensible_heat) .+ interior(turbulent.latent_heat) .+
                 interior(radiative.upwelling_longwave) .-
                 interior(radiative.downwelling_longwave) .-
                 interior(radiative.downwelling_shortwave)
    @test all(G .≈ expected_G)

    # The prescribed surface energy balance leaves `G` untouched through a land step.
    G_before = copy(G)
    time_step!(land, 60.0)
    @test interior(state.ground_heat_flux) == G_before
    @test all(isfinite.(interior(state.internal_energy)))
end
