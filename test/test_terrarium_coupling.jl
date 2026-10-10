include("runtests_setup.jl")

using Terrarium
using Oceananigans.TimeSteppers: update_state!, time_step!
using Oceananigans.Fields: interior

# Couple a Terrarium `LandModel` into NumericalEarth's `EarthSystemModel`, with the surface
# turbulent fluxes computed by NumericalEarth (Monin-Obukhov) and consumed by Terrarium.
# Requires `NumericalEarthTerrariumExt` (loaded automatically when Terrarium is present) and
# a Terrarium build that provides the prescribed-skin-temperature `solve_surface_energy_balance!`.
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

    # Advancing the coupled model integrates the soil and populates the ground heat flux
    # residual (nonzero once a land step has run).
    for _ in 1:3
        time_step!(model, 60.0)
    end
    @test all(isfinite.(interior(state.internal_energy)))
    @test all(isfinite.(interior(state.ground_heat_flux)))
    @test any(abs.(Array(interior(state.ground_heat_flux))) .> 0)
end
