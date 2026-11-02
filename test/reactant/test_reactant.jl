using Test
using Reactant
using Reactant: @trace
using Oceananigans: Oceananigans, CPU, interior, set!
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, Flat, LatitudeLongitudeGrid, Periodic
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: SplitExplicitFreeSurface
using Oceananigans.TimeSteppers: Clock, time_step!, update_state!, first_time_step!
using Oceananigans.Units: minutes
using NumericalEarth
using CUDA
using Statistics: mean

gpu_test = get(ENV, "GPU_TEST", "false") == "true"

if gpu_test
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

@testset "Reactant extension tests" begin
    arch = ReactantState()
    grid = LatitudeLongitudeGrid(arch;
                                 size = (256, 128, 10),
                                 longitude = (0, 360),
                                 latitude = (-80, 80),
                                 halo = (7, 7, 7),
                                 z = (-6000, 0))

    free_surface = SplitExplicitFreeSurface(substeps=10)
    ocean = ocean_simulation(grid; Δt=300, free_surface)

    # We use an idealized atmosphere to avoid downloading the whole JRA55 data
    atmos_grid  = LatitudeLongitudeGrid(arch, Float32; size=(320, 200),
                                                       latitude=(-90, 90),
                                                       longitude=(0, 360),
                                                       topology=(Periodic, Bounded, Flat))

    atmos_times = range(0, 360Oceananigans.Units.days, length=10)
    atmosphere  = PrescribedAtmosphere(atmos_grid, atmos_times)

    coupled_model = OceanOnlyModel(ocean; atmosphere)

    # reconcile_state! initializes the exchanger regridder at construction (via @jit on Reactant).
    exchanger = coupled_model.interfaces.exchanger.atmosphere
    state     = exchanger.state
    regridder = exchanger.regridder
    @test any(regridder.i .!= 0)
    @test any(regridder.j .!= 0)

    # update_state! populates the exchange state with the interpolated air temperature
    @test any(state.T .!= 0)
end

@testset "Reactant land energy flux assembly" begin
    for (temperature, expected_flux) in ((BulkTemperature(), 200),
                                        (SoilSkinTemperature(1.5, 0.05), -150))
        grid = LatitudeLongitudeGrid(ReactantState(); size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        atmosphere = PrescribedAtmosphere(grid, [0.0, 1.0e8])
        land = SlabLand(grid)
        fluxes = SimilarityTheoryFluxes(; solver_stop_criteria=FixedIterations(8))
        model = AtmosphereLandModel(atmosphere, land; atmosphere_land_fluxes=fluxes, atmosphere_land_interface_state=(; temperature, specific_humidity=BulkHumidity()))
        interface = model.interfaces.atmosphere_land_interface
        set!(land; T=290)
        set!(interface.temperature, 295)
        set!(interface.fluxes.sensible_heat, 80)
        set!(interface.fluxes.latent_heat, 120)

        Reactant.@jit NumericalEarth.EarthSystemModels.update_net_fluxes!(model, land)
        @test only(Array(interior(land.fluxes.surface_energy_flux))) ≈ expected_flux
    end
end

# A coupled model whose state is `set!` *after* construction must reconcile that state on the
# first compiled step (issue #403): under `@trace` the `if clock.iteration == 0` guard can't
# branch on the traced iteration, so the Reactant extension no-ops `maybe_prepare_first_time_step!`
# and refreshes through `first_time_step!`. Without the fix the first step drifts by tens of K.
@testset "AtmosphereLandModel first-step reconcile under Reactant (issue #403)" begin
    IC = NumericalEarth.EarthSystemModels.InterfaceComputations

    make_land_grid(arch) = LatitudeLongitudeGrid(arch; latitude = (0, 1), longitude = (0, 1),
                                                 size = (1, 1), topology = (Bounded, Bounded, Flat))

    function build_atmosphere_land_model(arch)
        grid  = make_land_grid(arch)
        atmos = PrescribedAtmosphere(grid, [0.0, 1.0e8])
        set!(atmos.velocities.u, 4)
        set!(atmos.temperature, 290)
        set!(atmos.specific_humidity, 0.004)
        set!(atmos.pressure, 101325)
        update_state!(atmos)
        land = SlabLand(grid)
        # Default atmosphere--land fluxes, but with FixedIterations so the flux solver has a fixed
        # trip count — the default `while`-loop convergence criterion won't trace identically.
        FT = eltype(grid)
        fluxes = IC.SimilarityTheoryFluxes(FT;
                                           stability_functions          = IC.atmosphere_land_stability_functions(FT),
                                           momentum_roughness_length    = 0.1,
                                           temperature_roughness_length = 0.01,
                                           water_vapor_roughness_length = 0.01,
                                           solver_stop_criteria         = IC.FixedIterations(8))
        interface = atmosphere_land_interface(grid, atmos, land; fluxes)
        return AtmosphereLandModel(atmos, land; clock = Clock(grid), atmosphere_land_interface = interface)
    end

    Δt = 10minutes
    N  = 16
    skin_temperature = 320
    land_temperature_mean(model) = mean(interior(model.land.temperature, :, :, 1))

    # Eager CPU reference: `set!` the skin temperature after construction, then step. The first
    # eager `time_step!` reconciles the flux state from the seeded temperature.
    cpu_model = build_atmosphere_land_model(CPU())
    set!(cpu_model.land.temperature, skin_temperature)
    for _ in 1:N
        time_step!(cpu_model, Δt)
    end
    T_cpu = land_temperature_mean(cpu_model)

    # Compiled Reactant run: `first_time_step!` refreshes the flux state and advances one step,
    # then the remaining steps run in a `@trace` loop.
    function run_reactant(model, Δt, n)
        set!(model.land.temperature, skin_temperature)
        first_time_step!(model, Δt)
        @trace track_numbers=false for _ in 1:(n - 1)
            time_step!(model, Δt)
        end
        return land_temperature_mean(model)
    end

    reactant_model = build_atmosphere_land_model(ReactantState())
    compiled_run = Reactant.@compile raise=true raise_first=true sync=true run_reactant(reactant_model, Δt, N)
    T_reactant = Reactant.to_number(compiled_run(reactant_model, Δt, N))

    @test T_reactant ≈ T_cpu rtol=1e-4
end

# Broadcasting into `interior` of a ReactantState field reads device memory element by element, which
# Reactant refuses, so the level pressure goes through `set!`.
@testset "ERA5 pressure_level_field on ReactantState" begin
    levels = Float32[1000, 850, 500] .* 100
    for arch in (CPU(), ReactantState())
        grid = LatitudeLongitudeGrid(arch, Float32; size = (4, 4, 3), longitude = (0, 1), latitude = (0, 1), z = (0, 1))
        pressure = NumericalEarth.DataWrangling.ERA5.pressure_level_field(grid, (; pressure_levels = levels), arch)
        @test all(Array(interior(pressure))[i, j, :] == levels for i in 1:4, j in 1:4)
    end
end
