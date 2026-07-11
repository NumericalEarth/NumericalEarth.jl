using Test
using Reactant
using Reactant: @trace
using Oceananigans: Oceananigans, CPU, interior, set!
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, Flat, LatitudeLongitudeGrid, Periodic
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: SplitExplicitFreeSurface
using Oceananigans.TimeSteppers: time_step!, update_state!, first_time_step!
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

# The exchanger regridder and construction-time flux state are set up when the model is built,
# but a coupled model whose prognostic state is `set!` *after* construction must reconcile that
# state on the first compiled step (issue #403). Under `@trace` the generic
# `if clock.iteration == 0` first-step guard cannot branch on the traced iteration, so the
# coupled-model Reactant extension makes `maybe_prepare_first_time_step!` a no-op and routes the
# refresh through `first_time_step!`. This checks that a compiled run using `first_time_step!`
# reproduces the eager CPU result after a post-construction `set!` of the skin temperature —
# without the fix the first compiled step consumes the pre-`set!` flux state and drifts by tens
# of kelvin.
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
        # These are the default atmosphere--land fluxes except for FixedIterations, which
        # gives the flux solver a fixed trip count so the compiled and eager paths trace
        # identically — the default convergence criterion is a data-dependent `while` loop
        # that would not.
        FT = eltype(grid)
        fluxes = IC.SimilarityTheoryFluxes(FT;
                                           stability_functions          = IC.atmosphere_land_stability_functions(FT),
                                           momentum_roughness_length    = 0.1,
                                           temperature_roughness_length = 0.01,
                                           water_vapor_roughness_length = 0.01,
                                           solver_stop_criteria         = IC.FixedIterations(8))
        interface = atmosphere_land_interface(grid, atmos, land; fluxes)
        return AtmosphereLandModel(atmos, land; atmosphere_land_interface = interface)
    end

    Δt = 10minutes
    N  = 16
    skin_temperature = 320
    land_temperature_mean(model) = mean(interior(model.land.temperature, :, :, 1))

    # Eager CPU reference: `set!` the skin temperature after construction, then step. The first
    # eager `time_step!` reconciles the flux state from the seeded temperature.
    cpu_model = build_atmosphere_land_model(CPU())
    parent(cpu_model.land.temperature) .= skin_temperature
    for _ in 1:N
        time_step!(cpu_model, Δt)
    end
    T_cpu = land_temperature_mean(cpu_model)

    # Compiled Reactant run: `first_time_step!` refreshes the flux state and advances one step,
    # then the remaining steps run in a `@trace` loop.
    function run_reactant(model, Δt, n)
        parent(model.land.temperature) .= skin_temperature
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
