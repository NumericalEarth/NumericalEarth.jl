include("runtests_setup.jl")

using Glob
using Oceananigans.OutputReaders: OnDisk, InMemory
using Oceananigans.OutputWriters: Checkpointer
using Oceananigans.TimeSteppers: reset!
using NumericalEarth.EarthSystemModels: components, AbstractPrescribedComponent

function make_coupled_model(grid)
    @inline hi(λ, φ) = φ > 70 || φ < -70

    ocean = ocean_simulation(grid, closure=nothing)
    set!(ocean.model, T=20, S=35, u=0.01, v=-0.005)

    sea_ice = sea_ice_simulation(grid, ocean)
    set!(sea_ice.model, h=hi, ℵ=hi)

    arch = architecture(grid)
    atmosphere = synthetic_prescribed_atmosphere(arch)
    land = synthetic_prescribed_land(arch)
    radiation = synthetic_prescribed_radiation(arch)

    return OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation)
end

function test_clock_time_and_iteration(simulation, expected_iteration)
    Δt = simulation.Δt
    @test simulation.model.clock.iteration == expected_iteration
    @test simulation.model.clock.time == expected_iteration * Δt
    for component in components(simulation.model)
        if component isa AbstractPrescribedComponent
            @test component.clock.iteration == expected_iteration
            @test component.clock.time == expected_iteration * Δt
        else
            @test component.model.clock.iteration == expected_iteration
            @test component.model.clock.time == expected_iteration * Δt
        end
    end
end

@testset "EarthSystemModel checkpointing" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing EarthSystemModel checkpointing on $A"

        # Create a minimal grid
        grid = LatitudeLongitudeGrid(arch;
                                     size = (20, 20, 4),
                                     z = (-100, 0),
                                     latitude = (-80, 80),
                                     longitude = (0, 360),
                                     halo = (6, 6, 6))

        # Reference run: 3 iterations, then continue to 6
        # (This matches the checkpointed workflow where we create a new Simulation
        # after iteration 3, which is what happens during checkpoint restore)
        model = make_coupled_model(grid)
        simulation = Simulation(model, Δt=60, stop_iteration=3, verbose=false)
        run!(simulation)

        # Continue on the same model (simulates what happens after checkpoint restore)
        simulation = Simulation(model, Δt=60, stop_iteration=6, verbose=false)
        run!(simulation)

        # Store reference states at iteration 6
        ref_T  = Array(interior(model.ocean.model.tracers.T))
        ref_S  = Array(interior(model.ocean.model.tracers.S))
        ref_u  = Array(interior(model.ocean.model.velocities.u))
        ref_v  = Array(interior(model.ocean.model.velocities.v))
        ref_h  = Array(interior(model.sea_ice.model.ice_thickness))
        ref_ui = Array(interior(model.sea_ice.model.velocities.u))
        ref_vi = Array(interior(model.sea_ice.model.velocities.v))
        ref_time = model.clock.time
        ref_iteration = model.clock.iteration

        # Checkpointed run: 3 iterations, then checkpoint
        model = make_coupled_model(grid)
        simulation = Simulation(model, Δt=60, stop_iteration=3, verbose=false)

        prefix = "osm_checkpointer_test_$(typeof(arch))"
        simulation.output_writers[:checkpointer] = Checkpointer(simulation.model;
                                                                schedule = IterationInterval(3),
                                                                prefix = prefix)

        run!(simulation)

        @test isfile("$(prefix)_iteration3.jld2")

        # Create new model and restore from checkpoint
        model = make_coupled_model(grid)
        simulation = Simulation(model, Δt=60, stop_iteration=6, verbose=false)

        simulation.output_writers[:checkpointer] = Checkpointer(model;
                                                                schedule = IterationInterval(3),
                                                                prefix = prefix)

        set!(simulation; checkpoint=:latest)

        test_clock_time_and_iteration(simulation, 3)

        set!(simulation; iteration=3)

        test_clock_time_and_iteration(simulation, 3)

        run!(simulation)

        # Compare final states
        T  = Array(interior(model.ocean.model.tracers.T))
        S  = Array(interior(model.ocean.model.tracers.S))
        u  = Array(interior(model.ocean.model.velocities.u))
        v  = Array(interior(model.ocean.model.velocities.v))
        h  = Array(interior(model.sea_ice.model.ice_thickness))
        ui = Array(interior(model.sea_ice.model.velocities.u))
        vi = Array(interior(model.sea_ice.model.velocities.v))

        # Checkpoint restore produces results within numerical precision of the iterative solvers.
        # The split-explicit (ocean) and EVP (sea ice) solvers accumulate floating point
        # differences during substepping, even with identical initial conditions.
        @test T ≈ ref_T rtol=1e-13
        @test S ≈ ref_S rtol=1e-13
        @test h ≈ ref_h rtol=1e-13
        @test u ≈ ref_u rtol=1e-10  # split-explicit solver precision
        @test v ≈ ref_v rtol=1e-10  # split-explicit solver precision
        @test ui ≈ ref_ui rtol=1e-10  # EVP solver precision
        @test vi ≈ ref_vi rtol=1e-10  # EVP solver precision
        @test model.clock.time == ref_time
        @test model.clock.iteration == ref_iteration

        # Cleanup
        rm.(glob("$(prefix)_iteration*.jld2"), force=true)
    end
end

@testset "Checkpointing with windowed atmosphere time series" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing checkpointing with windowed atmosphere time series on $A"

        grid = LatitudeLongitudeGrid(arch;
                                     size = (10, 10, 4),
                                     z = (-100, 0),
                                     latitude = (30, 40),
                                     longitude = (-70, -60),
                                     halo = (7, 7, 7))

        times = 0:60:600.0
        directory = mktempdir()

        # u, v, T and p hold 3 of the 11 times in memory; q holds all 11.
        for (name, value) in ((:u, t -> 5), (:v, t -> 1), (:T, t -> 290 + t / 60), (:p, t -> 101325))
            fts = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend = OnDisk(),
                                                           path = joinpath(directory, "$name.jld2"),
                                                           name = string(name))
            field = Field{Center, Center, Nothing}(grid)
            for n in eachindex(times)
                set!(field, value(times[n]))
                set!(fts, field, n)
            end
        end

        windowed(name) = FieldTimeSeries(joinpath(directory, "$name.jld2"), string(name);
                                         backend = InMemory(3), architecture = arch)

        function make_model()
            ocean = ocean_simulation(grid, closure=nothing)
            set!(ocean.model, T=20, S=35)

            q = FieldTimeSeries{Center, Center, Nothing}(grid, times)
            for n in eachindex(times)
                set!(q[n], 0.008 + times[n] / 6e5)
            end

            atmosphere = PrescribedAtmosphere(grid, times;
                                              velocities = (u = windowed(:u), v = windowed(:v)),
                                              temperature = windowed(:T),
                                              specific_humidity = q,
                                              pressure = windowed(:p))

            return OceanOnlyModel(ocean; atmosphere, radiation = PrescribedRadiation(grid))
        end

        # The windows of u, v, T and p start at time index 3 in the uninterrupted run and at
        # index 4 after restoring at iteration 4.
        model = make_model()
        simulation = Simulation(model, Δt=60, stop_iteration=4, verbose=false)
        run!(simulation)
        simulation = Simulation(model, Δt=60, stop_iteration=6, verbose=false)
        run!(simulation)

        ref_T = Array(interior(model.ocean.model.tracers.T))
        ref_S = Array(interior(model.ocean.model.tracers.S))

        prefix = "windowed_atmosphere_checkpointer_test_$(typeof(arch))"

        model = make_model()
        simulation = Simulation(model, Δt=60, stop_iteration=4, verbose=false)
        simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = IterationInterval(4), prefix)
        run!(simulation)

        model = make_model()
        simulation = Simulation(model, Δt=60, stop_iteration=6, verbose=false)
        simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = IterationInterval(4), prefix)
        set!(simulation; checkpoint=:latest)
        run!(simulation)

        @test Array(interior(model.ocean.model.tracers.T)) ≈ ref_T rtol=1e-13
        @test Array(interior(model.ocean.model.tracers.S)) ≈ ref_S rtol=1e-13

        rm.(glob("$(prefix)_iteration*.jld2"), force=true)
    end
end

@testset "EarthSystemModel reset! and re-run!" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing EarthSystemModel reset! and re-run! on $A"

        grid = LatitudeLongitudeGrid(arch;
                                     size = (20, 20, 4),
                                     z = (-100, 0),
                                     latitude = (-80, 80),
                                     longitude = (0, 360),
                                     halo = (6, 6, 6))

        model = make_coupled_model(grid)
        simulation = Simulation(model; Δt=60, stop_iteration=3, verbose=false)

        # check that clock stops when intended
        @test model.atmosphere isa AbstractPrescribedComponent
        @test model.land isa AbstractPrescribedComponent
        @test model.radiation isa AbstractPrescribedComponent

        # check that clock stops when intended
        run!(simulation)

        test_clock_time_and_iteration(simulation, 3)

        # check that reset!(simulation) rewinds model clock and its components
        reset!(simulation)

        test_clock_time_and_iteration(simulation, 0)

        # check we can restart the simulation
        simulation.stop_iteration = 2
        run!(simulation)

        test_clock_time_and_iteration(simulation, 2)
    end
end
