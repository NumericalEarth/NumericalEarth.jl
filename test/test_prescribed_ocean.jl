include("runtests_setup.jl")

@testset "PrescribedOcean" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Construction on $A" begin
            grid = RectilinearGrid(arch; size = (), topology = (Flat, Flat, Flat))

            ocean = PrescribedOcean(grid, NamedTuple())

            @test ocean isa PrescribedOcean
            @test ocean.grid === grid
            @test ocean.density == 1025.6
            @test ocean.heat_capacity == 3995.6
            @test ocean.clock.time == 0
        end

        @testset "Setting tracer fields on $A" begin
            grid = RectilinearGrid(arch; size = (), topology = (Flat, Flat, Flat))

            ocean = PrescribedOcean(grid, NamedTuple())

            set!(ocean.tracers.T, 15.0)
            set!(ocean.tracers.S, 35.0)

            @allowscalar begin
                @test ocean.tracers.T[1, 1, 1] == 15.0
                @test ocean.tracers.S[1, 1, 1] == 35.0
            end
        end

        @testset "EarthSystemModel interface on $A" begin
            grid = RectilinearGrid(arch; size = (), topology = (Flat, Flat, Flat))

            ocean = PrescribedOcean(grid, NamedTuple())
            set!(ocean.tracers.T, 20.0)
            set!(ocean.tracers.S, 35.0)

            @test NumericalEarth.EarthSystemModels.reference_density(ocean) == 1025.6
            @test NumericalEarth.EarthSystemModels.heat_capacity(ocean) == 3995.6
            @test NumericalEarth.EarthSystemModels.exchange_grid(nothing, ocean, nothing) === grid
            @test NumericalEarth.EarthSystemModels.ocean_temperature(ocean) === ocean.tracers.T
            @test NumericalEarth.EarthSystemModels.ocean_salinity(ocean) === ocean.tracers.S
        end

        @testset "AtmosphereOceanModel coupling on $A" begin
            grid = RectilinearGrid(arch; size = (), topology = (Flat, Flat, Flat))

            ocean = PrescribedOcean(grid, NamedTuple())
            set!(ocean.tracers.T, 15.0)
            set!(ocean.tracers.S, 35.0)

            atmos_grid = RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))
            atmos_times = [0.0, 86400.0]
            atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times)

            parent(atmosphere.velocities.u) .= 10.0
            parent(atmosphere.tracers.T) .= 270.0
            parent(atmosphere.tracers.q) .= 0.005

            radiation = Radiation(arch)
            coupled_model = AtmosphereOceanModel(atmosphere, ocean; radiation)

            @test coupled_model isa NumericalEarth.EarthSystemModels.EarthSystemModel

            # Check that fluxes were computed (non-zero for this state)
            fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
            @allowscalar begin
                Qsens = first(interior(fluxes.sensible_heat))
                Qlat  = first(interior(fluxes.latent_heat))
                @test abs(Qsens) > 0
                @test abs(Qlat)  > 0
            end
        end

        @testset "Time stepping on $A" begin
            grid = RectilinearGrid(arch; size = (), topology = (Flat, Flat, Flat))

            ocean = PrescribedOcean(grid, NamedTuple())
            set!(ocean.tracers.T, 15.0)
            set!(ocean.tracers.S, 35.0)

            atmos_grid = RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))
            atmos_times = [0.0, 86400.0]
            atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times)

            parent(atmosphere.velocities.u) .= 10.0
            parent(atmosphere.tracers.T) .= 270.0
            parent(atmosphere.tracers.q) .= 0.005

            radiation = Radiation(arch)
            coupled_model = AtmosphereOceanModel(atmosphere, ocean; radiation)

            # Time step the coupled model
            Δt = 60.0
            time_step!(coupled_model, Δt)
            @test ocean.clock.time == Δt

            time_step!(coupled_model, Δt)
            @test ocean.clock.time == 2Δt

            # Temperature should be unchanged (prescribed, no timeseries)
            @allowscalar begin
                @test ocean.tracers.T[1, 1, 1] == 15.0
            end
        end

        @testset "OceanOnlyModel guard on $A" begin
            grid = RectilinearGrid(arch; size = (), topology = (Flat, Flat, Flat))

            ocean = PrescribedOcean(grid, NamedTuple())
            @test_throws ArgumentError OceanOnlyModel(ocean)
        end
    end
end
