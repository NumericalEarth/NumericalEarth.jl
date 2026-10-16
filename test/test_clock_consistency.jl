include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: adopt_clock
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.Oceans: PrescribedOcean, SlabOcean

@testset "Clock type consistency across components" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing component clock consistency on $A..."

        grid = LatitudeLongitudeGrid(arch,
                                     size = (8, 8, 1),
                                     z = (-100, 0),
                                     latitude = (-80, 80),
                                     longitude = (0, 360),
                                     halo = (6, 6, 3))

        model_clock = Clock{Float64}(time=0)

        # Fallbacks: `nothing` and an unrecognized component are left untouched
        # (the latter is the SpeedyWeather/Veros path, which tracks time internally).
        @test adopt_clock(nothing, model_clock) === nothing
        @test adopt_clock(42, model_clock) === 42

        # A prescribed component with a mismatched (Float32) clock is coerced, with a warning.
        atmos32 = PrescribedAtmosphere(grid; clock = Clock{Float32}(time=0))
        @test typeof(atmos32.clock.time) === Float32
        atmos64 = @test_logs (:warn,) adopt_clock(atmos32, model_clock)
        @test atmos64 isa PrescribedAtmosphere
        @test typeof(atmos64.clock.time) === Float64
        # Fields other than the clock are shared by reference, not rebuilt.
        @test atmos64.velocities === atmos32.velocities
        @test atmos64.grid === atmos32.grid

        # A matching clock is a no-op: the same object is returned and nothing is logged.
        atmos_ok = PrescribedAtmosphere(grid; clock = Clock{Float64}(time=0))
        @test (@test_logs adopt_clock(atmos_ok, model_clock)) === atmos_ok

        # PrescribedOcean and SlabOcean carry a separate `FT` parameter that is not
        # field-inferable; their coercion must preserve it while changing the clock type.
        pocean32 = PrescribedOcean(grid; FT=Float32, clock = Clock{Float32}(time=0))
        pocean64 = adopt_clock(pocean32, model_clock)
        @test pocean64 isa PrescribedOcean
        @test typeof(pocean64.clock.time) === Float64

        slab32 = SlabOcean(grid; clock = Clock{Float32}(time=0))
        slab64 = adopt_clock(slab32, model_clock)
        @test slab64 isa SlabOcean
        @test typeof(slab64.clock.time) === Float64
        @test eltype(slab64.temperature) === eltype(slab32.temperature)

        # A `Simulation` clock type is fixed by its grid: a match passes, a mismatch errors.
        ocean = ocean_simulation(grid, closure=nothing)
        @test adopt_clock(ocean, model_clock) === ocean
        @test_throws ArgumentError adopt_clock(ocean, Clock{Float32}(time=0))

        # The EarthSystemModel enforces its (authoritative) clock type on every component,
        # so a mismatched component cannot survive construction and time stepping proceeds.
        set!(ocean.model, T=20, S=35)
        atmos = PrescribedAtmosphere(grid; clock = Clock{Float32}(time=0))
        model = OceanOnlyModel(ocean; atmosphere=atmos)

        @test typeof(model.clock.time) === Float64
        @test typeof(model.atmosphere.clock.time) === Float64
        @test typeof(model.ocean.model.clock.time) === Float64

        time_step!(model, 60)
        @test model.clock.iteration == 1
        @test typeof(model.atmosphere.clock.time) === Float64
    end
end
