include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: adopt_clock

@testset "VERIFY clock consistency (PrescribedOcean fix)" begin
    include("test_clock_consistency.jl")
end

@testset "VERIFY reclock shares land fields and coerces clock" begin
    arch = CPU()
    grid = LatitudeLongitudeGrid(arch, Float32;
                                 size = 1,
                                 latitude = 10,
                                 longitude = 10,
                                 z = (-1, 0),
                                 topology = (Flat, Flat, Bounded))

    rivers = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0])
    land = PrescribedLand((; rivers); clock = Clock{Float32}(time=0))
    @test typeof(land.clock.time) === Float32

    model_clock = Clock{Float64}(time=0)
    land64 = adopt_clock(land, model_clock)

    @test land64 isa PrescribedLand
    @test typeof(land64.clock.time) === Float64
    # The exact invariants the surface-fluxes test now relies on:
    @test land64.freshwater_flux === land.freshwater_flux
    @test typeof(land64.clock.time) === typeof(model_clock.time)

    # And a matching clock is a no-op (identity preserved).
    land_ok = PrescribedLand((; rivers); clock = Clock{Float64}(time=0))
    @test adopt_clock(land_ok, model_clock) === land_ok
end
