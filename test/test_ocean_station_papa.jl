include("runtests_setup.jl")

# Exercises the OceanStationPapa worked example in docs/src/developers/,
# which shows how a station dataset uses `region = Column(...)` to plug
# into the generic Metadata pipeline without per-dataset `native_grid` or
# `set!` overrides. Including the example script here guarantees it stays
# runnable as the API evolves; any breakage surfaces in CI rather than at
# doc-build time.

const OSP_EXAMPLE_PATH = joinpath(@__DIR__, "..", "docs", "src", "developers", "ocean_station_papa.jl")

module OceanStationPapaExampleModule
    const SCRIPT = joinpath(@__DIR__, "..", "docs", "src", "developers", "ocean_station_papa.jl")
    include(SCRIPT)
end

using .OceanStationPapaExampleModule: OceanStationPapa, OSP_LONGITUDE,
                                       OSP_LATITUDE, OSP_VARIABLE_NAMES
using NumericalEarth.DataWrangling: test_dataset_contract, is_conforming,
                                    Metadatum, Column

@testset "OceanStationPapa worked example" begin

    @testset "conformance" begin
        report = test_dataset_contract(OceanStationPapa(); verbose=false)
        @test is_conforming(report)

        # dataset_url should be defined
        u = only(filter(c -> c.name === :dataset_url, report.checks))
        @test u.status === :ok
    end

    @testset "round-trip 3-D profile" begin
        col = Column(OSP_LONGITUDE, OSP_LATITUDE)
        md = Metadatum(:temperature; dataset=OceanStationPapa(),
                       region=col, date=DateTime(2012, 10, 1))
        native = NumericalEarth.DataWrangling.Field(md, CPU())
        @test size(native, 3) >= 10

        target_grid = RectilinearGrid(size=20, x=OSP_LONGITUDE, y=OSP_LATITUDE,
                                       z=(-200, 0),
                                       topology=(Flat, Flat, Bounded),
                                       halo=(3,))
        target = Field{Center, Center, Center}(target_grid)
        set!(target, md)
        v = Array(interior(target, 1, 1, :))
        @test length(v) == 20
        @test all(isfinite, v)
        # Oct 1 2012 temperatures ~4-12 °C
        @test all(-2 <= x <= 20 for x in v)
    end

    @testset "round-trip surface scalar" begin
        col = Column(OSP_LONGITUDE, OSP_LATITUDE)
        md = Metadatum(:sea_level_pressure; dataset=OceanStationPapa(),
                       region=col, date=DateTime(2012, 10, 1))
        target_grid = RectilinearGrid(size=1, x=OSP_LONGITUDE, y=OSP_LATITUDE,
                                       z=(-1, 0),
                                       topology=(Flat, Flat, Bounded),
                                       halo=(3,))
        target = Field{Center, Center, Center}(target_grid)
        set!(target, md)
        v = first(interior(target))
        # Millibar → Pa conversion should have been applied
        @test 80_000 < v < 120_000
    end
end
