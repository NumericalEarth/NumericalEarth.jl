include("runtests_setup.jl")

# Exercises the OceanStationPapa worked example in docs/src/developers/, which
# demonstrates the StationColumn extension API from first principles.
# Including the example script here guarantees it stays runnable as the API
# evolves; any breakage surfaces in CI rather than at doc-build time.

const OSP_EXAMPLE_PATH = joinpath(@__DIR__, "..", "docs", "src", "developers", "ocean_station_papa.jl")

module OceanStationPapaExampleModule
    const SCRIPT = joinpath(@__DIR__, "..", "docs", "src", "developers", "ocean_station_papa.jl")
    include(SCRIPT)
end

using .OceanStationPapaExampleModule: OceanStationPapa, OSP_LONGITUDE,
                                       OSP_LATITUDE, OSP_VARIABLE_NAMES
using NumericalEarth.DataWrangling: test_dataset_contract, is_conforming, Metadatum

@testset "OceanStationPapa worked example" begin

    @testset "conformance" begin
        report = test_dataset_contract(OceanStationPapa(); verbose=false)
        @test is_conforming(report)

        # StationColumn should be declared (not default)
        sl = only(filter(c -> c.name === :spatial_layout, report.checks))
        @test sl.status === :ok
        @test occursin("StationColumn", sl.detail)

        # dataset_url should be defined
        u = only(filter(c -> c.name === :dataset_url, report.checks))
        @test u.status === :ok
    end

    @testset "round-trip 3-D profile" begin
        md = Metadatum(:temperature; dataset=OceanStationPapa(), date=DateTime(2012, 10, 1))
        native = NumericalEarth.DataWrangling.Field(md, CPU())
        @test size(native)[1:2] == (1, 1)
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
        md = Metadatum(:sea_level_pressure; dataset=OceanStationPapa(),
                       date=DateTime(2012, 10, 1))
        target_grid = RectilinearGrid(size=(), topology=(Flat, Flat, Flat))
        target = Field{Center, Center, Center}(target_grid)
        set!(target, md)
        v = first(interior(target))
        # Millibar → Pa conversion should have been applied
        @test 80_000 < v < 120_000
    end
end
