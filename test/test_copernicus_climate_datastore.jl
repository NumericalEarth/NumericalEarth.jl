include("runtests_setup.jl")

using CopernicusClimateDataStore
using Dates

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5Metadata, ERA5Metadatum

# Internal extension module
const CDSExt = Base.get_extension(NumericalEarth, :NumericalEarthCopernicusClimateDataStoreExt)

@testset "CopernicusClimateDataStore extension" begin
    @info "Testing CopernicusClimateDataStore extension loading..."

    @testset "Extension is loaded" begin
        @test !isnothing(CDSExt)
    end

    @testset "Downloads.download methods are defined" begin
        # Test that the extension defines Downloads.download for ERA5Metadata/Metadatum types
        dataset = ERA5HourlySingleLevel()
        date = DateTime(2020, 1, 1, 0)

        # Create a metadatum (single timestep)
        metadatum = Metadatum(:temperature; dataset, date)

        # Check that Downloads.download method exists for ERA5Metadatum
        @test hasmethod(Downloads.download, Tuple{typeof(metadatum)})

        # Create metadata (multiple timesteps)
        dates = DateTime(2020, 1, 1, 0):Hour(1):DateTime(2020, 1, 1, 2)
        metadata = Metadata(:temperature; dataset, dates)

        # Check that Downloads.download method exists for ERA5Metadata
        @test hasmethod(Downloads.download, Tuple{typeof(metadata)})
    end

    @testset "Area builder utilities" begin
        # Test that the bounding box area builder is accessible
        @test isdefined(CDSExt, :build_era5_area)

        # Test with nothing
        @test isnothing(CDSExt.build_era5_area(nothing))

        # Test with a bounding box
        bbox = NumericalEarth.DataWrangling.BoundingBox(
            longitude = (0, 10),
            latitude = (40, 50)
        )
        area = CDSExt.build_era5_area(bbox)
        @test !isnothing(area)
        @test haskey(area, :lat)
        @test haskey(area, :lon)
        @test area.lat == (40, 50)
        @test area.lon == (0, 10)
    end

    @info "✓ CopernicusClimateDataStore extension tests passed"
end
