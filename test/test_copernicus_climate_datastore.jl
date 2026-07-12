include("runtests_setup.jl")

using CopernicusClimateDataStore
using Dates

import Downloads

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5HourlyPressureLevels,
                                         ERA5Metadata, ERA5Metadatum, hPa

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

    @testset "Download dispatch and skip_existing" begin
        dataset = ERA5HourlySingleLevel()

        mktempdir() do dir
            # Single timestep: a pre-existing file makes skip_existing return without downloading
            date = DateTime(2020, 1, 1, 0)
            metadatum = Metadatum(:temperature; dataset, date, dir)
            output_path = joinpath(dir, NumericalEarth.DataWrangling.metadata_filename(metadatum))
            touch(output_path)
            @test Downloads.download(metadatum; skip_existing=true) == output_path

            # Collection: the batched download finds every per-datetime file present
            # and returns their paths without submitting a request
            dates = DateTime(2020, 1, 1, 0):Hour(1):DateTime(2020, 1, 1, 2)
            metadata = Metadata(:temperature; dataset, dates, dir)
            for m in metadata
                touch(joinpath(dir, NumericalEarth.DataWrangling.metadata_filename(m)))
            end
            paths = Downloads.download(metadata; skip_existing=true)
            @test length(paths) == length(metadata)
            @test all(isfile, paths)

            # MetadataSet: the multi-variable batched path sees every file present
            # and skips the request too
            mset = MetadataSet(:temperature, :eastward_velocity; dataset, dates, dir)
            for name in keys(mset), m in mset[name]
                touch(joinpath(dir, NumericalEarth.DataWrangling.metadata_filename(m)))
            end
            paths = Downloads.download(mset; skip_existing=true)
            @test length(paths) == 2length(dates)
            @test all(isfile, paths)
        end
    end

    @testset "era5cli_levels" begin
        pl = ERA5HourlyPressureLevels(pressure_levels=[500, 850]hPa)
        sl = ERA5HourlySingleLevel()

        # Pressure-level datasets pass their levels in hPa, sorted descending by the constructor
        @test CDSExt.era5cli_levels(pl, "temperature") == [850, 500]

        # The single-level geopotential is ambiguous on CDS; :surface disambiguates it
        @test CDSExt.era5cli_levels(sl, "geopotential") == :surface
        @test CDSExt.era5cli_levels(sl, "2m_temperature") === nothing
    end

    @testset "era5cli_request_area pads by two native cells" begin
        sl = ERA5HourlySingleLevel()
        bbox = NumericalEarth.DataWrangling.BoundingBox(
            longitude = (0, 10),
            latitude = (40, 50)
        )

        @test isnothing(CDSExt.era5cli_request_area(nothing, sl, :temperature))

        # Atmospheric single-level variables live on the 0.25° grid → 0.5° margin
        area = CDSExt.era5cli_request_area(bbox, sl, :temperature)
        @test area.lon == (-0.5, 10.5)
        @test area.lat == (39.5, 50.5)

        # The margin never pushes latitude beyond the poles
        polar = NumericalEarth.DataWrangling.BoundingBox(
            longitude = (0, 10),
            latitude = (85, 90)
        )
        area = CDSExt.era5cli_request_area(polar, sl, :temperature)
        @test area.lat == (84.5, 90.0)
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
