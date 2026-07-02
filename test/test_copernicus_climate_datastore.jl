include("runtests_setup.jl")

using CopernicusClimateDataStore
using Dates

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5YearlySingleLevel, ERA5MonthlySingleLevel,
                                          ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                          ERA5Metadata, ERA5Metadatum

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

    @testset "New ERA5 dataset types" begin
        # Test ERA5YearlySingleLevel
        yearly_dataset = ERA5YearlySingleLevel()
        @test yearly_dataset isa ERA5.ERA5Dataset

        # Test ERA5MonthlySingleLevel
        monthly_dataset = ERA5MonthlySingleLevel()
        @test monthly_dataset isa ERA5.ERA5Dataset

        # Test ERA5HourlyPressureLevels
        pressure_levels = [100000.0, 85000.0, 50000.0]  # Pa
        hourly_pl = ERA5HourlyPressureLevels(pressure_levels)
        @test hourly_pl isa ERA5.ERA5Dataset
        @test hourly_pl.pressure_levels == pressure_levels

        # Test ERA5MonthlyPressureLevels
        monthly_pl = ERA5MonthlyPressureLevels(pressure_levels)
        @test monthly_pl isa ERA5.ERA5Dataset
        @test monthly_pl.pressure_levels == pressure_levels
    end

    @testset "Download methods for new dataset types" begin
        # Test ERA5YearlySingleLevel
        date = DateTime(2020, 1, 1)
        metadatum = Metadatum(:temperature; dataset=ERA5YearlySingleLevel(), date)
        @test hasmethod(Downloads.download, Tuple{typeof(metadatum)})

        # Test ERA5MonthlySingleLevel
        metadatum = Metadatum(:temperature; dataset=ERA5MonthlySingleLevel(), date)
        @test hasmethod(Downloads.download, Tuple{typeof(metadatum)})

        # Test ERA5HourlyPressureLevels
        metadatum = Metadatum(:temperature; dataset=ERA5HourlyPressureLevels([100000.0]), date)
        @test hasmethod(Downloads.download, Tuple{typeof(metadatum)})

        # Test ERA5MonthlyPressureLevels
        metadatum = Metadatum(:temperature; dataset=ERA5MonthlyPressureLevels([100000.0]), date)
        @test hasmethod(Downloads.download, Tuple{typeof(metadatum)})
    end

    @testset "Helper function dispatch" begin
        # Test variable_name_mapping
        @test isdefined(CDSExt, :variable_name_mapping)

        # Test pressure_levels extraction
        @test isdefined(CDSExt, :pressure_levels)
        pl_dataset = ERA5HourlyPressureLevels([100000.0, 50000.0])
        @test CDSExt.pressure_levels(pl_dataset) == [100000.0, 50000.0]
        @test isnothing(CDSExt.pressure_levels(ERA5YearlySingleLevel()))

        # Test date_keywords
        @test isdefined(CDSExt, :date_keywords)
        date = DateTime(2020, 6, 15, 12)

        # Yearly
        kw = CDSExt.date_keywords(ERA5YearlySingleLevel(), date)
        @test kw.years == 2020

        # Monthly
        kw = CDSExt.date_keywords(ERA5MonthlySingleLevel(), date)
        @test kw.year == 2020
        @test kw.month == 6

        # Hourly pressure levels
        kw = CDSExt.date_keywords(ERA5HourlyPressureLevels([100000.0]), date)
        @test kw.startyear == 2020
        @test kw.months == 6
        @test kw.days == 15
        @test kw.hours == 12

        # Monthly pressure levels
        kw = CDSExt.date_keywords(ERA5MonthlyPressureLevels([100000.0]), date)
        @test kw.year == 2020
        @test kw.month == 6

        # Test cds_download_function
        @test isdefined(CDSExt, :cds_download_function)
        @test CDSExt.cds_download_function(ERA5YearlySingleLevel()) == CopernicusClimateDataStore.yearly
        @test CDSExt.cds_download_function(ERA5MonthlySingleLevel()) == CopernicusClimateDataStore.monthly
        @test CDSExt.cds_download_function(ERA5HourlyPressureLevels([100000.0])) == CopernicusClimateDataStore.hourly
        @test CDSExt.cds_download_function(ERA5MonthlyPressureLevels([100000.0])) == CopernicusClimateDataStore.monthly
    end

    @info "✓ CopernicusClimateDataStore extension tests passed"
end
