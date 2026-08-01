include("runtests_setup.jl")

using CopernicusClimateDataStore
using Dates
import Downloads

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5YearlySingleLevel, ERA5MonthlySingleLevel,
                                          ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                          ERA5HourlyLand, ERA5MonthlyLand,
                                          ERA5Land_dataset_variable_names, ERA5Land_netcdf_variable_names,
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
        @test length(area) == 4
        @test area[1] == 40   # south
        @test area[2] == 0    # west
        @test area[3] == 50   # north
        @test area[4] == 10   # east
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

    @testset "ERA5-Land dataset types" begin
        hourly_land  = ERA5HourlyLand()
        monthly_land = ERA5MonthlyLand()
        @test hourly_land  isa ERA5.ERA5Dataset
        @test monthly_land isa ERA5.ERA5Dataset

        supported = NumericalEarth.DataWrangling.supported_datasets()
        @test ERA5HourlyLand  in supported
        @test ERA5MonthlyLand in supported

        # 0.1° global grid: 3600 lon × 1800 lat cells (file's 1801 latitude rows
        # fold to 1800 cells via AverageNorthSouth mangling, same as single-level).
        @test Base.size(hourly_land,  :skin_temperature) == (3600, 1800, 1)
        @test Base.size(monthly_land, :skin_temperature) == (3600, 1800, 1)

        # Region-cropped Metadatum: exercise the same construction path a real
        # download would use, not just a global-date metadatum.
        region = NumericalEarth.DataWrangling.BoundingBox(longitude=(113, 115), latitude=(0.5, 2.5))
        metadatum = Metadatum(:skin_temperature; dataset=hourly_land, region, date=DateTime(2020, 1, 1))
        @test NumericalEarth.DataWrangling.is_three_dimensional(metadatum) == false

        # Interfaces: half a 0.1° cell offset, global latitude coverage.
        # Both axes must resolve to exactly 0.1° — Ny=1801 would break the latitude one.
        lon_i = NumericalEarth.DataWrangling.longitude_interfaces(metadatum)
        lat_i = NumericalEarth.DataWrangling.latitude_interfaces(metadatum)
        @test lon_i == (-0.05, 359.95)
        @test lat_i == (-90, 90)
        @test (lon_i[2] - lon_i[1]) / 3600 ≈ 0.1
        @test (lat_i[2] - lat_i[1]) / 1800 ≈ 0.1

        # The file carries one extra latitude row (1801) that folds into the 1800
        # grid cells through AverageNorthSouth — the same convention as single levels.
        @test NumericalEarth.DataWrangling.mangling_for(metadatum, 1801) isa
              NumericalEarth.DataWrangling.AverageNorthSouth

        # Stored north-to-south, like single levels (inherited).
        @test NumericalEarth.DataWrangling.reversed_latitude_axis(hourly_land) == true

        # API-name and netcdf-name dicts cover the same variable set
        @test keys(ERA5Land_dataset_variable_names) == keys(ERA5Land_netcdf_variable_names)

        # All four soil temperature levels and soil water layers are present.
        for n in 1:4
            @test haskey(ERA5Land_dataset_variable_names, Symbol("soil_temperature_level_$n"))
            @test haskey(ERA5Land_dataset_variable_names, Symbol("volumetric_soil_water_layer_$n"))
        end

        # CDS catalogue names.
        @test ERA5Land_dataset_variable_names[:skin_temperature] == "skin_temperature"
        @test ERA5Land_dataset_variable_names[:soil_temperature_level_1] == "soil_temperature_level_1"
        @test ERA5Land_dataset_variable_names[:volumetric_soil_water_layer_4] == "volumetric_soil_water_layer_4"
        @test ERA5Land_dataset_variable_names[:temperature] == "2m_temperature"
        @test ERA5Land_dataset_variable_names[:dewpoint_temperature] == "2m_dewpoint_temperature"
        @test ERA5Land_dataset_variable_names[:snow_water_equivalent] == "snow_depth_water_equivalent"
        @test ERA5Land_dataset_variable_names[:snow_depth] == "snow_depth"

        # NetCDF short names (verified against a real ERA5-Land download).
        @test ERA5Land_netcdf_variable_names[:skin_temperature] == "skt"
        @test ERA5Land_netcdf_variable_names[:soil_temperature_level_3] == "stl3"
        @test ERA5Land_netcdf_variable_names[:volumetric_soil_water_layer_2] == "swvl2"
        @test ERA5Land_netcdf_variable_names[:temperature] == "t2m"
        @test ERA5Land_netcdf_variable_names[:dewpoint_temperature] == "d2m"
        @test ERA5Land_netcdf_variable_names[:snow_water_equivalent] == "sd"
        @test ERA5Land_netcdf_variable_names[:snow_depth] == "sde"

        # available_variables returns the API-name dict; dataset_variable_name
        # returns the netcdf short name — same swap risk as single-level ERA5
        @test NumericalEarth.DataWrangling.available_variables(hourly_land) === ERA5Land_dataset_variable_names
        @test NumericalEarth.DataWrangling.dataset_variable_name(metadatum) == "skt"

        # No unit conversion or inpainting for ERA5-Land (masked ocean cells must stay masked)
        @test NumericalEarth.DataWrangling.conversion_units(metadatum) === nothing
        @test NumericalEarth.DataWrangling.default_inpainting(metadatum) === nothing

        # Data availability ranges
        hourly_dates = NumericalEarth.DataWrangling.all_dates(hourly_land, :skin_temperature)
        @test first(hourly_dates) == DateTime("1950-01-01")
        @test step(hourly_dates) == Hour(1)

        monthly_dates = NumericalEarth.DataWrangling.all_dates(monthly_land, :skin_temperature)
        @test first(monthly_dates) == DateTime("1950-01-01")
        @test step(monthly_dates) == Month(1)
    end

    @testset "ERA5-Land download dispatch" begin
        date = DateTime(2020, 1, 1)

        for dataset in (ERA5HourlyLand(), ERA5MonthlyLand())
            metadatum = Metadatum(:skin_temperature; dataset, date)
            @test hasmethod(Downloads.download, Tuple{typeof(metadatum)})

            @test CDSExt.variable_name_mapping(dataset) === ERA5Land_dataset_variable_names
            @test isnothing(CDSExt.pressure_levels(dataset))
            @test CDSExt.cds_dataset_keyword(dataset) == :era5_land
        end

        # ERA5-Land routes through hourly()/monthly() like their non-land counterparts
        @test CDSExt.cds_download_function(ERA5HourlyLand())  == CopernicusClimateDataStore.hourly
        @test CDSExt.cds_download_function(ERA5MonthlyLand()) == CopernicusClimateDataStore.monthly

        # Regular (non-land) datasets keep the :era5 keyword
        @test CDSExt.cds_dataset_keyword(ERA5HourlySingleLevel()) == :era5
        @test CDSExt.cds_dataset_keyword(ERA5HourlyPressureLevels([100000.0])) == :era5

        # date_keywords: hourly land matches hourly pressure-level shape, monthly land matches monthly shape
        dt = DateTime(2020, 6, 15, 12)
        kw_h = CDSExt.date_keywords(ERA5HourlyLand(), dt)
        @test kw_h.startyear == 2020
        @test kw_h.months == 6
        @test kw_h.days == 15
        @test kw_h.hours == 12

        kw_m = CDSExt.date_keywords(ERA5MonthlyLand(), dt)
        @test kw_m.year == 2020
        @test kw_m.month == 6
    end

    @info "✓ CopernicusClimateDataStore extension tests passed"
end
