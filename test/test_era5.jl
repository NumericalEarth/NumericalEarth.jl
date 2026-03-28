include("runtests_setup.jl")

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5_dataset_variable_names,
                                         ERA5_netcdf_variable_names,
                                         ERA5_wave_variables,
                                         metadata_filename,
                                         region_suffix,
                                         is_three_dimensional

using NumericalEarth.DataWrangling: dataset_location, dataset_variable_name,
                                    BoundingBox, Column

using Oceananigans.Fields: Center

@testset "ERA5 dataset types" begin
    @testset "ERA5Hourly basics" begin
        ds = ERA5Hourly()
        @test ds isa ERA5.ERA5Dataset

        # Atmospheric variables on 0.25° grid
        @test size(ds, :temperature) == (1440, 721, 1)
        @test size(ds, :eastward_velocity) == (1440, 721, 1)
        @test size(ds, :surface_pressure) == (1440, 721, 1)

        # Wave variables on 0.5° grid
        @test size(ds, :eastward_stokes_drift) == (720, 361, 1)
        @test size(ds, :significant_wave_height) == (720, 361, 1)
    end

    @testset "ERA5Monthly basics" begin
        ds = ERA5Monthly()
        @test ds isa ERA5.ERA5Dataset
        @test size(ds, :temperature) == (1440, 721, 1)
    end

    @testset "ERA5 date ranges" begin
        hourly_dates = all_dates(ERA5Hourly(), :temperature)
        @test first(hourly_dates) == DateTime("1940-01-01")
        @test last(hourly_dates) == DateTime("2024-12-31")

        monthly_dates = all_dates(ERA5Monthly(), :temperature)
        @test first(monthly_dates) == DateTime("1940-01-01")
        @test last(monthly_dates) == DateTime("2024-12-01")
        @test length(monthly_dates) == (2024 - 1940) * 12 + 12
    end
end

@testset "ERA5 metadata" begin
    @testset "ERA5 is 2D surface data" begin
        md = Metadatum(:temperature; dataset=ERA5Hourly(),
                       date=DateTime(2020, 1, 1))
        @test !is_three_dimensional(md)
    end

    @testset "ERA5 location is surface-only" begin
        @test dataset_location(ERA5Hourly(), :temperature) == (Center, Center, Nothing)
        @test dataset_location(ERA5Monthly(), :eastward_velocity) == (Center, Center, Nothing)
    end

    @testset "ERA5 variable name mappings" begin
        # CDS API names
        @test ERA5_dataset_variable_names[:temperature] == "2m_temperature"
        @test ERA5_dataset_variable_names[:eastward_velocity] == "10m_u_component_of_wind"
        @test ERA5_dataset_variable_names[:surface_pressure] == "surface_pressure"
        @test ERA5_dataset_variable_names[:downwelling_shortwave_radiation] == "surface_solar_radiation_downwards"

        # NetCDF short names
        @test ERA5_netcdf_variable_names[:temperature] == "t2m"
        @test ERA5_netcdf_variable_names[:eastward_velocity] == "u10"
        @test ERA5_netcdf_variable_names[:specific_humidity] == "q"

        # dataset_variable_name dispatch
        md = Metadatum(:temperature; dataset=ERA5Hourly(), date=DateTime(2020, 1, 1))
        @test dataset_variable_name(md) == "2m_temperature"
    end

    @testset "ERA5 metadata filename construction" begin
        ds = ERA5Hourly()

        # Single date, no region
        fn = metadata_filename(ds, :temperature, DateTime(2020, 3, 15), nothing)
        @test endswith(fn, ".nc")
        @test occursin("2m_temperature", fn)
        @test occursin("ERA5Hourly", fn)
        @test occursin("2020-03", fn)

        # With BoundingBox region
        bbox = BoundingBox(longitude=(10, 20), latitude=(-30, -20))
        fn_bbox = metadata_filename(ds, :temperature, DateTime(2020, 3, 15), bbox)
        @test fn_bbox != fn  # region changes the filename
        @test occursin("10.0", fn_bbox)
        @test occursin("20.0", fn_bbox)

        # With Column region
        col = Column(15.5, -25.0)
        fn_col = metadata_filename(ds, :temperature, DateTime(2020, 3, 15), col)
        @test occursin("15.5", fn_col)
    end

    @testset "ERA5 region_suffix" begin
        @test region_suffix(nothing) == ""

        bbox = BoundingBox(longitude=(10, 20), latitude=(-30, -20))
        suffix = region_suffix(bbox)
        @test length(suffix) > 0
        @test occursin("10.0", suffix)
    end
end

@testset "ERA5 Metadata construction" begin
    @testset "ERA5 Metadatum" begin
        md = Metadatum(:temperature; dataset=ERA5Hourly(),
                       date=DateTime(2020, 6, 15, 12))
        @test md.name == :temperature
        @test md.dataset isa ERA5Hourly
        @test md.dates == DateTime(2020, 6, 15, 12)
    end

    @testset "ERA5 Metadata with date range" begin
        dates = DateTime(2020, 1, 1):Month(1):DateTime(2020, 6, 1)
        md = Metadata(:temperature; dataset=ERA5Monthly(), dates=dates)
        @test length(md) == 6
        @test first(md).dates == DateTime(2020, 1, 1)
        @test last(md).dates == DateTime(2020, 6, 1)
    end

    @testset "ERA5 Metadata with Column region" begin
        col = Column(200.0, 35.0)
        md = Metadatum(:temperature; dataset=ERA5Hourly(),
                       date=DateTime(2020, 1, 1), region=col)
        @test md.region isa Column
        @test md.region.longitude == 200.0
    end

    @testset "ERA5 Metadata with BoundingBox region" begin
        bbox = BoundingBox(longitude=(200, 220), latitude=(35, 55))
        md = Metadatum(:temperature; dataset=ERA5Hourly(),
                       date=DateTime(2020, 1, 1), region=bbox)
        @test md.region isa BoundingBox
    end

    @testset "ERA5 wave variable classification" begin
        @test :eastward_stokes_drift in ERA5_wave_variables
        @test :significant_wave_height in ERA5_wave_variables
        @test :temperature ∉ ERA5_wave_variables
        @test :surface_pressure ∉ ERA5_wave_variables
    end
end
