include("runtests_setup.jl")

using CopernicusClimateDataStore
using Dates
import Downloads
using NCDatasets

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5YearlySingleLevel, ERA5MonthlySingleLevel,
                                         ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                         ERA5HourlyLand, ERA5MonthlyLand,
                                         ERA5Land_dataset_variable_names, ERA5Land_netcdf_variable_names,
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

            # Ownership (issue #530): `Downloads.download` for ERA5-Land metadata is the
            # src method, which outranks any extension's generic `Metadatum{<:ERA5Dataset}`
            # method under any load order; the CDS backend enters via `download_era5_land`.
            @test which(Downloads.download, Tuple{typeof(metadatum)}).module ===
                  NumericalEarth.DataWrangling.ERA5

            metadata = Metadata(:skin_temperature; dataset, dates=[date, DateTime(2020, 6, 1)])
            @test which(Downloads.download, Tuple{typeof(metadata)}).module ===
                  NumericalEarth.DataWrangling.ERA5

            # This extension implements the `download_era5_land` stub.
            @test which(NumericalEarth.DataWrangling.ERA5.download_era5_land,
                        Tuple{typeof(metadatum)}).module === CDSExt
        end

        # A file already on disk short-circuits the download without touching any backend.
        mktempdir() do dir
            metadatum = Metadatum(:skin_temperature; dataset=ERA5MonthlyLand(), date, dir)
            path = NumericalEarth.DataWrangling.metadata_path(metadatum)
            touch(path)
            @test Downloads.download(metadatum) == path
        end

        # Land datasets have their own yearly-file download method and are NOT
        # dispatched through the generic single-level/pressure-level helpers.
        @test CDSExt.cds_dataset_keyword(ERA5HourlySingleLevel()) == :era5
        @test CDSExt.cds_dataset_keyword(ERA5HourlyPressureLevels([100000.0])) == :era5
        @test !hasmethod(CDSExt.cds_dataset_keyword, Tuple{ERA5HourlyLand})
        @test !hasmethod(CDSExt.variable_name_mapping, Tuple{ERA5HourlyLand})
        @test !hasmethod(CDSExt.date_keywords, Tuple{ERA5HourlyLand, DateTime})

        # CDS product ids
        @test CDSExt.cds_land_product(ERA5HourlyLand())  == "reanalysis-era5-land"
        @test CDSExt.cds_land_product(ERA5MonthlyLand()) == "reanalysis-era5-land-monthly-means"

        # era5_land_request: uses the current CDS API v2 keys (not the legacy
        # "format" key, which is what previously triggered CDS's zip-wrapping bug)
        dts = [DateTime(2020, 6, 15, 12), DateTime(2020, 6, 15, 18)]
        req_h = CDSExt.era5_land_request("skin_temperature", ERA5HourlyLand(), dts, nothing)
        @test req_h["data_format"] == "netcdf"
        @test req_h["download_format"] == "unarchived"
        @test req_h["year"] == ["2020"]
        @test req_h["month"] == ["06"]
        @test sort(req_h["day"]) == ["15"]
        @test sort(req_h["time"]) == ["12:00", "18:00"]
        @test !haskey(req_h, "format")

        req_m = CDSExt.era5_land_request("skin_temperature", ERA5MonthlyLand(), dts, nothing)
        @test req_m["data_format"] == "netcdf"
        @test req_m["download_format"] == "unarchived"
        @test req_m["product_type"] == ["monthly_averaged_reanalysis"]
        @test !haskey(req_m, "format")

        # era5_land_year_batches: hourly splits a year into calendar-month chunks
        # (a full year of hourly data exceeds CDS's per-request cost limit);
        # monthly means fit a whole year in a single request.
        year_dates = collect(DateTime(2020, 1, 1):Hour(1):DateTime(2020, 3, 1))
        batches = CDSExt.era5_land_year_batches(ERA5HourlyLand(), year_dates)
        @test length(batches) == 3
        @test all(dt -> Dates.month(dt) == 1, batches[1])
        @test all(dt -> Dates.month(dt) == 2, batches[2])

        monthly_dates = collect(DateTime(2020, 1, 1):Month(1):DateTime(2020, 12, 1))
        @test CDSExt.era5_land_year_batches(ERA5MonthlyLand(), monthly_dates) == [monthly_dates]
    end

    @testset "ERA5-Land / ERA5YearlySingleLevel build_filename supports multi-year requests" begin
        DatewiseFilename = NumericalEarth.DataWrangling.DatewiseFilename
        build_filename = NumericalEarth.DataWrangling.build_filename
        metadata_filename = NumericalEarth.DataWrangling.metadata_filename

        for (ds, name) in ((ERA5HourlyLand(), :skin_temperature), (ERA5YearlySingleLevel(), :temperature))
            # One yearly file covers every date within that year — same-year requests
            # resolve every date to that one (repeated) filename.
            same_year_dates = [DateTime(2020, 1, 1), DateTime(2020, 6, 15)]
            same_year = build_filename(ds, name, same_year_dates, nothing)
            @test same_year isa DatewiseFilename
            @test same_year.filenames == fill(metadata_filename(ds, name, same_year_dates[1], nothing), 2)

            # A date range spanning more than one calendar year must resolve each
            # date to ITS OWN year's file, not silently collapse to the first year's.
            cross_year_dates = [DateTime(2020, 12, 31), DateTime(2021, 1, 1)]
            cross_year = build_filename(ds, name, cross_year_dates, nothing)
            @test cross_year isa DatewiseFilename
            @test cross_year.filenames[1] == metadata_filename(ds, name, cross_year_dates[1], nothing)
            @test cross_year.filenames[2] == metadata_filename(ds, name, cross_year_dates[2], nothing)
            @test cross_year.filenames[1] != cross_year.filenames[2]
        end
    end

    @testset "ERA5MonthlySingleLevel build_filename supports multi-month requests" begin
        DatewiseFilename = NumericalEarth.DataWrangling.DatewiseFilename
        build_filename = NumericalEarth.DataWrangling.build_filename
        metadata_filename = NumericalEarth.DataWrangling.metadata_filename

        ds, name = ERA5MonthlySingleLevel(), :temperature

        # One monthly file covers every date within that month — same-month requests
        # resolve every date to that one (repeated) filename.
        same_month_dates = [DateTime(2020, 1, 1), DateTime(2020, 1, 15)]
        same_month = build_filename(ds, name, same_month_dates, nothing)
        @test same_month isa DatewiseFilename
        @test same_month.filenames == fill(metadata_filename(ds, name, same_month_dates[1], nothing), 2)

        # A date range spanning more than one month must resolve each date to ITS
        # OWN month's file, not silently collapse to the first month's.
        cross_month_dates = [DateTime(2020, 1, 31), DateTime(2020, 2, 1), DateTime(2021, 1, 1)]
        cross_month = build_filename(ds, name, cross_month_dates, nothing)
        @test cross_month isa DatewiseFilename
        @test cross_month.filenames[1] == metadata_filename(ds, name, cross_month_dates[1], nothing)
        @test cross_month.filenames[2] == metadata_filename(ds, name, cross_month_dates[2], nothing)
        @test cross_month.filenames[3] == metadata_filename(ds, name, cross_month_dates[3], nothing)
        @test allunique(cross_month.filenames)
    end

    @testset "read_era5_yearly_series reads across a year boundary" begin
        read_era5_yearly_series = NumericalEarth.DataWrangling.ERA5.read_era5_yearly_series

        # Write two small synthetic yearly files, each with distinct marker values
        # so we can confirm the stitched-together series preserves chronological order.
        function write_year_file(path, times, marker)
            NCDatasets.Dataset(path, "c") do ds
                NCDatasets.defDim(ds, "longitude", 2)
                NCDatasets.defDim(ds, "latitude", 2)
                NCDatasets.defDim(ds, "valid_time", length(times))
                NCDatasets.defVar(ds, "longitude", Float64, ("longitude",))[:] = [-1.0, 1.0]
                NCDatasets.defVar(ds, "latitude", Float64, ("latitude",))[:] = [40.0, 41.0]
                tv = NCDatasets.defVar(ds, "valid_time", Float64, ("valid_time",);
                                       attrib = ["units" => "seconds since 1970-01-01"])
                tv[:] = times
                t2m = NCDatasets.defVar(ds, "t2m", Float32, ("longitude", "latitude", "valid_time"))
                for (k, _) in enumerate(times), j in 1:2, i in 1:2
                    t2m[i, j, k] = Float32(marker + k)
                end
            end
        end

        mktempdir() do dir
            path_2020 = joinpath(dir, "t2m_2020.nc")
            path_2021 = joinpath(dir, "t2m_2021.nc")
            times_2020 = [DateTime(2020, 12, 31, 22), DateTime(2020, 12, 31, 23)]
            times_2021 = [DateTime(2021, 1, 1, 0), DateTime(2021, 1, 1, 1)]
            write_year_file(path_2020, times_2020, 0.0)     # markers 1, 2
            write_year_file(path_2021, times_2021, 100.0)   # markers 101, 102

            @testset "single file (String path, backward-compatible)" begin
                raw, λc, φc = read_era5_yearly_series(path_2020, times_2020, "t2m")
                @test size(raw) == (2, 2, 2)
                @test length(λc) == 2 && length(φc) == 2
                @test all(==(1f0), raw[:, :, 1])
                @test all(==(2f0), raw[:, :, 2])
            end

            @testset "multiple files spanning a year boundary (Vector of paths)" begin
                paths = [path_2020, path_2020, path_2021, path_2021]
                requested_times = vcat(times_2020, times_2021)
                raw, λc, φc = read_era5_yearly_series(paths, requested_times, "t2m")
                @test size(raw) == (2, 2, 4)
                @test all(==(1f0),   raw[:, :, 1])
                @test all(==(2f0),   raw[:, :, 2])
                @test all(==(101f0), raw[:, :, 3])
                @test all(==(102f0), raw[:, :, 4])
            end

            @testset "requesting a subset within one of the files" begin
                # Only the second timestep of the 2020 file, then both of 2021 —
                # exercises the single-index (non-range) read path within a group.
                paths = [path_2020, path_2021, path_2021]
                requested_times = [times_2020[2], times_2021[1], times_2021[2]]
                raw, _, _ = read_era5_yearly_series(paths, requested_times, "t2m")
                @test size(raw) == (2, 2, 3)
                @test all(==(2f0),   raw[:, :, 1])
                @test all(==(101f0), raw[:, :, 2])
                @test all(==(102f0), raw[:, :, 3])
            end
        end
    end

    @info "✓ CopernicusClimateDataStore extension tests passed"
end
