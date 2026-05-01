include("runtests_setup.jl")
include("download_utils.jl")

using CDSAPI
using Dates
using NCDatasets

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5MonthlySingleLevel,
                                         ERA5_dataset_variable_names, ERA5_netcdf_variable_names
using NumericalEarth.DataWrangling.ERA5: ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                         ERA5_all_pressure_levels, ERA5PL_dataset_variable_names,
                                         ERA5PL_netcdf_variable_names, pressure_field
using NumericalEarth.DataWrangling: metadata_path, download_dataset, BoundingBox, Column, Linear, Nearest

# Internal extension module — exposes dispatch helpers and NetCDF utilities
# that are not part of the public API but worth pinning behavior for.
const CDSExt = Base.get_extension(NumericalEarth, :NumericalEarthCDSAPIExt)

# Test date: Kyoto Protocol ratification date, February 16, 2005
start_date = DateTime(2005, 2, 16, 12)

@testset "ERA5 data downloading and utilities" begin
    @info "Testing ERA5 downloading and NetCDF file verification..."

    dataset = ERA5HourlySingleLevel()

    # Use a small bounding box to reduce download time
    region = NumericalEarth.DataWrangling.BoundingBox(longitude=(0, 5), latitude=(40, 45))

    @testset "Download ERA5 temperature data" begin
        variable = :temperature
        metadatum = Metadatum(variable; dataset, region, date=start_date)

        # Clean up any existing file
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)

        # Download the data (falls back to NumericalEarthArtifacts if CDS is unreachable)
        download_dataset_with_fallback(filepath; dataset_name="ERA5Hourly $variable") do
            download_dataset(metadatum)
        end
        @test isfile(filepath)

        # Verify the NetCDF file structure
        ds = NCDataset(filepath)

        # Check that it has the expected variable (t2m for 2m_temperature)
        @test haskey(ds, "t2m")

        # Check that it has coordinate variables
        @test haskey(ds, "longitude")
        @test haskey(ds, "latitude")
        @test haskey(ds, "time") || haskey(ds, "valid_time")

        # Check data dimensions
        lon = ds["longitude"][:]
        lat = ds["latitude"][:]
        @test length(lon) > 0
        @test length(lat) > 0

        # Check that data is within expected bounds
        @test minimum(lon) >= -1  # Allow some tolerance
        @test maximum(lon) <= 6
        @test minimum(lat) >= 39
        @test maximum(lat) <= 46

        # Check that the temperature data exists and is valid
        t2m = ds["t2m"]
        @test ndims(t2m) >= 2

        close(ds)

        # Note: leave `filepath` in place; downstream surface-level testsets reuse it.
    end

    @testset "Availability of ERA5 variables" begin
        # Test that we have defined the key ERA5 variables
        @test haskey(ERA5_dataset_variable_names, :temperature)
        @test haskey(ERA5_dataset_variable_names, :eastward_velocity)
        @test haskey(ERA5_dataset_variable_names, :northward_velocity)
        @test haskey(ERA5_dataset_variable_names, :surface_pressure)
        @test haskey(ERA5_dataset_variable_names, :downwelling_shortwave_radiation)
        @test haskey(ERA5_dataset_variable_names, :downwelling_longwave_radiation)

        # Verify variable name mappings
        @test ERA5_dataset_variable_names[:temperature] == "2m_temperature"
        @test ERA5_dataset_variable_names[:eastward_velocity] == "10m_u_component_of_wind"
        @test ERA5_dataset_variable_names[:northward_velocity] == "10m_v_component_of_wind"
    end

    @testset "ERA5 metadata properties" begin
        variable = :temperature
        metadatum = Metadatum(variable; dataset, region, date=start_date)

        # Test metadata properties
        @test metadatum.name == :temperature
        @test metadatum.dataset isa ERA5HourlySingleLevel
        @test metadatum.dates == start_date
        @test metadatum.region == region

        # Test size (should be global ERA5 size with 1 time step)
        Nx, Ny, Nz, Nt = size(metadatum)
        @test Nx == 1440  # ERA5 longitude points
        @test Ny == 720   # ERA5 latitude points (poles averaged into adjacent cells)
        @test Nz == 1     # 2D surface data
        @test Nt == 1     # Single time step

        # Test that ERA5 is correctly identified as 2D
        @test NumericalEarth.DataWrangling.ERA5.is_three_dimensional(metadatum) == false
    end

    @testset "ERA5 wave variable metadata sizes" begin
        # Wave variables should be on the 0.5° grid (720×360)
        for wave_var in (:eastward_stokes_drift, :northward_stokes_drift,
                         :significant_wave_height, :mean_wave_period, :mean_wave_direction)
            metadatum = Metadatum(wave_var; dataset, date=start_date)
            Nx, Ny, Nz, Nt = size(metadatum)
            @test Nx == 720
            @test Ny == 360
            @test Nz == 1
            @test Nt == 1
        end

        # Atmospheric variables should remain on the 0.25° grid (1440×720)
        for atmos_var in (:temperature, :eastward_velocity, :surface_pressure)
            metadatum = Metadatum(atmos_var; dataset, date=start_date)
            Nx, Ny, Nz, Nt = size(metadatum)
            @test Nx == 1440
            @test Ny == 720
            @test Nz == 1
            @test Nt == 1
        end
    end

    @testset "ERA5 Monthly dataset" begin
        monthly_dataset = ERA5MonthlySingleLevel()
        @test monthly_dataset isa ERA5MonthlySingleLevel

        # Test that all_dates returns a valid range
        dates = NumericalEarth.DataWrangling.all_dates(monthly_dataset, :temperature)
        @test first(dates) == DateTime("1940-01-01")
        @test step(dates) == Month(1)
    end

    @testset "ERA5 single-level all_dates (Hourly)" begin
        hourly_dataset = ERA5HourlySingleLevel()
        dates = NumericalEarth.DataWrangling.all_dates(hourly_dataset, :temperature)
        @test first(dates) == DateTime("1940-01-01")
        @test step(dates) == Hour(1)
    end

    @testset "ERA5 single-level dispatch helpers" begin
        ds = ERA5HourlySingleLevel()
        md = Metadatum(:temperature; dataset=ds, region, date=start_date)

        # API-name and netcdf-name dicts cover the same variable set —
        # catches forgetting to add a new variable to one of the two
        @test keys(ERA5_dataset_variable_names) == keys(ERA5_netcdf_variable_names)

        # available_variables returns the API-name dict (used to build CDS requests),
        # not the netcdf short-name dict — guards against the easy swap-mistake
        @test NumericalEarth.DataWrangling.available_variables(ds) === ERA5_dataset_variable_names

        # dataset_variable_name returns the netcdf short name (read from file),
        # not the API catalog name — same swap risk
        @test NumericalEarth.DataWrangling.dataset_variable_name(md) == "t2m"

        # default_inpainting is nothing for ERA5 (vs NearestNeighborInpainting for ECCO);
        # accidentally enabling it would massively slow Field construction
        @test NumericalEarth.DataWrangling.default_inpainting(md) === nothing
    end

    @testset "ERA5 single-level metadata_prefix" begin
        ds = ERA5HourlySingleLevel()

        # Single-date metadatum, with region: prefix should not duplicate the date
        md_single = Metadatum(:temperature; dataset=ds, region, date=start_date)
        prefix_single = NumericalEarth.DataWrangling.ERA5.metadata_prefix(md_single)
        @test occursin("2m_temperature", prefix_single)
        @test occursin("ERA5HourlySingleLevel", prefix_single)
        @test occursin("2005-02-16", prefix_single)
        @test count("2005-02-16", prefix_single) == 1   # date appears once for single-date
        @test occursin("0.0", prefix_single)            # west bound
        @test occursin("5.0", prefix_single)            # east bound
        @test occursin("40.0", prefix_single)           # south bound
        @test occursin("45.0", prefix_single)           # north bound
        # Filename safety
        @test !occursin(":", prefix_single)             # colons replaced by dashes
        @test !occursin(" ", prefix_single)             # spaces replaced by underscores

        # Single-date metadatum, no region: suffix should be empty
        md_no_region = Metadatum(:temperature; dataset=ds, date=start_date)
        prefix_no_region = NumericalEarth.DataWrangling.ERA5.metadata_prefix(md_no_region)
        @test !occursin("0.0", prefix_no_region)
        @test !occursin("nothing", prefix_no_region)

        # Multi-date metadata: prefix should include both start and end dates
        end_date = start_date + Hour(2)
        md_multi = Metadata(:temperature; dataset=ds, region,
                            dates=start_date:Hour(1):end_date)
        prefix_multi = NumericalEarth.DataWrangling.ERA5.metadata_prefix(md_multi)
        @test occursin("2005-02-16T12", prefix_multi)
        @test occursin("2005-02-16T14", prefix_multi)
    end

    @testset "ERA5HourlyPressureLevels construction and metadata" begin
        # Default constructor uses all 37 standard levels
        ds_full = ERA5HourlyPressureLevels()
        @test ds_full isa ERA5HourlyPressureLevels
        @test length(ds_full.pressure_levels) == 37
        @test Base.size(ds_full, :temperature) == (1440, 720, 37)

        # Subset constructor
        ds_sub = ERA5HourlyPressureLevels(pressure_levels=[850, 500]hPa)
        @test Base.size(ds_sub, :temperature) == (1440, 720, 2)

        # Monthly variant
        ds_monthly = ERA5MonthlyPressureLevels()
        @test ds_monthly isa ERA5MonthlyPressureLevels

        # Metadatum size propagates Nz correctly
        meta = Metadatum(:temperature; dataset=ds_sub, region=region, date=start_date)
        Nx, Ny, Nz, Nt = size(meta)
        @test Nz == 2
        @test NumericalEarth.DataWrangling.ERA5.is_three_dimensional(meta) == true

        # Variable name lookups
        @test ERA5PL_dataset_variable_names[:temperature] == "temperature"
        @test ERA5PL_dataset_variable_names[:geopotential_height] == "geopotential"
    end

    @testset "ERA5 pressure-level z_interfaces (standard atmosphere)" begin
        levels_2 = [850, 500]hPa
        z = standard_atmosphere_z_interfaces(levels_2)
        @test length(z) == 3                    # Nz+1 interfaces
        @test issorted(z)                        # monotonically increasing with altitude
        # 850 hPa ≈ 1457 m, 500 hPa ≈ 5575 m
        @test z[1] < 1457.0 < z[2] < 5575.0 < z[3]

        # Single level
        z1 = standard_atmosphere_z_interfaces([500]hPa)
        @test length(z1) == 2
        @test z1[1] < z1[2]
    end

    @testset "ERA5 pressure-level constructors sort levels descending" begin
        # Pass ASCENDING input so the test fails if the inner constructor's
        # `sort(...; rev=true)` regresses to a no-op or different order.
        ds_h = ERA5HourlyPressureLevels([500, 850]hPa)
        @test ds_h.pressure_levels == [850 * hPa, 500 * hPa]    # stored highest-pressure-first
        @test ds_h.z === nothing
        @test ds_h.mean_geopotential_height == true             # default kwarg

        ds_m = ERA5MonthlyPressureLevels([500, 850]hPa)
        @test ds_m.pressure_levels == [850 * hPa, 500 * hPa]
        @test ds_m.z === nothing
        @test ds_m.mean_geopotential_height == true

        # Already-descending input is preserved (sort is a no-op here)
        ds_h2 = ERA5HourlyPressureLevels([850, 500]hPa)
        @test ds_h2.pressure_levels == [850 * hPa, 500 * hPa]

        # Three-level shuffled input
        ds_h3 = ERA5HourlyPressureLevels([500, 850, 700]hPa)
        @test ds_h3.pressure_levels == [850 * hPa, 700 * hPa, 500 * hPa]
    end

    @testset "ERA5 pressure-level stagger" begin
        stagger = NumericalEarth.DataWrangling.ERA5.stagger

        # Two-element input (evenly spaced): bottom and top faces are
        # extrapolated symmetrically; result is Nz+1 monotonic.
        zf = stagger([0.0, 1.0])
        @test length(zf) == 3
        @test issorted(zf)
        @test zf ≈ [-0.5, 0.5, 1.5]

        # Three-element evenly-spaced: every interior interface is the
        # midpoint of the adjacent centers; bottom/top are extrapolated.
        zf = stagger([1.0, 3.0, 5.0])
        @test length(zf) == 4
        @test zf ≈ [0.0, 2.0, 4.0, 6.0]

        # Three-element irregularly-spaced: interior midpoints honor the
        # actual spacing (not assumed-uniform).
        zf = stagger([1.0, 3.0, 7.0])
        @test length(zf) == 4
        @test zf[2] ≈ 2.0   # midpoint(1, 3)
        @test zf[3] ≈ 5.0   # midpoint(3, 7)
        # Boundaries extrapolated at half the adjacent interior spacing
        @test zf[1] ≈ 1.0 - (zf[2] - 1.0)
        @test zf[4] ≈ 7.0 + (7.0 - zf[3])
    end

    for arch in test_architectures
        A = typeof(arch)

        @testset "Field creation from ERA5 on $A" begin
            variable = :temperature
            metadatum = Metadatum(variable; dataset, region, date=start_date)

            # Download if not present (falls back to NumericalEarthArtifacts if CDS is unreachable)
            filepath = metadata_path(metadatum)
            isfile(filepath) || download_dataset_with_fallback(filepath; dataset_name="ERA5Hourly $variable") do
                download_dataset(metadatum)
            end

            # Create a Field from the downloaded data
            ψ = Field(metadatum, arch)
            @test ψ isa Field

            # ERA5 is 2D data, so field should have Nz=1
            Nx, Ny, Nz = size(ψ)
            @test Nz == 1

            # Verify the field has non-zero data (temperature in Kelvin ~250-310K)
            @allowscalar begin
                @test !all(iszero, interior(ψ))
            end

            # Note: cleanup happens in the last surface-level testset below.
        end

        @testset "Setting a field from ERA5 metadata on $A" begin
            variable = :temperature
            metadatum = Metadatum(variable; dataset, region, date=start_date)

            # Download if not present (falls back to NumericalEarthArtifacts if CDS is unreachable)
            filepath = metadata_path(metadatum)
            isfile(filepath) || download_dataset_with_fallback(filepath; dataset_name="ERA5Hourly $variable") do
                download_dataset(metadatum)
            end

            # Create a target grid matching the bounding box region
            grid = LatitudeLongitudeGrid(arch;
                                         size = (10, 10, 1),
                                         latitude = (40, 45),
                                         longitude = (0, 5),
                                         z = (0, 1))

            field = CenterField(grid)

            # Set the field from metadata
            set!(field, metadatum)

            # Verify the field was set with non-zero data
            @allowscalar begin
                @test !all(iszero, interior(field))
            end

            # Clean up
            rm(filepath; force=true)
            inpainted_path = NumericalEarth.DataWrangling.inpainted_metadata_path(metadatum)
            isfile(inpainted_path) && rm(inpainted_path; force=true)
        end
    end

    @testset "ERA5 pressure-level download and Field on CPU" begin
        arch = CPU()
        ds_pl = ERA5HourlyPressureLevels(pressure_levels=[850, 500]hPa)

        @testset "Download and 3D Field" begin
            meta = Metadatum(:temperature; dataset=ds_pl, region, date=start_date)
            filepath = metadata_path(meta)
            isfile(filepath) && rm(filepath; force=true)

            download_dataset(meta)
            @test isfile(filepath)

            # Verify the NetCDF has a pressure_level dimension and the right variable
            ds_nc = NCDataset(filepath)
            @test haskey(ds_nc, "t")
            @test haskey(ds_nc, "pressure_level") || haskey(ds_nc, "level")
            close(ds_nc)

            f = Field(meta, arch)
            @test f isa Field
            Nx, Ny, Nz = size(f)
            @test Nz == 2

            @allowscalar begin
                @test !all(iszero, interior(f))
                # Temperature at these levels should be in a plausible range (K)
                @test all(x -> 180 < x < 340, filter(!isnan, vec(interior(f))))
            end

            rm(filepath; force=true)
            inpainted_path = NumericalEarth.DataWrangling.inpainted_metadata_path(meta)
            isfile(inpainted_path) && rm(inpainted_path; force=true)
        end

        @testset "Geopotential height conversion" begin
            meta_z = Metadatum(:geopotential_height; dataset=ds_pl, region, date=start_date)
            filepath = metadata_path(meta_z)

            # Field() downloads if needed; the file may already be on disk from
            # the previous testset's z_interfaces side-effect.
            fz = Field(meta_z, arch)

            @allowscalar begin
                max_z = maximum(filter(!isnan, vec(interior(fz))))
                # 500 hPa geopotential height ≈ 5500 m
                @test 4000 < max_z < 7000
            end

            rm(filepath; force=true)
            inpainted_path = NumericalEarth.DataWrangling.inpainted_metadata_path(meta_z)
            isfile(inpainted_path) && rm(inpainted_path; force=true)
        end

        @testset "pressure_field" begin
            meta = Metadatum(:temperature; dataset=ds_pl, region, date=start_date)
            pf = pressure_field(meta, arch)
            @test pf isa Field
            Nx, Ny, Nz = size(pf)
            @test Nz == 2

            @allowscalar begin
                # k=1 should be 850 hPa = 85000 Pa (highest pressure, lowest altitude)
                @test interior(pf)[1, 1, 1] ≈ Float32(850hPa)
                # k=2 should be 500 hPa = 50000 Pa
                @test interior(pf)[1, 1, 2] ≈ Float32(500hPa)
            end
        end
    end
end

@testset "ERA5 CDSAPIExt dispatch helpers and area construction" begin
    sl = ERA5HourlySingleLevel()
    pl = ERA5HourlyPressureLevels(pressure_levels=[500hPa, 850hPa])

    @testset "cds_product / cds_varnames / nc_varnames" begin
        @test CDSExt.cds_product(sl) == "reanalysis-era5-single-levels"
        @test CDSExt.cds_product(pl) == "reanalysis-era5-pressure-levels"

        @test CDSExt.cds_varnames(sl) === ERA5_dataset_variable_names
        @test CDSExt.cds_varnames(pl) === ERA5PL_dataset_variable_names

        @test CDSExt.nc_varnames(sl) === ERA5_netcdf_variable_names
        @test CDSExt.nc_varnames(pl) === ERA5PL_netcdf_variable_names
    end

    @testset "coord_vars" begin
        sl_coords = CDSExt.coord_vars(sl)
        pl_coords = CDSExt.coord_vars(pl)

        @test sl_coords isa Set
        @test "longitude" in sl_coords
        @test "latitude" in sl_coords
        @test "valid_time" in sl_coords
        @test !("pressure_level" in sl_coords)

        @test "longitude" in pl_coords
        @test "pressure_level" in pl_coords
        @test "level" in pl_coords
    end

    @testset "extra_request_keys!" begin
        # ERA5Dataset (single level): no-op
        request = Dict{String, Any}("variable" => ["2m_temperature"])
        CDSExt.extra_request_keys!(request, sl)
        @test !haskey(request, "pressure_level")

        # ERA5PressureLevelsDataset: populates `pressure_level` (in hPa, as strings)
        CDSExt.extra_request_keys!(request, pl)
        @test haskey(request, "pressure_level")
        @test Set(request["pressure_level"]) == Set(["500", "850"])
    end

    @testset "build_era5_area" begin
        # Nothing → nothing
        @test CDSExt.build_era5_area(nothing) === nothing

        # BoundingBox with both axes → [N, W, S, E]
        bbox = BoundingBox(longitude=(-10.0, 5.0), latitude=(40.0, 50.0))
        @test CDSExt.build_era5_area(bbox) == [50.0, -10.0, 40.0, 5.0]

        # BoundingBox with one axis missing → nothing (CDS gets the global slab)
        bbox_no_lat = BoundingBox(longitude=(-10.0, 5.0))
        @test CDSExt.build_era5_area(bbox_no_lat) === nothing
        bbox_no_lon = BoundingBox(latitude=(40.0, 50.0))
        @test CDSExt.build_era5_area(bbox_no_lon) === nothing

        # Column with Nearest interpolation → tight ε=1e-3 box around the point
        col_nr = Column(-61.5, 18.0; interpolation=Nearest())
        area_nr = CDSExt.build_era5_area(col_nr)
        @test length(area_nr) == 4
        # [N, W, S, E]
        @test area_nr[1] ≈ 18.0 + 1e-3      # north
        @test area_nr[2] ≈ -61.5 - 1e-3     # west
        @test area_nr[3] ≈ 18.0 - 1e-3      # south
        @test area_nr[4] ≈ -61.5 + 1e-3     # east

        # Column with Linear interpolation → ε=0.3 padding for 2x2 stencil
        col_lin = Column(-61.5, 18.0; interpolation=Linear())
        area_lin = CDSExt.build_era5_area(col_lin)
        @test area_lin[1] ≈ 18.0 + 0.3
        @test area_lin[2] ≈ -61.5 - 0.3
        @test area_lin[3] ≈ 18.0 - 0.3
        @test area_lin[4] ≈ -61.5 + 0.3
        # Linear box must enclose more than one ERA5 grid cell (0.25°)
        @test (area_lin[1] - area_lin[3]) > 0.25
        @test (area_lin[4] - area_lin[2]) > 0.25
    end
end

@testset "ERA5 CDSAPIExt _group_by_calendar_day" begin
    # Single calendar day with multiple hours
    same_day = [DateTime(2005, 2, 16, 0),
                DateTime(2005, 2, 16, 6),
                DateTime(2005, 2, 16, 23)]
    g = CDSExt._group_by_calendar_day(same_day)
    @test length(g) == 1
    @test Date(2005, 2, 16) in keys(g)
    @test length(g[Date(2005, 2, 16)]) == 3

    # Boundary: 00:00 belongs to its OWN day (not the previous one)
    midnight_pair = [DateTime(2005, 2, 16, 23),
                     DateTime(2005, 2, 17, 0)]
    g = CDSExt._group_by_calendar_day(midnight_pair)
    @test length(g) == 2
    @test g[Date(2005, 2, 16)] == [DateTime(2005, 2, 16, 23)]
    @test g[Date(2005, 2, 17)] == [DateTime(2005, 2, 17, 0)]

    # Multiple days, interleaved order — grouping must be order-independent
    mixed = [DateTime(2005, 2, 17, 6),
             DateTime(2005, 2, 16, 6),
             DateTime(2005, 2, 17, 12),
             DateTime(2005, 2, 16, 18)]
    g = CDSExt._group_by_calendar_day(mixed)
    @test length(g) == 2
    @test Set(g[Date(2005, 2, 16)]) == Set([DateTime(2005, 2, 16, 6), DateTime(2005, 2, 16, 18)])
    @test Set(g[Date(2005, 2, 17)]) == Set([DateTime(2005, 2, 17, 6), DateTime(2005, 2, 17, 12)])

    # Duplicate datetimes are preserved (CDS will dedupe; we don't)
    dups = [DateTime(2005, 2, 16, 12), DateTime(2005, 2, 16, 12)]
    g = CDSExt._group_by_calendar_day(dups)
    @test length(g) == 1
    @test length(g[Date(2005, 2, 16)]) == 2

    # Single-element input
    g = CDSExt._group_by_calendar_day([DateTime(2005, 2, 16, 12)])
    @test length(g) == 1
    @test g[Date(2005, 2, 16)] == [DateTime(2005, 2, 16, 12)]
end

@testset "ERA5 CDSAPIExt skip_existing short-circuit" begin
    # Build a temporary directory and pre-create the expected output files so
    # `download_dataset(...; skip_existing=true)` returns without contacting CDS.
    # If the short-circuit ever regresses, these tests will throw a credentials
    # error (or 4xx from the CDS API) and fail loudly.
    region = NumericalEarth.DataWrangling.BoundingBox(longitude=(0, 5), latitude=(40, 45))
    mktempdir() do tmp
        ds_pl  = ERA5HourlyPressureLevels(pressure_levels=[850, 500]hPa)
        date1  = DateTime(2005, 2, 16, 12)
        date2  = DateTime(2005, 2, 16, 18)
        names  = [:temperature, :eastward_velocity]

        # Helper: pre-create the file that `download_dataset` would write
        function touch_expected(name, dataset, date)
            md = Metadatum(name; dataset, region, date, dir=tmp)
            path = metadata_path(md)
            mkpath(dirname(path))
            touch(path)
            return path
        end

        @testset "multi-variable pressure-level (single date)" begin
            paths = [touch_expected(name, ds_pl, date1) for name in names]
            meta = Metadatum(:temperature; dataset=ds_pl, region, date=date1, dir=tmp)

            result = download_dataset(names, meta; skip_existing=true)
            @test result isa Vector{String}
            @test length(result) == length(names)
            @test Set(result) == Set(paths)
        end

        @testset "single-variable multi-date (download_era5_day)" begin
            # All hours of date1, date2 already on disk
            ds_sl = ERA5HourlySingleLevel()
            for dt in (date1, date2)
                touch_expected(:temperature, ds_sl, dt)
            end

            # Returns nothing without raising — the early-return guard fires
            @test CDSExt.download_era5_day(:temperature, ds_sl, [date1, date2];
                                            region, dir=tmp,
                                            skip_existing=true, cleanup=true) === nothing
        end

        @testset "multi-variable multi-date (download_era5_multivar_day)" begin
            ds_sl = ERA5HourlySingleLevel()
            for name in names, dt in (date1, date2)
                touch_expected(name, ds_sl, dt)
            end

            @test CDSExt.download_era5_multivar_day(names, ds_sl, [date1, date2];
                                                     region, dir=tmp,
                                                     skip_existing=true, cleanup=true) === nothing
        end
    end
end

@testset "ERA5 CDSAPIExt NetCDF copy and split helpers" begin
    # Helper: write a synthetic ERA5-like NetCDF with `Nt` timesteps and two
    # variables (`u`, `v`) on dims (longitude, latitude, valid_time).
    function write_synthetic_era5_nc(path; Nx=2, Ny=2, Nt=3)
        NCDatasets.Dataset(path, "c") do ds
            NCDatasets.defDim(ds, "longitude", Nx)
            NCDatasets.defDim(ds, "latitude",  Ny)
            NCDatasets.defDim(ds, "valid_time", Nt)
            ds.attrib["title"] = "synthetic_era5_test"

            lon = NCDatasets.defVar(ds, "longitude", Float64, ("longitude",))
            lat = NCDatasets.defVar(ds, "latitude",  Float64, ("latitude",))
            t   = NCDatasets.defVar(ds, "valid_time", Int64,  ("valid_time",))
            lon[:] = collect(range(-1.0, 1.0; length=Nx))
            lat[:] = collect(range(40.0, 41.0; length=Ny))
            t[:]   = collect(1:Nt)

            # u: includes _FillValue and a custom attribute
            u = NCDatasets.defVar(ds, "u", Float32,
                                  ("longitude", "latitude", "valid_time");
                                  fillvalue=Float32(-9999.0))
            u.attrib["units"] = "m s**-1"
            u.attrib["long_name"] = "u_component_of_wind"
            for k in 1:Nt, j in 1:Ny, i in 1:Nx
                u[i, j, k] = Float32(100k + 10j + i)
            end

            # v: no fill value, no extra attributes
            v = NCDatasets.defVar(ds, "v", Float32,
                                  ("longitude", "latitude", "valid_time"))
            for k in 1:Nt, j in 1:Ny, i in 1:Nx
                v[i, j, k] = Float32(-(100k + 10j + i))
            end
        end
    end

    coord_vars = CDSExt.ERA5_COORD_VARS

    @testset "ncvar_copy! preserves data, attributes, fill value" begin
        mktempdir() do dir
            src_path = joinpath(dir, "src.nc")
            dst_path = joinpath(dir, "dst.nc")
            write_synthetic_era5_nc(src_path; Nx=3, Ny=2, Nt=1)

            NCDatasets.Dataset(src_path, "r") do src
                NCDatasets.Dataset(dst_path, "c") do dst
                    for (dname, dlen) in src.dim
                        NCDatasets.defDim(dst, dname, dlen)
                    end
                    CDSExt.ncvar_copy!(dst, src["u"], "u")
                end
            end

            NCDatasets.Dataset(dst_path, "r") do dst
                @test haskey(dst, "u")
                @test eltype(dst["u"].var) == Float32
                @test dst["u"].attrib["units"] == "m s**-1"
                @test dst["u"].attrib["long_name"] == "u_component_of_wind"
                @test dst["u"].attrib["_FillValue"] == Float32(-9999.0)

                NCDatasets.Dataset(src_path, "r") do src
                    @test dst["u"].var[:] == src["u"].var[:]
                end
            end
        end
    end

    @testset "ncvar_copy_tslice! extracts a single timestep" begin
        mktempdir() do dir
            src_path = joinpath(dir, "src.nc")
            dst_path = joinpath(dir, "dst.nc")
            write_synthetic_era5_nc(src_path; Nx=2, Ny=2, Nt=3)

            tidx = 2
            time_dimnames = Set(["valid_time"])

            NCDatasets.Dataset(src_path, "r") do src
                NCDatasets.Dataset(dst_path, "c") do dst
                    for (dname, dlen) in src.dim
                        out_len = dname in time_dimnames ? 1 : dlen
                        NCDatasets.defDim(dst, dname, out_len)
                    end
                    CDSExt.ncvar_copy_tslice!(dst, src["u"], "u", tidx, time_dimnames)
                    # `valid_time` is a coord variable in the file — copy that too,
                    # using the same tslice path. Exercises the has_time branch.
                    CDSExt.ncvar_copy_tslice!(dst, src["valid_time"], "valid_time", tidx, time_dimnames)
                    # `longitude` has no time dim — exercises the !has_time branch.
                    CDSExt.ncvar_copy_tslice!(dst, src["longitude"], "longitude", tidx, time_dimnames)
                end
            end

            NCDatasets.Dataset(dst_path, "r") do dst
                @test dst.dim["valid_time"] == 1
                @test size(dst["u"]) == (2, 2, 1)
                @test dst["valid_time"][:] == [tidx]

                NCDatasets.Dataset(src_path, "r") do src
                    @test dst["u"].var[:, :, 1] == src["u"].var[:, :, tidx]
                    @test dst["longitude"][:] == src["longitude"][:]
                end
            end
        end
    end

    @testset "split_era5_nc produces per-variable files" begin
        mktempdir() do dir
            src_path = joinpath(dir, "src.nc")
            write_synthetic_era5_nc(src_path; Nx=2, Ny=2, Nt=1)

            pairs = [
                ("u", joinpath(dir, "u_only.nc")),
                ("v", joinpath(dir, "v_only.nc")),
                ("missing_var", joinpath(dir, "should_not_exist.nc")),
            ]

            CDSExt.split_era5_nc(src_path, pairs, coord_vars)

            @test !isfile(joinpath(dir, "should_not_exist.nc"))

            for (vname, dst_path) in pairs[1:2]
                @test isfile(dst_path)
                NCDatasets.Dataset(dst_path, "r") do dst
                    @test haskey(dst, vname)
                    other = vname == "u" ? "v" : "u"
                    @test !haskey(dst, other)
                    NCDatasets.Dataset(src_path, "r") do src
                        @test dst[vname].var[:] == src[vname].var[:]
                    end
                end
            end
        end
    end

    @testset "split_era5_nc_multistep produces per-(var,timestep) files" begin
        mktempdir() do dir
            src_path = joinpath(dir, "src.nc")
            write_synthetic_era5_nc(src_path; Nx=2, Ny=2, Nt=3)

            triples = [
                ("u", 1, joinpath(dir, "u_t1.nc")),
                ("u", 3, joinpath(dir, "u_t3.nc")),
                ("v", 2, joinpath(dir, "v_t2.nc")),
                # Variable not present in source — silently skipped, no file.
                ("missing_var", 1, joinpath(dir, "should_not_exist.nc")),
            ]
            time_dimnames = Set(["valid_time"])

            CDSExt.split_era5_nc_multistep(src_path, triples, coord_vars, time_dimnames)

            # The skipped variable produces no output.
            @test !isfile(joinpath(dir, "should_not_exist.nc"))

            for (vname, tidx, dst_path) in triples[1:3]
                @test isfile(dst_path)
                NCDatasets.Dataset(dst_path, "r") do dst
                    @test haskey(dst, vname)
                    @test dst.dim["valid_time"] == 1
                    @test haskey(dst, "longitude")
                    @test haskey(dst, "latitude")
                    # The other ERA5 variable should not have leaked in.
                    other = vname == "u" ? "v" : "u"
                    @test !haskey(dst, other)

                    NCDatasets.Dataset(src_path, "r") do src
                        @test dst[vname].var[:, :, 1] == src[vname].var[:, :, tidx]
                    end
                end
            end
        end
    end
end

@testset "ERA5 CDSAPIExt is_zip and foreach_nc" begin
    @testset "is_zip" begin
        mktempdir() do tmp
            # File starting with the ZIP magic header
            zip_path = joinpath(tmp, "fake.zip")
            open(zip_path, "w") do io
                write(io, UInt8[0x50, 0x4b, 0x03, 0x04, 0x00, 0x00])
            end
            @test CDSExt.is_zip(zip_path) == true

            # File with arbitrary non-magic bytes (NetCDF-3 starts with "CDF\x01")
            nc_path = joinpath(tmp, "fake.nc")
            open(nc_path, "w") do io
                write(io, UInt8[0x43, 0x44, 0x46, 0x01])
            end
            @test CDSExt.is_zip(nc_path) == false

            # Short file (<4 bytes) — length check guards against false positives
            short_path = joinpath(tmp, "short.bin")
            open(short_path, "w") do io
                write(io, UInt8[0x50, 0x4b])   # only 2 of the 4 magic bytes
            end
            @test CDSExt.is_zip(short_path) == false
        end
    end

    @testset "foreach_nc — non-zip path calls f exactly once" begin
        mktempdir() do tmp
            nc_path = joinpath(tmp, "data.nc")
            touch(nc_path)

            received = String[]
            CDSExt.foreach_nc(p -> push!(received, p), nc_path, tmp)

            @test received == [nc_path]
        end
    end

    @testset "foreach_nc — zip path extracts and visits each .nc" begin
        mktempdir() do tmp
            # Build a ZIP fixture containing two .nc files (and a non-.nc file
            # that should be ignored).
            nc1 = joinpath(tmp, "a.nc"); touch(nc1)
            nc2 = joinpath(tmp, "b.nc"); touch(nc2)
            other = joinpath(tmp, "readme.txt"); touch(other)

            zip_path = joinpath(tmp, "bundle.zip")
            run(`zip -j -q $zip_path $nc1 $nc2 $other`)

            received = String[]
            CDSExt.foreach_nc(p -> push!(received, basename(p)), zip_path, tmp)

            @test sort(received) == ["a.nc", "b.nc"]   # readme.txt filtered out
        end
    end
end
