include("runtests_setup.jl")
include("download_utils.jl")

using CDSAPI
using Dates
using NCDatasets

using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5MonthlySingleLevel, ERA5_dataset_variable_names
using NumericalEarth.DataWrangling.ERA5: ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                         ERA5_all_pressure_levels, ERA5PL_dataset_variable_names,
                                         pressure_field
using NumericalEarth.DataWrangling: metadata_path, download_dataset

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

        # Clean up
        rm(filepath; force=true)
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

            # Clean up
            rm(filepath; force=true)
            inpainted_path = NumericalEarth.DataWrangling.inpainted_metadata_path(metadatum)
            isfile(inpainted_path) && rm(inpainted_path; force=true)
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
            isfile(filepath) && rm(filepath; force=true)

            download_dataset(meta_z)
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
