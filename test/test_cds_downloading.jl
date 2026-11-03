include("runtests_setup.jl")
include("download_utils.jl")

using CDSAPI
using Dates
using NCDatasets

using NumericalEarth.DataWrangling: metadata_path, BoundingBox
using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5MonthlySingleLevel,
                                         ERA5_dataset_variable_names, ERA5_netcdf_variable_names
using NumericalEarth.DataWrangling.ERA5: ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                         ERA5_all_pressure_levels, ERA5PL_dataset_variable_names,
                                         ERA5PL_netcdf_variable_names, pressure_field
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
            download(metadatum)
        end
        @test isfile(filepath)

        ds = NCDataset(filepath)

        @test haskey(ds, "t2m")

        @test haskey(ds, "longitude")
        @test haskey(ds, "latitude")
        @test haskey(ds, "time") || haskey(ds, "valid_time")

        lon = ds["longitude"][:]
        lat = ds["latitude"][:]

        @test minimum(lon) ≥ -1  # Allow some tolerance
        @test maximum(lon) ≤ 6
        @test minimum(lat) ≥ 39
        @test maximum(lat) ≤ 46

        t2m = ds["t2m"]
        @test ndims(t2m) ≥ 2

        close(ds)

        # Note: leave `filepath` in place; downstream surface-level testsets reuse it.
    end

    for arch in test_architectures
        A = typeof(arch)

        @testset "Field creation from ERA5 on $A" begin
            variable = :temperature
            metadatum = Metadatum(variable; dataset, region, date=start_date)

            # Download if not present (falls back to NumericalEarthArtifacts if CDS is unreachable)
            filepath = metadata_path(metadatum)
            isfile(filepath) || download_dataset_with_fallback(filepath; dataset_name="ERA5Hourly $variable") do
                download(metadatum)
            end

            # Create a Field from the downloaded data
            ψ = Field(metadatum, arch)

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
                download(metadatum)
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

            download(meta)
            @test isfile(filepath)

            # Verify the NetCDF has a pressure_level dimension and the right variable
            ds_nc = NCDataset(filepath)
            @test haskey(ds_nc, "t")
            @test haskey(ds_nc, "pressure_level") || haskey(ds_nc, "level")
            close(ds_nc)

            f = Field(meta, arch)
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
