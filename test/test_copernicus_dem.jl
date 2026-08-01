include("runtests_setup.jl")

using NumericalEarth.DataWrangling.CopernicusDEM
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces, z_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename
using NumericalEarth.Bathymetry: regrid_bathymetry

# The actual data read requires a DestinE token, the Zarr extension, and network
# access, so only the dataset-interface and coverage-validation logic is exercised
# here. The regridding read path is verified manually / in a token-gated job.

@testset "Copernicus DEM metadata interfaces" begin

    @testset "GLO30 metadata" begin
        ds = GLO30()
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (-90, 90)
        @test z_interfaces(ds) == (0, 1)
        Nx, Ny, Nz = size(ds)
        @test Nx == 1296000   # 360° at 1 arc-second
        @test Ny == 648000    # 180° at 1 arc-second
        @test Nz == 1

        region = BoundingBox(longitude = (9, 11), latitude = (45, 47))
        meta = Metadatum(:bottom_height; dataset = ds, region)
        @test dataset_variable_name(meta) == "z"

        filename = metadata_filename(ds, :bottom_height, nothing, region)
        @test startswith(filename, "GLO30_")
        @test endswith(filename, ".nc")
    end

    @testset "GLO90 metadata" begin
        ds = GLO90()
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (-90, 90)
        Nx, Ny, Nz = size(ds)
        @test Nx == 432000   # 360° at 3 arc-second
        @test Ny == 216000   # 180° at 3 arc-second
        @test Nz == 1

        region = BoundingBox(longitude = (9, 11), latitude = (45, 47))
        filename = metadata_filename(ds, :bottom_height, nothing, region)
        @test startswith(filename, "GLO90_")
        @test endswith(filename, ".nc")
    end

    @testset "Region-keyed filenames are distinct" begin
        ds = GLO30()
        region_a = BoundingBox(longitude = (9, 11), latitude = (45, 47))
        region_b = BoundingBox(longitude = (0, 2), latitude = (50, 52))
        name_a = metadata_filename(ds, :bottom_height, nothing, region_a)
        name_b = metadata_filename(ds, :bottom_height, nothing, region_b)
        @test name_a != name_b
    end
end

@testset "Copernicus DEM requires a bounded region" begin
    # No region → must error (a global 30 m read is not allowed).
    meta_global = Metadatum(:bottom_height; dataset = GLO30())
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (10, 10, 1),
                                 longitude = (9, 11),
                                 latitude = (45, 47),
                                 z = (-1, 0))
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)
    @test_throws ErrorException regrid_bathymetry(grid, meta_global)

    # Bounded region → passes validation.
    region = BoundingBox(longitude = (9, 11), latitude = (45, 47))
    meta_region = Metadatum(:bottom_height; dataset = GLO30(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end
