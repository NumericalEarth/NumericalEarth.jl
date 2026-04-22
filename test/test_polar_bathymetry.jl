include("runtests_setup.jl")

using NumericalEarth.DataWrangling.IBCSO
using NumericalEarth.DataWrangling.GEBCO
using NumericalEarth.DataWrangling.IBCAO
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces, z_interfaces,
                                    dataset_variable_name, validate_dataset_coverage
using NumericalEarth.Bathymetry: regrid_bathymetry

@testset "Polar bathymetry metadata interfaces" begin

    @testset "IBCSOv2 metadata" begin
        ds = IBCSOv2()
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (-90, -50)
        @test z_interfaces(ds) == (0, 1)
        @test size(ds) == (33812, 3757, 1)

        meta = Metadatum(:bottom_height, dataset=ds)
        @test dataset_variable_name(meta) == "z"
        @test endswith(metadata_filename(ds, :bottom_height, nothing, nothing), ".nc")
    end

    @testset "IBCSOv2 coverage validation" begin
        meta = Metadatum(:bottom_height, dataset=IBCSOv2())

        # Grid that extends north of -50°S should throw
        grid_bad = LatitudeLongitudeGrid(CPU();
                                         size = (10, 10, 1),
                                         longitude = (0, 360),
                                         latitude = (-60, 0),
                                         z = (-1, 0))
        @test_throws ErrorException validate_dataset_coverage(grid_bad, meta)

        # Grid entirely south of -50°S should pass
        grid_ok = LatitudeLongitudeGrid(CPU();
                                        size = (10, 10, 1),
                                        longitude = (0, 360),
                                        latitude = (-90, -55),
                                        z = (-1, 0))
        @test validate_dataset_coverage(grid_ok, meta) === nothing
    end

    @testset "GEBCO2024 metadata" begin
        ds = GEBCO2024()
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (-90, 90)
        @test z_interfaces(ds) == (0, 1)
        Nx, Ny, Nz = size(ds)
        @test Nx == 86400   # 360° at 15 arc-second
        @test Ny == 43200   # 180° at 15 arc-second
        @test Nz == 1

        meta = Metadatum(:bottom_height, dataset=ds)
        @test dataset_variable_name(meta) == "elevation"
        @test endswith(metadata_filename(ds, :bottom_height, nothing, nothing), ".nc")
    end

    @testset "IBCAOv5 metadata" begin
        ds = IBCAOv5()
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (64, 90)
        @test z_interfaces(ds) == (0, 1)
        @test size(ds) == (36000, 2600, 1)

        meta = Metadatum(:bottom_height, dataset=ds)
        @test dataset_variable_name(meta) == "z"
        @test endswith(metadata_filename(ds, :bottom_height, nothing, nothing), ".nc")
    end

    @testset "IBCAOv5 coverage validation" begin
        meta = Metadatum(:bottom_height, dataset=IBCAOv5())

        # Grid that extends south of 64°N should throw
        grid_bad = LatitudeLongitudeGrid(CPU();
                                         size = (10, 10, 1),
                                         longitude = (-20, 20),
                                         latitude = (50, 80),
                                         z = (-1, 0))
        @test_throws ErrorException validate_dataset_coverage(grid_bad, meta)

        # Grid entirely north of 64°N should pass
        grid_ok = LatitudeLongitudeGrid(CPU();
                                        size = (10, 10, 1),
                                        longitude = (-20, 20),
                                        latitude = (70, 85),
                                        z = (-1, 0))
        @test validate_dataset_coverage(grid_ok, meta) === nothing
    end

end

@testset "IBCSO regridding" begin
    @info "Testing IBCSO regridding (downloads ~1.5 GB on first run)..."

    # Drake Passage: open deep ocean well within IBCSO coverage
    grid = LatitudeLongitudeGrid(CPU();
                                  size = (20, 20, 1),
                                  longitude = (-70, -60),
                                  latitude = (-60, -55),
                                  z = (-6000, 0))

    meta = Metadatum(:bottom_height, dataset=IBCSOv2())
    bathy = regrid_bathymetry(grid, meta; cache=false)
    z = interior(bathy, :, :, 1)

    # All values should be finite (no NaN or Inf from interpolation gaps)
    @test all(isfinite, z)

    # Drake Passage is deep open ocean: all cells should be below sea level
    @test maximum(z) ≤ 0

    # Realistic ocean depths: deeper than 500 m, shallower than the deepest ocean
    @test minimum(z) > -12000
    @test minimum(z) < -500
end
