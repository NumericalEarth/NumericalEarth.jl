include("runtests_setup.jl")

using Oceananigans
using Statistics
using JLD2
using NumericalEarth

using NumericalEarth.Bathymetry: remove_minor_basins!,
                                 BathymetryRegridding,
                                 cache_filename,
                                 load_bathymetry_cache,
                                 save_bathymetry_cache
using NumericalEarth.DataWrangling.ETOPO

@testset "Bathymetry construction and smoothing" begin
    @info "Testing Bathymetry construction and smoothing..."
    for arch in test_architectures
        ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022())

        # Testing downloading
        NumericalEarth.DataWrangling.download_dataset(ETOPOmetadata)
        @test isfile(metadata_path(ETOPOmetadata))

        grid = LatitudeLongitudeGrid(arch;
                                     size = (100, 100, 10),
                                     longitude = (0, 100),
                                     latitude = (0, 50),
                                     z = (-6000, 0))

        # Test that remove_minor_basins!(Z, Inf) does nothing
        control_bottom_height = regrid_bathymetry(grid, ETOPOmetadata)
        bottom_height = deepcopy(control_bottom_height)
        @test_throws ArgumentError remove_minor_basins!(bottom_height, Inf)

        # A fictitiously large number which should presumably keep all the basins
        remove_minor_basins!(bottom_height, 10000000)
        @test parent(bottom_height) == parent(control_bottom_height)

        # Test that remove_minor_basins!(Z, 2) remove the correct number of Basins
        bottom_height = Field{Center, Center, Nothing}(grid)
        control_bottom_height = Field{Center, Center, Nothing}(grid)

        # A two-basins bathymetry
        bottom(x, y) = - 1000 * Int((x < 10) | (x > 50))

        set!(bottom_height, bottom)
        set!(control_bottom_height, bottom)

        # This should have not changed anything
        remove_minor_basins!(bottom_height, 2)
        @test parent(bottom_height) == parent(control_bottom_height)

        # This should have removed the left basin
        remove_minor_basins!(bottom_height, 1)

        # The remaining bottom cells that are not immersed should be only on the right hand side
        # The left half of the domain should be fully immersed, i.e., bottom == 0
        @test sum(view(bottom_height, 1:50, :, 1)) == 0

        # While the right side should be not immersed, with a mean bottom depth
        # of -1000 meters
        @test mean(view(bottom_height, 51:100, :, 1)) == -1000

        grid = LatitudeLongitudeGrid(arch;
                                     size = (200, 200, 10),
                                     longitude = (0, 100),
                                     latitude = (-10, 50),
                                     z = (-6000, 0))

        control_bottom_height = regrid_bathymetry(grid)
        interpolated_bottom_height = regrid_bathymetry(grid; interpolation_passes=10)

        # Testing that multiple passes _do_ change the solution when coarsening the grid
        @test parent(control_bottom_height) != parent(interpolated_bottom_height)
    end
end

@testset "BathymetryRegridding configuration" begin
    @info "Testing BathymetryRegridding configuration..."

    grid = LatitudeLongitudeGrid(CPU();
                                 size = (100, 100, 10),
                                 longitude = (0, 100),
                                 latitude = (0, 50),
                                 z = (-6000, 0))

    metadata = Metadatum(:bottom_height, dataset=ETOPO2022())

    # Test construction and equality
    config1 = BathymetryRegridding(grid, metadata)
    config2 = BathymetryRegridding(grid, metadata)
    @test config1 == config2
    @test hash(config1) == hash(config2)

    # Test that different parameters produce different configs
    config3 = BathymetryRegridding(grid, metadata; interpolation_passes=5)
    @test config1 != config3
    @test hash(config1) != hash(config3)

    config4 = BathymetryRegridding(grid, metadata; minimum_depth=10)
    @test config1 != config4

    # Test cache filename: same config → same filename
    @test cache_filename(config1) == cache_filename(config2)
    # Different config → different filename
    @test cache_filename(config1) != cache_filename(config3)

    # Test JLD2 round-trip of BathymetryRegridding
    tmpfile = tempname() * ".jld2"
    jldopen(tmpfile, "w") do file
        file["config"] = config1
    end
    loaded_config = jldopen(tmpfile, "r") do file
        file["config"]
    end
    rm(tmpfile)
    @test loaded_config == config1
end

@testset "Bathymetry caching round-trip" begin
    @info "Testing bathymetry caching round-trip..."

    grid = LatitudeLongitudeGrid(CPU();
                                 size = (100, 100, 10),
                                 longitude = (0, 100),
                                 latitude = (0, 50),
                                 z = (-6000, 0))

    # First call computes and caches
    result1 = regrid_bathymetry(grid; cache=true)

    # Second call should load from cache and produce the same result
    result2 = regrid_bathymetry(grid; cache=true)
    @test parent(result1) == parent(result2)

    # Different parameters should produce different results (cache invalidation)
    result3 = regrid_bathymetry(grid; cache=true, interpolation_passes=5)
    @test parent(result1) != parent(result3)

    # cache=false should still produce correct results
    result4 = regrid_bathymetry(grid; cache=false)
    @test parent(result1) == parent(result4)
end
