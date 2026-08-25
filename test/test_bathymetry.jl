include("runtests_setup.jl")
include("download_utils.jl")

using JLD2
using NumericalEarth.Bathymetry: remove_minor_basins!,
                                 BathymetryRegridding,
                                 cache_filename,
                                 load_bathymetry_cache,
                                 save_bathymetry_cache,
                                 label_ocean_basins,
                                 find_label_at_point,
                                 Basin,
                                 atlantic_ocean_basin,
                                 meridional_barrier,
                                 atlantic_ocean_barriers
using NumericalEarth.DataWrangling: BoundingBox
using NumericalEarth.Bathymetry: remove_minor_basins!, bathymetry_regridding_key
using NumericalEarth.DataWrangling: field_cache_filename, save_field_cache
using NumericalEarth.DataWrangling.ETOPO
using Statistics

@testset "Topography smoothing preserves bounded constants" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch, Float32;
                               size = (8, 8),
                               extent = (1, 1),
                               topology = (Bounded, Bounded, Flat))

        elevation = Field{Center, Center, Nothing}(grid)
        set!(elevation, 1000)
        smooth_topography!(elevation; passes = 2)

        @test all(==(1000), Array(interior(elevation)))
    end
end

@testset "Bathymetry construction and smoothing" begin
    @info "Testing Bathymetry construction and smoothing..."
    for arch in test_architectures
        ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022())
        filepath = metadata_path(ETOPOmetadata)

        # Testing downloading
        download_dataset_with_fallback(filepath; dataset_name="ETOPO2022") do
            download(ETOPOmetadata)
        end
        @test isfile(filepath)

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

@testset "Bathymetry cache key" begin
    @info "Testing bathymetry cache keys..."

    grid = LatitudeLongitudeGrid(CPU();
                                 size = (100, 100, 10),
                                 longitude = (0, 100),
                                 latitude = (0, 50),
                                 z = (-6000, 0))

    metadata = Metadatum(:bottom_height, dataset=ETOPO2022())

    key(; height_above_water = nothing, minimum_depth = 0, interpolation_passes = 1, major_basins = 1) =
        bathymetry_regridding_key(grid, metadata; height_above_water, minimum_depth,
                                  interpolation_passes, major_basins)

    # Test construction and equality
    config1 = key()
    config2 = key()
    @test config1 == config2
    @test hash(config1) == hash(config2)

    # Test that different parameters produce different configs
    config3 = key(interpolation_passes = 5)
    @test config1 != config3
    @test hash(config1) != hash(config3)

    config4 = key(minimum_depth = 10)
    @test config1 != config4

    # Integer and float parameter values key identically
    @test key(minimum_depth = 10) == key(minimum_depth = 10.0)

    # Test cache filename: same config → same filename
    @test field_cache_filename(config1) == field_cache_filename(config2)
    # Different config → different filename
    @test field_cache_filename(config1) != field_cache_filename(config3)

    # Test JLD2 round-trip of the key
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

    # Use a grid size distinct from the first test block (100x100, 200x200)
    # to avoid loading GPU-computed cache on CPU (floating-point differences).
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (80, 80, 10),
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

    # overwrite_cache=true skips the lookup and refreshes the entry
    metadata = Metadatum(:bottom_height, dataset=ETOPO2022())
    config = bathymetry_regridding_key(grid, metadata;
                                       height_above_water = nothing, minimum_depth = 0,
                                       interpolation_passes = 1, major_basins = 1)
    save_field_cache(config, zeros(size(grid, 1), size(grid, 2)))
    poisoned = regrid_bathymetry(grid; cache=true)
    @test all(iszero, interior(poisoned))

    result5 = regrid_bathymetry(grid; cache=true, overwrite_cache=true)
    @test parent(result1) == parent(result5)

    result6 = regrid_bathymetry(grid; cache=true)
    @test parent(result1) == parent(result6)
end

@testset "Barrier geometry" begin
    @info "Testing barrier geometry utilities..."

    meridional = meridional_barrier(20, -36, -30)
    @test meridional.longitude == (19, 21)   # 20 ± width/2
    @test meridional.latitude  == (-36, -30)
    @test meridional_barrier(20, -36, -30; width=4).longitude == (18, 22)
end

@testset "Ocean basin labeling with barriers" begin
    @info "Testing ocean basin labeling with barriers..."

    for arch in test_architectures
        # Create a global grid
        grid = LatitudeLongitudeGrid(arch;
                                     size = (90, 45, 10),
                                     longitude = (-180, 180),
                                     latitude = (-90, 90),
                                     z = (-6000, 0))

        bottom_height = regrid_bathymetry(grid)
        ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

        # Unbarriered, the Atlantic and Pacific are one basin: they connect via the Southern Ocean.
        labels = label_ocean_basins(ibg)
        atlantic_label = find_label_at_point(labels, ibg, -30.0, 0.0)
        @test atlantic_label > 0
        @test atlantic_label == find_label_at_point(labels, ibg, -170.0, 0.0)

        barriered = label_ocean_basins(ibg; barriers=atlantic_ocean_barriers)
        @test find_label_at_point(barriered, ibg, -30.0, 0.0) > 0
    end
end

@testset "Basin creation" begin
    @info "Testing Basin creation..."

    for arch in test_architectures
        # Create a global grid at 1° resolution (needed to properly resolve
        # Central America and separate Atlantic from Pacific)
        grid = LatitudeLongitudeGrid(arch;
                                     size = (360, 180, 10),
                                     longitude = (-180, 180),
                                     latitude = (-90, 90),
                                     z = (-6000, 0))

        bottom_height = regrid_bathymetry(grid)
        ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

        atlantic = atlantic_ocean_basin(ibg)
        @test atlantic isa Basin
        @test sum(interior(atlantic.mask)) > 0

        # The Atlantic mask must exclude the Pacific.
        mask = on_architecture(CPU(), atlantic.mask)
        pacific_i = findfirst(λ -> -175 < λ < -165, range(-180, 180, length=360))
        @test !isnothing(pacific_i)
        @test mask[pacific_i, 90, 1] == 0
    end
end
