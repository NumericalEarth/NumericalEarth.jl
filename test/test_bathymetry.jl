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
                                 ATLANTIC_OCEAN_BARRIERS
using NumericalEarth.DataWrangling: BoundingBox
using NumericalEarth.DataWrangling.ETOPO
using Statistics

@testset "Bathymetry construction and smoothing" begin
    @info "Testing Bathymetry construction and smoothing..."
    for arch in test_architectures
        ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022())
        filepath = metadata_path(ETOPOmetadata)

        # Testing downloading
        download_dataset_with_fallback(filepath; dataset_name="ETOPO2022") do
            NumericalEarth.DataWrangling.download_dataset(ETOPOmetadata)
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
end

@testset "Barrier geometry" begin
    @info "Testing barrier geometry utilities..."

    # A barrier is a `BoundingBox` whose horizontal extent defines the rectangle
    # to be treated as land during connected-component labeling.
    barrier = BoundingBox(longitude=(-10, 10), latitude=(-5, 5))
    @test barrier.longitude == (-10, 10)
    @test barrier.latitude  == (-5, 5)

    # Test meridional_barrier constructor (longitude, south, north)
    meridional = meridional_barrier(20, -36, -30)
    @test meridional.longitude == (19, 21)   # 20 ± 2/2
    @test meridional.latitude  == (-36, -30)

    # Test meridional_barrier with custom width
    meridional_wide = meridional_barrier(20, -36, -30; width=4)
    @test meridional_wide.longitude == (18, 22)

    # Latitude band covering the full longitude range
    band = BoundingBox(longitude=(-180, 180), latitude=(-60, -55))
    @test band.longitude == (-180, 180)
    @test band.latitude  == (-60, -55)
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

        # Regrid real bathymetry onto this grid
        bottom_height = regrid_bathymetry(grid)
        ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

        # Test labeling without barriers - all oceans should have the same label
        # (they're connected via the Southern Ocean)
        labels_no_barrier = label_ocean_basins(ibg)

        # Find labels at Atlantic and Pacific seed points
        atlantic_label = find_label_at_point(labels_no_barrier, ibg, -30.0, 0.0)
        pacific_label = find_label_at_point(labels_no_barrier, ibg, -170.0, 0.0)

        # Without barriers, Atlantic and Pacific should have the same label
        # (connected via Southern Ocean)
        @test atlantic_label == pacific_label
        @test atlantic_label > 0  # Should find a valid basin

        # Test labeling with barriers - oceans should be separated
        labels_with_barrier = label_ocean_basins(ibg; barriers=ATLANTIC_OCEAN_BARRIERS)

        # With barriers, Atlantic should still be found
        atlantic_label_with_barrier = find_label_at_point(labels_with_barrier, ibg, -30.0, 0.0)
        @test atlantic_label_with_barrier > 0
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

        # Test atlantic_ocean_basin creation
        atlantic = atlantic_ocean_basin(ibg)
        @test atlantic isa Basin
        @test sum(interior(atlantic.mask)) > 0  # Should have some ocean cells

        mask = on_architecture(CPU(), atlantic.mask)

        # Test that the mask is properly bounded
        # Atlantic mask should not include cells in the Pacific
        # (seed point at -170°, 0° should be 0)
        pacific_point_i = findfirst(i -> -175 < i < -165, range(-180, 180, length=360))
        equator_j = 90  # equator for 180 latitude points
        if !isnothing(pacific_point_i)
            @test mask[pacific_point_i, equator_j, 1] == 0
        end
    end
end
