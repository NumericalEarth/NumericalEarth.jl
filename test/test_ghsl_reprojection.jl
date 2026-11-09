include("runtests_setup.jl")

using ArchGDAL
using Downloads: Downloads
using Profile: Profile
using ZipFile: ZipFile
using NumericalEarth.Lands: aerodynamic_parameters
using NumericalEarth.DataWrangling.GHSL: GHSL, ghsl_tile_index, ghsl_tile_url, ghsl_tile_tif_name,
                                         native_resolution, ghsl_pixel_window,
                                         longitude_latitude_to_mollweide, MOLLWEIDE_RADIUS, GHSL_ORIGIN_X, GHSL_TILE_SIZE

function write_ghsl_tiles(dataset, surface, height; longitude, latitude, west, north)
    directory = joinpath(GHSL.download_GHSL_cache, "tiles")
    mkpath(directory)
    row, column = ghsl_tile_index(longitude, latitude)
    for (product, values) in ((dataset.surface, surface), (dataset.height, height))
        step = native_resolution(product)
        filename = ghsl_tile_tif_name(product, row, column)
        path = joinpath(directory, filename)
        ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"), width = size(values, 1),
                        height = size(values, 2), nbands = 1, dtype = Float64) do raster
            ArchGDAL.setgeotransform!(raster, [west, step, 0, north, 0, -step] .* 1.0)
            ArchGDAL.setproj!(raster, ArchGDAL.toWKT(ArchGDAL.importUserInput("ESRI:54009")))
            band = ArchGDAL.getband(raster, 1)
            ArchGDAL.setnodatavalue!(band, 65535)
            ArchGDAL.write!(band, values)
        end
        archive = ZipFile.Writer(joinpath(directory, basename(ghsl_tile_url(product, row, column))))
        try
            write(ZipFile.addfile(archive, filename), read(path))
        finally
            close(archive)
        end
    end
end

function with_ghsl_tiles(f, dataset, surface, height;
                         longitude = 0.005, latitude = 0.005, west = 0, north = 1200)
    default_cache = GHSL.download_GHSL_cache
    mktempdir() do cache
        setglobal!(GHSL, :download_GHSL_cache, cache)
        try
            write_ghsl_tiles(dataset, surface, height; longitude, latitude, west, north)
            f()
        finally
            setglobal!(GHSL, :download_GHSL_cache, default_cache)
        end
    end
end

@testset "GHSL conserves native built area and volume across longitude conventions" begin
    for arch in test_architectures, resolution in (GHSBuiltS100m, GHSBuiltS10m),
        (longitude, latitude) in ((-74, 40), (140, 36), (-120, -60), (-118.45, 34.05)),
        longitude_shift in (0, 360)
        dataset = GHSBuilt(surface = GHSBuiltS(; resolution))
        step = native_resolution(dataset.surface)
        region = BoundingBox(longitude = longitude .+ (-0.025, 0.025),
                             latitude = latitude .+ (-0.025, 0.025))
        west, east, south, north = ghsl_pixel_window(region, 100)
        surface = zeros(Int((east - west) / step), Int((north - south) / step))
        height = fill(20.0, Int((east - west) / 100), Int((north - south) / 100))
        x, y = longitude_latitude_to_mollweide(longitude, latitude)
        for (offset, fraction, building_height) in ((0, 0.2, 10), (200, 0.6, 30))
            i = floor(Int, (x + offset - west) / step) + 1
            j = floor(Int, (north - y) / step) + 1
            surface[i, j] = fraction * step^2
            i = floor(Int, (x + offset - west) / 100) + 1
            j = floor(Int, (north - y) / 100) + 1
            height[i, j] = building_height
        end
        longitude_faces = longitude + longitude_shift .+ [-0.015, -0.003, 0.015]
        latitude_faces = latitude .+ [-0.015, 0.004, 0.015]
        region = BoundingBox(longitude = region.longitude .+ longitude_shift, latitude = region.latitude)
        grid = LatitudeLongitudeGrid(arch; size = (2, 2),
                                     longitude = longitude_faces, latitude = latitude_faces,
                                     topology = (Bounded, Bounded, Flat))
        areas = MOLLWEIDE_RADIUS^2 .* deg2rad.(diff(longitude_faces)) .* diff(sind.(latitude_faces))'
        with_ghsl_tiles(dataset, surface, height; longitude, latitude, west, north) do
            for maximum_raster_cells in (100_000_000, 10_000)
                properties = building_morphometry(grid, dataset; maximum_raster_cells)
                fraction = Array(interior(properties.plan_area_index, :, :, 1))
                mean_height = Array(interior(properties.mean_building_height, :, :, 1))
                @test sum(areas .* fraction) ≈ 0.8step^2 rtol = 1e-6
                @test sum(areas .* fraction .* mean_height) ≈ 20step^2 rtol = 1e-6
            end
            regridded_fraction = CenterField(grid)
            regrid!(regridded_fraction, Field(Metadatum(:built_up_fraction; dataset, region), arch))
            @test sum(areas .* Array(interior(regridded_fraction, :, :, 1))) ≈ 0.8step^2 rtol = 1e-6
        end
        fill!(surface, 0.5step^2)
        fill!(height, 20)
        with_ghsl_tiles(dataset, surface, height; longitude, latitude, west, north) do
            properties = building_morphometry(grid, dataset)
            @test all(isapprox.(interior(properties.plan_area_index), 0.5; rtol=1e-6))
            @test all(isapprox.(interior(properties.mean_building_height), 20; rtol=1e-6))
            for (name, product, expected) in ((:built_up_fraction, dataset.surface, 0.5),
                                             (:building_height, dataset.height, 20))
                field = Field(Metadatum(name; dataset = product, region = BoundingBox(grid)), arch)
                @test all(isapprox.(interior(field), expected; rtol=1e-6))
            end
        end
    end
end

# Equatorial Mollweide scales for a small domain wholly inside the synthetic tiles.
longitude_per_meter = rad2deg(π / (2sqrt(2) * 6378137))
latitude_per_meter = rad2deg(2sqrt(2) / (π * 6378137))

@testset "GHSL preserves density-height covariance on $arch at $resolution with size $grid_size" for arch in test_architectures,
    resolution in (GHSBuiltS100m, GHSBuiltS10m), grid_size in ((4, 2), (1, 1))
    dataset = GHSBuilt(surface = GHSBuiltS(; resolution))
    pixel_size = native_resolution(dataset.surface)
    alternating = isodd.((1:12) .+ (1:12)')
    surface = repeat(ifelse.(alternating, 0.2, 0.6); inner = (100 ÷ pixel_size, 100 ÷ pixel_size)) .* pixel_size^2
    height = ifelse.(alternating, 10.0, 30.0)
    grid = LatitudeLongitudeGrid(arch; size = grid_size,
                                 longitude = (400, 800) .* longitude_per_meter,
                                 latitude = (400, 800) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    with_ghsl_tiles(dataset, surface, height) do
        for maximum_raster_cells in (100_000_000, 137)
            properties = building_morphometry(grid, dataset; maximum_raster_cells)
            fraction = interior(properties.plan_area_index)
            mean_height = interior(properties.mean_building_height)
            # Every mixture of (λᵖ, λᵖh) = (0.2, 2 m) and (0.6, 18 m) satisfies V = 40λᵖ - 6 m.
            @test any(0.3 .< fraction .< 0.5)
            @test fraction .* mean_height ≈ 40fraction .- 6 rtol = 1e-5
        end
    end
end

@testset "GHSL keeps footprints without heights on $arch at $resolution" for arch in test_architectures,
                                                                              resolution in (GHSBuiltS100m, GHSBuiltS10m)
    dataset = GHSBuilt(surface = GHSBuiltS(; resolution))
    pixel_size = native_resolution(dataset.surface)
    # Each cell spans two 100 m columns, one of them missing (65535) or zero in one product.
    grid = LatitudeLongitudeGrid(arch; size = (2, 2),
                                 longitude = (400, 800) .* longitude_per_meter,
                                 latitude = (400, 800) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    for missing_height in (true, false), missing_value in (65535, 0)
        surface = fill(5000.0, 12, 12)
        height = fill(20.0, 12, 12)
        missing_values = missing_height ? height : surface
        missing_values[1:2:end, :] .= missing_value
        surface = repeat(surface; inner = (100 ÷ pixel_size, 100 ÷ pixel_size))
        surface .= ifelse.(surface .== 5000, 0.5pixel_size^2, surface)
        with_ghsl_tiles(dataset, surface, height) do
            properties = building_morphometry(grid, dataset)
            # A footprint without a height stays built; a pixel without a footprint does not.
            expected_fraction = missing_height ? 0.5 : 0.25
            @test all(isapprox.(interior(properties.plan_area_index), expected_fraction; rtol=1e-6))
            @test all(isapprox.(interior(properties.mean_building_height), 20; rtol=1e-6))
        end
    end
end

@testset "GHSL pools measured heights across missing pixels on $arch at $resolution" for arch in test_architectures,
                                                                                          resolution in (GHSBuiltS100m, GHSBuiltS10m)
    dataset = GHSBuilt(surface = GHSBuiltS(; resolution))
    pixel_size = native_resolution(dataset.surface)
    fractions = repeat([0.2, 0.6, 0.4], 4, 12)
    surface = repeat(fractions; inner = (100 ÷ pixel_size, 100 ÷ pixel_size)) .* pixel_size^2
    for missing_height in (0.0, 65535.0)
        height = repeat([10.0, 30.0, missing_height], 4, 12)
        with_ghsl_tiles(dataset, surface, height) do
            for grid_size in ((1, 1), (6, 1))
                grid = LatitudeLongitudeGrid(arch; size = grid_size,
                                             longitude = (300, 900) .* longitude_per_meter,
                                             latitude = (550, 650) .* latitude_per_meter,
                                             topology = (Bounded, Bounded, Flat))
                neighborhood = grid.radius * deg2rad(600longitude_per_meter)
                fields = urban_roughness(grid, dataset; surface_layer_height = 100, neighborhood)
                roughness, displacement = aerodynamic_parameters(MorphometricRoughness(), 0.4, 25.0)
                @test all(isapprox.(interior(fields.building_height), 25; rtol=1e-5))
                @test all(isapprox.(interior(fields.momentum_roughness_length), roughness; rtol=1e-5))
                @test all(isapprox.(interior(fields.zero_plane_displacement), displacement; rtol=1e-5))
            end
        end
    end
end

@testset "GHSL retains 10 m footprints on $arch" for arch in test_architectures
    dataset = GHSBuilt()
    surface = repeat(ifelse.(1:120 .<= 60, 0.0, 100.0), 1, 120)
    fine_grid = LatitudeLongitudeGrid(arch; size = (4, 1),
                                      longitude = (580, 620) .* longitude_per_meter,
                                      latitude = (500, 600) .* latitude_per_meter,
                                      topology = (Bounded, Bounded, Flat))
    with_ghsl_tiles(dataset, surface, fill(20.0, 12, 12)) do
        properties = building_morphometry(fine_grid, dataset)
        fraction = vec(Array(interior(properties.plan_area_index)))
        @test first(fraction) ≈ 0 atol = 1e-6
        @test last(fraction) ≈ 1 atol = 1e-6
        built_height = interior(properties.mean_building_height)[interior(properties.plan_area_index) .> 0]
        @test all(isapprox.(built_height, 20; rtol=1e-6))
    end
end

@testset "GHSL uniform footprints survive offset window boundaries on $arch with size $grid_size" for arch in test_architectures,
                                                                                                    grid_size in ((2, 2), (1, 1))
    dataset = GHSBuilt(surface = GHSBuiltS(resolution = GHSBuiltS10m))
    offset_grid = LatitudeLongitudeGrid(arch; size = grid_size,
                                        longitude = (450, 1050) .* longitude_per_meter,
                                        latitude = (250, 850) .* latitude_per_meter,
                                        topology = (Bounded, Bounded, Flat))
    with_ghsl_tiles(dataset, fill(50.0, 120, 120), fill(20.0, 12, 12)) do
        for maximum_raster_cells in (100_000_000, 1000)
            properties = building_morphometry(offset_grid, dataset; maximum_raster_cells)
            @test all(isapprox.(interior(properties.plan_area_index), 0.5; rtol=1e-6))
            @test all(isapprox.(interior(properties.mean_building_height), 20; rtol=1e-6))
        end
    end
end

@testset "GHSL windows ending at or crossing 180°E on $arch at $resolution with longitude shift $longitude_shift" for arch in test_architectures,
    resolution in (GHSBuiltS100m, GHSBuiltS10m), longitude_shift in (0, 360)
    dataset = GHSBuilt(surface = GHSBuiltS(; resolution))
    step = native_resolution(dataset.surface)
    latitude = 17.8
    # Uniform tiles on each side of the antimeridian: half built at 20 m east of it, a fifth built at 10 m west.
    east_tile, west_tile = map((((179.95, 180), 0.5, 20.0), ((-180, -179.95), 0.2, 10.0))) do (longitudes, fraction, building_height)
        region = BoundingBox(longitude = longitudes, latitude = latitude .+ (-0.025, 0.025))
        west, east, south, north = ghsl_pixel_window(region, 100)
        (; surface = fill(fraction * step^2, Int((east - west) / step), Int((north - south) / step)),
           height = fill(building_height, Int((east - west) / 100), Int((north - south) / 100)),
           longitude = sum(longitudes) / 2, west, north)
    end
    with_ghsl_tiles(dataset, east_tile.surface, east_tile.height; east_tile.longitude, latitude, east_tile.west, east_tile.north) do
        write_ghsl_tiles(dataset, west_tile.surface, west_tile.height; west_tile.longitude, latitude, west_tile.west, west_tile.north)
        for longitude_faces in ([179.97, 179.985, 180], [179.97, 179.99, 180.03])
            grid = LatitudeLongitudeGrid(arch; size = (2, 2), longitude = longitude_faces .+ longitude_shift,
                                         latitude = latitude .+ (-0.015, 0.015), topology = (Bounded, Bounded, Flat))
            properties = building_morphometry(grid, dataset)
            east_width = diff(min.(longitude_faces, 180))
            west_width = diff(max.(longitude_faces, 180))
            built_width = 0.5east_width .+ 0.2west_width
            @test Array(interior(properties.plan_area_index, :, :, 1)) ≈ repeat(built_width ./ diff(longitude_faces), 1, 2) rtol = 1e-6
            @test Array(interior(properties.mean_building_height, :, :, 1)) ≈ repeat((10east_width .+ 2west_width) ./ built_width, 1, 2) rtol = 1e-6
            for (name, product, east_value, west_value) in ((:built_up_fraction, dataset.surface, 0.5, 0.2),
                                                           (:building_height, dataset.height, 20, 10))
                field = Field(Metadatum(name; dataset = product, region = BoundingBox(grid)), arch)
                expected = ifelse.(mod.(Array(λnodes(field.grid, Center())), 360) .< 180, east_value, west_value)
                @test Array(interior(field, :, :, 1)) ≈ repeat(expected, 1, size(field, 2)) rtol = 1e-6
            end
        end
    end
end

@testset "GHSL bounds raster allocations within a single coarse cell" begin
    dataset = GHSBuilt()
    grid = LatitudeLongitudeGrid(CPU(); size = (1, 1),
                                 longitude = (450, 1050) .* longitude_per_meter,
                                 latitude = (250, 850) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    maximum_raster_cells = 1000
    with_ghsl_tiles(dataset, fill(50.0, 120, 120), fill(20.0, 12, 12)) do
        building_morphometry(grid, dataset; maximum_raster_cells)
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=1.0 building_morphometry(grid, dataset; maximum_raster_cells)
        allocations = Profile.Allocs.fetch().allocs
        raster_allocations = filter(allocation -> allocation.type === Profile.Allocs.BufferType ||
                                                 allocation.type <: AbstractArray{Float32}, allocations)
        @test !isempty(raster_allocations)
        @test maximum(allocation.size for allocation in raster_allocations) <= Base.summarysize(zeros(Float32, maximum_raster_cells))
        Profile.Allocs.clear()
    end
end

@testset "GHSL missing ocean tiles contribute no buildings on $arch" for arch in test_architectures
    dataset = GHSBuilt()
    coast = GHSL_ORIGIN_X + 18GHSL_TILE_SIZE
    longitude = (coast + 300) * longitude_per_meter
    grid = LatitudeLongitudeGrid(arch; size = (2, 1),
                                 longitude = (coast - 300, coast + 300) .* longitude_per_meter,
                                 latitude = (400, 800) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    ocean_grid = LatitudeLongitudeGrid(arch; size = (1, 1),
                                       longitude = (coast - 500, coast - 200) .* longitude_per_meter,
                                       latitude = (400, 800) .* latitude_per_meter,
                                       topology = (Bounded, Bounded, Flat))
    downloader = Downloads.Downloader()
    downloader.easy_hook = (easy, info) -> throw(Downloads.RequestError(info.url, 0, "",
        Downloads.Response("https", info.url, 404, "Not Found", Pair{String, String}[])))
    previous_downloader = Downloads.DOWNLOADER[]
    Downloads.default_downloader!(downloader)
    try
        with_ghsl_tiles(dataset, fill(50.0, 120, 120), fill(20.0, 12, 12); longitude, west = coast) do
            properties = building_morphometry(grid, dataset; maximum_raster_cells = 1000)
            @test vec(Array(interior(properties.plan_area_index))) ≈ [0, 0.5] atol = 1e-6
            @test last(Array(interior(properties.mean_building_height))) ≈ 20
            properties = building_morphometry(ocean_grid, dataset; maximum_raster_cells = 1000)
            @test all(iszero, interior(properties.plan_area_index))
            @test all(iszero, interior(properties.mean_building_height))
            for (name, product) in ((:built_up_fraction, dataset.surface), (:building_height, dataset.height))
                metadata = Metadatum(name; dataset = product, region = BoundingBox(ocean_grid))
                field = Field(metadata, arch)
                @test all(isnan, interior(field))
            end

            row, column = ghsl_tile_index(longitude, 0.005)
            rm(joinpath(GHSL.download_GHSL_cache, "tiles", basename(ghsl_tile_url(dataset.height, row, column))))
            properties = building_morphometry(grid, dataset; maximum_raster_cells = 1000)
            @test vec(Array(interior(properties.plan_area_index))) ≈ [0, 0.5] atol = 1e-6
            @test all(iszero, interior(properties.measured_plan_area_index))
            @test isnan(last(Array(interior(properties.mean_building_height))))
            @test_throws ArgumentError urban_roughness(grid, dataset; surface_layer_height = 100, neighborhood = 100)
        end
    finally
        Downloads.default_downloader!(previous_downloader)
    end
end

@testset "GHSL includes the last periodic longitude cell on $arch" for arch in test_architectures
    dataset = GHSBuilt()
    grid = LatitudeLongitudeGrid(arch; size = (2, 2),
                                 longitude = (400, 800) .* longitude_per_meter,
                                 latitude = (400, 800) .* latitude_per_meter,
                                 topology = (Periodic, Bounded, Flat))
    with_ghsl_tiles(dataset, fill(50.0, 120, 120), fill(20.0, 12, 12)) do
        properties = building_morphometry(grid, dataset)
        @test all(interior(properties.plan_area_index) .≈ 0.5)
        @test all(interior(properties.mean_building_height) .≈ 20)
    end
end

@testset "GHSL urban roughness from native tiles on $arch with $FT" for arch in test_architectures, FT in (Float32, Float64)
    # Four 1200 m cells stacked south to north, each two 600 m rows: cell 1 holds a dense core in
    # its southern row, cell 2 a southern row alternating density and height by 100 m, cell 3 is
    # no-data, and cell 4 is uniformly built. Nothing varies east to west, where neighborhoods
    # overlap the lattice cells.
    row(y) = floor(Int, (y - 400) / 600) + 1
    odd_stripe(y) = isodd(floor(Int, y / 100))
    function built_surface(y)
        row(y) == 1 && return 50.0
        row(y) == 3 && return odd_stripe(y) ? 20.0 : 60.0
        row(y) in (5, 6) && return 65535.0
        return row(y) > 6 ? 50.0 : 0.0
    end
    building_height(y) = row(y) == 3 ? (odd_stripe(y) ? 10.0 : 30.0) : 20.0
    surface = [built_surface(5600 - 10j + 5) for i in 1:200, j in 1:560]
    height = [building_height(5600 - 100j + 50) for i in 1:20, j in 1:56]

    dataset = GHSBuilt()
    grid = LatitudeLongitudeGrid(arch, FT; size = (1, 4),
                                 longitude = (400, 1600) .* longitude_per_meter,
                                 latitude = (400, 5200) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    closures = (; variable = MorphometricRoughness(FT),
                  uniform = MorphometricRoughness(FT; height_distribution = UniformHeight()))
    # Neighborhoods as tall as the 600 m rows, in grid meters.
    neighborhood = grid.radius * deg2rad(600latitude_per_meter)
    with_ghsl_tiles(dataset, surface, height; north = 5600) do
        results = urban_roughness(grid, dataset; closure = closures, surface_layer_height = 100, neighborhood)
        for name in keys(closures)
            closure = closures[name]
            fields = map(field -> vec(Array(interior(field, :, :, 1))), results[name])
            core = aerodynamic_parameters(closure, FT(0.5), FT(20))
            # Built-area-weighted height, (0.2 · 10 + 0.6 · 30) / 0.8, at plan-area index 0.4.
            mixed = aerodynamic_parameters(closure, FT(0.4), FT(25))
            bare = aerodynamic_parameters(closure, 0, 0)
            urban = [1, 2, 4]
            @test fields.urban_momentum_roughness_length[urban] ≈ [core[1], mixed[1], core[1]] rtol = 1e-5
            @test fields.urban_zero_plane_displacement[urban] ≈ [core[2], mixed[2], core[2]] rtol = 1e-5
            f = [1/2, 1/2, 0, 1]
            @test fields.urban_fraction ≈ f rtol = 1e-5
            # The unbuilt halves of cells 1 and 2 are bare soil, blended with the urban half by drag.
            drag(ℓᵐ, d) = inv(log((100 - d) / ℓᵐ))^2
            urban_pairs = (core, mixed, bare, core)
            C = f .* [drag(pair...) for pair in urban_pairs] .+ (1 .- f) .* drag(bare...)
            @test drag.(fields.momentum_roughness_length, fields.zero_plane_displacement) ≈ C rtol = 1e-5
            @test fields.zero_plane_displacement ≈ f .* [drag(pair...) * pair[2] for pair in urban_pairs] ./ C rtol = 1e-5
            @test fields.building_height[urban] ≈ [20, 25, 20] rtol = 1e-5
            @test eltype(results[name].momentum_roughness_length) == FT
        end
        @test readdir(GHSL.download_GHSL_cache) == ["tiles"]
    end
end

@testset "Explicit GHSL raster retrieval" begin
    dataset = GHSBuilt()
    region = BoundingBox(longitude = (400, 800) .* longitude_per_meter,
                         latitude = (400, 800) .* latitude_per_meter)
    with_ghsl_tiles(dataset, fill(50.0, 120, 120), fill(20.0, 12, 12)) do
        for (name, expected) in ((:built_up_fraction, 0.5), (:building_volume, 10))
            field = Field(Metadatum(name; dataset, region), CPU())
            @test all(isapprox.(interior(field), expected; rtol=1e-5))
        end
    end
end
