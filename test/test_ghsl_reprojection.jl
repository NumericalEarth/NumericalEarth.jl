include("runtests_setup.jl")

using ArchGDAL
using ZipFile: ZipFile
using NumericalEarth.Lands: aerodynamic_parameters
using NumericalEarth.DataWrangling.GHSL: GHSL, ghsl_tile_index, ghsl_tile_url, ghsl_tile_tif_name,
                                         native_resolution, ghsl_pixel_window,
                                         longitude_latitude_to_mollweide, MOLLWEIDE_RADIUS

function with_ghsl_tiles(f, dataset, surface, height;
                         longitude = 0.005, latitude = 0.005, west = 0, north = 1200)
    default_cache = GHSL.download_GHSL_cache
    mktempdir() do cache
        setglobal!(GHSL, :download_GHSL_cache, cache)
        try
            directory = joinpath(cache, "tiles")
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
            f()
        finally
            setglobal!(GHSL, :download_GHSL_cache, default_cache)
        end
    end
end

@testset "GHSL conserves native built area and volume away from the equator" begin
    for arch in test_architectures, resolution in (GHSBuiltS100m, GHSBuiltS10m),
        (longitude, latitude) in ((-74, 40), (140, 36), (-120, -60))
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
        longitude_faces = longitude .+ [-0.015, -0.003, 0.015]
        latitude_faces = latitude .+ [-0.015, 0.004, 0.015]
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
        end
    end
end

# Equatorial Mollweide scales for a small domain wholly inside the synthetic tiles.
longitude_per_meter = rad2deg(π / (2sqrt(2) * 6378137))
latitude_per_meter = rad2deg(2sqrt(2) / (π * 6378137))

@testset "GHSL preserves density-height covariance on $arch at $resolution" for arch in test_architectures,
                                                                             resolution in (GHSBuiltS100m, GHSBuiltS10m)
    dataset = GHSBuilt(surface = GHSBuiltS(; resolution))
    pixel_size = native_resolution(dataset.surface)
    alternating = isodd.((1:12) .+ (1:12)')
    surface = repeat(ifelse.(alternating, 0.2, 0.6); inner = (100 ÷ pixel_size, 100 ÷ pixel_size)) .* pixel_size^2
    height = ifelse.(alternating, 10.0, 30.0)
    grid = LatitudeLongitudeGrid(arch; size = (4, 2),
                                 longitude = (400, 800) .* longitude_per_meter,
                                 latitude = (400, 800) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    with_ghsl_tiles(dataset, surface, height) do
        properties = building_morphometry(grid, dataset)
        fraction = interior(properties.plan_area_index)
        mean_height = interior(properties.mean_building_height)
        # Every mixture of (λᵖ, λᵖh) = (0.2, 2 m) and (0.6, 18 m) satisfies V = 40λᵖ - 6 m.
        @test any(0.3 .< fraction .< 0.5)
        @test fraction .* mean_height ≈ 40fraction .- 6 rtol = 1e-5
    end
end

@testset "GHSL jointly masks missing footprint and height pixels on $arch at $resolution" for arch in test_architectures,
                                                                                           resolution in (GHSBuiltS100m, GHSBuiltS10m)
    dataset = GHSBuilt(surface = GHSBuiltS(; resolution))
    pixel_size = native_resolution(dataset.surface)
    grid = LatitudeLongitudeGrid(arch; size = (4, 2),
                                 longitude = (400, 800) .* longitude_per_meter,
                                 latitude = (400, 800) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    for missing_height in (true, false)
        surface = fill(5000.0, 12, 12)
        height = fill(20.0, 12, 12)
        missing_values = missing_height ? height : surface
        missing_values[1:2:end, :] .= 65535
        surface = repeat(surface; inner = (100 ÷ pixel_size, 100 ÷ pixel_size))
        surface .= ifelse.(surface .== 65535, 65535, 0.5pixel_size^2)
        with_ghsl_tiles(dataset, surface, height) do
            properties = building_morphometry(grid, dataset)
            @test all(0 .< interior(properties.plan_area_index) .< 0.5)
            @test all(isapprox.(interior(properties.mean_building_height), 20; rtol=1e-6))
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

@testset "GHSL uniform footprints survive offset window boundaries on $arch" for arch in test_architectures
    dataset = GHSBuilt(surface = GHSBuiltS(resolution = GHSBuiltS10m))
    offset_grid = LatitudeLongitudeGrid(arch; size = (2, 2),
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

@testset "GHSL urban roughness from native tiles on $arch with $FT" for arch in test_architectures, FT in (Float32, Float64)
    # Four 1200 m cells of 600 m neighborhoods: cell (1, 1) holds a dense core in one neighborhood,
    # cell (1, 2) one neighborhood alternating density and height by 100 m column, cell (2, 1) is
    # no-data, and cell (2, 2) is uniformly built.
    neighborhood(x, y) = (floor(Int, (x - 400) / 600) + 1, floor(Int, (y - 400) / 600) + 1)
    odd_column(x) = isodd(floor(Int, x / 100))
    function built_surface(x, y)
        a, b = neighborhood(x, y)
        (a, b) == (1, 1) && return 50.0
        (a, b) == (1, 3) && return odd_column(x) ? 20.0 : 60.0
        a > 2 && return b > 2 ? 50.0 : 65535.0
        return 0.0
    end
    building_height(x, y) = neighborhood(x, y) == (1, 3) ? (odd_column(x) ? 10.0 : 30.0) : 20.0
    surface = [built_surface(10i - 5, 3200 - 10j + 5) for i in 1:320, j in 1:320]
    height = [building_height(100i - 50, 3200 - 100j + 50) for i in 1:32, j in 1:32]

    dataset = GHSBuilt()
    grid = LatitudeLongitudeGrid(arch, FT; size = (2, 2),
                                 longitude = (400, 2800) .* longitude_per_meter,
                                 latitude = (400, 2800) .* latitude_per_meter,
                                 topology = (Bounded, Bounded, Flat))
    closures = (; variable = MorphometricRoughness(FT),
                  uniform = MorphometricRoughness(FT; height_distribution = UniformHeight()))
    with_ghsl_tiles(dataset, surface, height; north = 3200) do
        results = urban_roughness(grid, dataset; closure = closures, surface_layer_height = 100, neighborhood = 600)
        for name in keys(closures)
            closure = closures[name]
            fields = map(field -> Array(interior(field, :, :, 1)), results[name])
            core = aerodynamic_parameters(closure, FT(0.5), FT(20))
            # Built-area-weighted height, (0.2 · 10 + 0.6 · 30) / 0.8, at plan-area index 0.4.
            mixed = aerodynamic_parameters(closure, FT(0.4), FT(25))
            bare = aerodynamic_parameters(closure, 0, 0)
            @test fields.momentum_roughness_length ≈ [core[1] mixed[1]; bare[1] core[1]] rtol = 1e-5
            @test fields.zero_plane_displacement ≈ [core[2] mixed[2]; bare[2] core[2]] rtol = 1e-5
            @test fields.urban_fraction ≈ [1/4 1/4; 0 1] rtol = 1e-5
            @test fields.building_height ≈ [20 25; 0 20] rtol = 1e-5
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
