include("runtests_setup.jl")

using ArchGDAL
using ZipFile: ZipFile
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
    for resolution in (GHSBuiltS100m, GHSBuiltS10m), (longitude, latitude) in ((-74, 40), (140, 36), (-120, -60))
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
        longitude_faces = range(longitude - 0.015, longitude + 0.015; length = 3)
        latitude_faces = range(latitude - 0.015, latitude + 0.015; length = 3)
        grid = LatitudeLongitudeGrid(CPU(); size = (2, 2),
                                     longitude = longitude_faces, latitude = latitude_faces,
                                     topology = (Bounded, Bounded, Flat))
        areas = MOLLWEIDE_RADIUS^2 .* deg2rad.(diff(longitude_faces)) .* diff(sind.(latitude_faces))'
        with_ghsl_tiles(dataset, surface, height; longitude, latitude, west, north) do
            for maximum_raster_cells in (100_000_000, 10_000)
                properties = building_morphometry(grid, dataset; maximum_raster_cells)
                fraction = interior(properties.plan_area_index, :, :, 1)
                mean_height = interior(properties.mean_building_height, :, :, 1)
                @test sum(areas .* fraction) ≈ 0.8step^2 rtol = 1e-6
                @test sum(areas .* fraction .* mean_height) ≈ 20step^2 rtol = 1e-6
            end
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
grid = LatitudeLongitudeGrid(CPU(); size = (4, 2),
                             longitude = (400, 800) .* longitude_per_meter,
                             latitude = (400, 800) .* latitude_per_meter,
                             topology = (Bounded, Bounded, Flat))

@testset "GHSL preserves density-height covariance through reprojection" begin
    surface = repeat(ifelse.(isodd.(1:12), 2000.0, 6000.0), 1, 12)
    height = repeat(ifelse.(isodd.(1:12), 10.0, 30.0), 1, 12)
    with_ghsl_tiles(GHSBuilt(), surface, height) do
        properties = building_morphometry(grid, GHSBuilt())
        fraction = interior(properties.plan_area_index)
        mean_height = interior(properties.mean_building_height)
        # Every mixture of (λᵖ, λᵖh) = (0.2, 2 m) and (0.6, 18 m) satisfies V = 40λᵖ - 6 m.
        @test any(0.3 .< fraction .< 0.5)
        @test fraction .* mean_height ≈ 40fraction .- 6 rtol = 1e-5
    end
end

@testset "GHSL jointly masks missing footprint and height pixels" begin
    for missing_height in (true, false)
        surface = fill(5000.0, 12, 12)
        height = fill(20.0, 12, 12)
        missing_values = missing_height ? height : surface
        missing_values[1:2:end, :] .= 65535
        with_ghsl_tiles(GHSBuilt(), surface, height) do
            properties = building_morphometry(grid, GHSBuilt())
            @test all(0 .< interior(properties.plan_area_index) .< 0.5)
            @test all(isapprox.(interior(properties.mean_building_height), 20; rtol=1e-6))
        end
    end
end

@testset "GHSL retains 10 m footprints" begin
    dataset = GHSBuilt(surface = GHSBuiltS(resolution = GHSBuiltS10m))
    surface = repeat(ifelse.(1:120 .<= 60, 0.0, 100.0), 1, 120)
    fine_grid = LatitudeLongitudeGrid(CPU(); size = (4, 1),
                                      longitude = (580, 620) .* longitude_per_meter,
                                      latitude = (500, 600) .* latitude_per_meter,
                                      topology = (Bounded, Bounded, Flat))
    with_ghsl_tiles(dataset, surface, fill(20.0, 12, 12)) do
        properties = building_morphometry(fine_grid, dataset)
        fraction = vec(interior(properties.plan_area_index))
        @test first(fraction) ≈ 0 atol = 1e-6
        @test last(fraction) ≈ 1 atol = 1e-6
        built_height = interior(properties.mean_building_height)[interior(properties.plan_area_index) .> 0]
        @test all(isapprox.(built_height, 20; rtol=1e-6))
    end
end

@testset "GHSL uniform footprints survive offset window boundaries" begin
    dataset = GHSBuilt(surface = GHSBuiltS(resolution = GHSBuiltS10m))
    offset_grid = LatitudeLongitudeGrid(CPU(); size = (2, 2),
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
