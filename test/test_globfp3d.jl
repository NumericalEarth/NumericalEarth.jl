include("runtests_setup.jl")

using NumericalEarth.DataWrangling.GloBFP3D
using NumericalEarth.DataWrangling.GloBFP3D: reduce_morphometry, native_region_grid,
                                             parse_tile_bounds, tile_intersects,
                                             native_cell_size, native_resolution,
                                             globfp3d_rasterize_to_netcdf
using NumericalEarth.DataWrangling: BoundingBox, Metadatum,
                                    longitude_interfaces, latitude_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, available_variables,
                                    is_three_dimensional, default_inpainting

using Oceananigans.Fields: location

#####
##### Per-cell morphometry reduced from a synthetic fine building-height raster.
#####

@testset "GloBFP3D morphometry reduction" begin
    # 6×6 fine raster of 10 m cells at the equator; a 30 m building filling the 3×3 block in
    # the middle → a 30 m × 30 m × 30 m block. One target cell covering the whole raster.
    Δ = rad2deg(10 / Oceananigans.defaults.planet_radius)   # 10 m N–S at the default planet radius
    lon = [(i - 1/2) * Δ for i in 1:6]
    lat = [(j - 1/2) * Δ for j in 1:6]
    height = zeros(6, 6); height[2:4, 2:4] .= 30.0
    target = LatitudeLongitudeGrid(CPU(), Float64; size = (1, 1),
                                   longitude = (0, 6Δ), latitude = (0, 6Δ),
                                   topology = (Bounded, Bounded, Flat))

    m = reduce_morphometry(height, lon, lat, target)
    @test m.built_up_fraction[1, 1]       ≈ 9 / 36            # 9 built of 36 cells
    @test m.mean_building_height[1, 1]    ≈ 30                # mean over built cells
    @test m.building_height_std[1, 1]     ≈ 0                 # uniform height
    @test m.maximum_building_height[1, 1] ≈ 30
    @test m.gross_building_height[1, 1]   ≈ 30 * 9 / 36       # mean over all cells = λp·H
    # λf: an isolated 30 m × 30 m × 30 m block has frontal area 900 m² / 3600 m² plan = 0.25.
    @test m.frontal_area_index[1, 1]      ≈ 0.25 rtol = 1e-6

    # Height heterogeneity: eight 10 m cells + one 40 m cell → nonzero σH, Hmax = 40.
    h2 = zeros(6, 6); h2[2:4, 2:4] .= 10.0; h2[3, 3] = 40.0
    m2 = reduce_morphometry(h2, lon, lat, target)
    mean_h = (8 * 10 + 40) / 9
    @test m2.building_height_std[1, 1]     ≈ sqrt((8 * 100 + 1600) / 9 - mean_h^2)
    @test m2.maximum_building_height[1, 1] ≈ 40
    @test m2.mean_building_height[1, 1]    ≈ mean_h

    # An empty raster reduces to zero everywhere (a park/river adds no building height).
    m0 = reduce_morphometry(zeros(6, 6), lon, lat, target)
    for field in m0
        @test field[1, 1] == 0
    end

    # Four target cells over the 6×6 raster: only the block's cells carry morphometry.
    quad = LatitudeLongitudeGrid(CPU(), Float64; size = (2, 2),
                                 longitude = (0, 6Δ), latitude = (0, 6Δ),
                                 topology = (Bounded, Bounded, Flat))
    mq = reduce_morphometry(height, lon, lat, quad)
    @test sum(mq.built_up_fraction) > 0
    @test all(0 .<= mq.built_up_fraction .<= 1)

    # A latitude/longitude-stretched target grid bins each fine cell by the target cell faces, so
    # it reduces correctly (one building per quadrant lands in the right cell with distinct values).
    stretched = LatitudeLongitudeGrid(CPU(), Float64; size = (2, 2),
                                      longitude = [0.0, 2Δ, 6Δ],   # unequal columns
                                      latitude  = [0.0, 4Δ, 6Δ],   # unequal rows
                                      topology = (Bounded, Bounded, Flat))
    hq = zeros(6, 6)
    hq[1, 1] = 10; hq[3, 1] = 20; hq[1, 5] = 30; hq[3, 5] = 40
    ms = reduce_morphometry(hq, lon, lat, stretched)
    @test ms.mean_building_height    ≈ [10.0 30.0; 20.0 40.0]
    @test ms.maximum_building_height ≈ [10.0 30.0; 20.0 40.0]
    @test ms.built_up_fraction       ≈ [1/8 1/4; 1/16 1/8]

    # A degenerate 1×1 fine raster has no fine step to difference: return a result, not a crash.
    m1 = reduce_morphometry(fill(20.0, 1, 1), [3Δ], [3Δ], target)
    @test m1.mean_building_height[1, 1]  ≈ 20
    @test m1.built_up_fraction[1, 1]     ≈ 1
    @test m1.frontal_area_index[1, 1]    == 0
end

@testset "GloBFP3D native aggregation grid" begin
    dataset = BuildingFootprints3D()
    region = BoundingBox(longitude = (-74.02, -73.93), latitude = (40.70, 40.82))
    Δ = native_cell_size(dataset)
    g = native_region_grid(region, Δ, Δ)
    @test g.west ≤ -74.02 && g.west + g.Nx * g.Δλ ≥ -73.93
    @test g.south ≤ 40.70 && g.south + g.Ny * g.Δφ ≥ 40.82
    @test g.Δλ ≈ Δ && g.Δφ ≈ Δ
    # Uniform in degrees, so the raster is a sub-window of the global lattice the shared
    # `Field(::Metadatum)` read path assumes (a latitude-dependent Δλ would misalign it).
    @test g.Δλ == g.Δφ
end

#####
##### Tile filename parsing + region selection.
#####

@testset "GloBFP3D tile selection" begin
    b = parse_tile_bounds("845_0.0_51.25_1.25_52.5_UK-ENG.zip")
    @test b.gid == 845
    @test (b.west, b.south, b.east, b.north) == (0.0, 51.25, 1.25, 52.5)
    b2 = parse_tile_bounds("985_-2.5_42.5_-1.25_43.75_FR_SP.shp")
    @test (b2.west, b2.south, b2.east, b2.north) == (-2.5, 42.5, -1.25, 43.75)
    @test isnothing(parse_tile_bounds("world_grid.zip"))

    nyc = BoundingBox(longitude = (-74.02, -73.93), latitude = (40.70, 40.82))
    @test tile_intersects((; west = -75.0, south = 40.0, east = -73.75, north = 41.25), nyc)
    @test !tile_intersects((; west = 0.0, south = 51.25, east = 1.25, north = 52.5), nyc)
end

#####
##### Dataset / metadatum interface.
#####

@testset "GloBFP3D dataset interface" begin
    region = BoundingBox(longitude = (-74.02, -73.93), latitude = (40.70, 40.82))

    @test BuildingFootprints3D().resolution == 3
    @test native_resolution(BuildingFootprints3D(resolution = 10)) == 10
    @test_throws ArgumentError BuildingFootprints3D(resolution = 0)

    dataset = BuildingFootprints3D()
    @test longitude_interfaces(dataset) == (-180, 180)
    @test latitude_interfaces(dataset)  == (-90, 90)
    Nx, Ny, Nz = size(dataset, :building_height)
    @test Nz == 1 && Nx > Ny > 0
    # A finer resolution gives a proportionally denser native grid.
    @test size(BuildingFootprints3D(resolution = 3), :building_height)[1] >
          size(BuildingFootprints3D(resolution = 30), :building_height)[1]

    @test Set(keys(available_variables(dataset))) == Set((:building_height,))
    md = Metadatum(:building_height; dataset, region)
    @test dataset_variable_name(md) == "building_height"
    @test is_three_dimensional(md) == false
    @test default_inpainting(md) === nothing
    @test location(md) == (Center, Center, Center)

    # Filenames disambiguate resolution and region.
    region_b = BoundingBox(longitude = (2, 3), latitude = (48, 49))
    @test metadata_filename(dataset, :building_height, nothing, region) !=
          metadata_filename(dataset, :building_height, nothing, region_b)
    @test metadata_filename(BuildingFootprints3D(resolution = 3), :building_height, nothing, region) !=
          metadata_filename(BuildingFootprints3D(resolution = 30), :building_height, nothing, region)
end

@testset "GloBFP3D requires a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (8, 8),
                                 longitude = (-0.2, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))

    global_md = Metadatum(:building_height; dataset = BuildingFootprints3D())
    @test_throws ErrorException validate_dataset_coverage(grid, global_md)

    region = BoundingBox(longitude = (-0.2, 0.1), latitude = (51.4, 51.6))
    region_md = Metadatum(:building_height; dataset = BuildingFootprints3D(), region)
    @test validate_dataset_coverage(grid, region_md) === nothing

    # An antimeridian-crossing region (west > east) is rejected rather than expanded to near-global.
    crossing = BoundingBox(longitude = (179.9, -179.9), latitude = (-17.0, -16.0))
    crossing_md = Metadatum(:building_height; dataset = BuildingFootprints3D(), region = crossing)
    @test_throws ErrorException validate_dataset_coverage(grid, crossing_md)

    # Swapped latitude bounds are rejected too.
    flipped = BoundingBox(longitude = (-0.2, 0.1), latitude = (51.6, 51.4))
    flipped_md = Metadatum(:building_height; dataset = BuildingFootprints3D(), region = flipped)
    @test_throws ErrorException validate_dataset_coverage(grid, flipped_md)
end

#####
##### The real rasterization is gated behind the ArchGDAL extension.
#####

@testset "GloBFP3D read is extension-gated" begin
    region = BoundingBox(longitude = (-74.02, -73.93), latitude = (40.70, 40.82))
    md = Metadatum(:building_height; dataset = BuildingFootprints3D(), region)
    if isnothing(Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt))
        @test_throws ErrorException globfp3d_rasterize_to_netcdf(md, tempname() * ".nc")
    end
end
