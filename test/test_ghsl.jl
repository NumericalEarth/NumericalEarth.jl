include("runtests_setup.jl")

using NumericalEarth.DataWrangling.GHSL
using NumericalEarth.DataWrangling.GHSL: ghsl_tile_index, ghsl_tiles_in_bbox,
                                         ghsl_tile_url, ghsl_tile_urls, ghsl_tile_tif_name,
                                         longitude_latitude_to_mollweide,
                                         mask_building_height, built_surface_to_fraction,
                                         dataset_prefix, native_resolution, ghsl_tiles_to_netcdf
using NumericalEarth.DataWrangling: BoundingBox, Metadatum,
                                    longitude_interfaces, latitude_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, available_variables,
                                    is_three_dimensional, default_inpainting

using Oceananigans.Fields: location

#####
##### World-Mollweide projection + GHSL tile-index arithmetic.
#####

@testset "GHSL Mollweide projection and tile index" begin
    # Origin and equator map to the center of the Mollweide plane.
    x0, y0 = longitude_latitude_to_mollweide(0, 0)
    @test x0 ≈ 0 atol = 1e-6
    @test y0 ≈ 0 atol = 1e-6

    # Poles: y → ±R√2; the x-extent at the equator is ±R·2√2.
    _, yN = longitude_latitude_to_mollweide(0, 90)
    @test yN ≈ 6378137 * sqrt(2) rtol = 1e-3
    xE, _ = longitude_latitude_to_mollweide(180, 0)
    @test xE ≈ 6378137 * 2sqrt(2) rtol = 1e-3

    # Known cities on the 18×36, 1000 km R{row}_C{col} grid (R1_C1 at the NW corner).
    @test ghsl_tile_index(2.35, 48.85)   == (4, 19)   # Paris
    @test ghsl_tile_index(-0.13, 51.51)  == (3, 19)   # London
    @test ghsl_tile_index(139.70, 35.68) == (5, 31)   # Tokyo

    # Row increases southward, column eastward.
    r_north, _ = ghsl_tile_index(0, 80)
    r_south, _ = ghsl_tile_index(0, 10)
    @test r_north < r_south
    _, c_west = ghsl_tile_index(-100, 0)
    _, c_east = ghsl_tile_index(100, 0)
    @test c_west < c_east

    # Indices are clamped to the valid 1:18 / 1:36 range at the extremes.
    @test all(1 .<= ghsl_tile_index(-180, -89) .<= (18, 36))
    @test all(1 .<= ghsl_tile_index(180, 89)   .<= (18, 36))
end

@testset "GHSL tiles intersecting a bbox" begin
    # A small window sits inside one tile.
    region = BoundingBox(longitude = (-0.2, 0.1), latitude = (51.4, 51.6))
    @test ghsl_tiles_in_bbox(region) == [(3, 19)]

    # A window straddling a 1000 km Mollweide boundary needs more than one tile.
    wide = BoundingBox(longitude = (-2.0, 6.0), latitude = (44.0, 52.0))
    @test length(ghsl_tiles_in_bbox(wide)) >= 2
    @test issorted(ghsl_tiles_in_bbox(wide))

    # Regression: a window spanning many tiles must not silently drop any. Every tile hit
    # by a dense point sampling of the window must be returned (a fixed-count sampling
    # skipped whole tile rows once the window spanned more than a handful of tiles).
    tall = BoundingBox(longitude = (0.0, 1.0), latitude = (-60.0, 60.0))
    sampled = Set(ghsl_tile_index(λ, φ)
                  for φ in range(-60, 60; length = 400), λ in range(0, 1; length = 20))
    @test sampled ⊆ Set(ghsl_tiles_in_bbox(tall))
    @test issorted(ghsl_tiles_in_bbox(tall))
end

#####
##### Pure no-data masking + built-surface → plan-area fraction.
#####

@testset "GHSL building-height masking" begin
    # A height of 0 over non-built land is a valid value and must be kept.
    @test mask_building_height(0)    == 0.0
    @test mask_building_height(12.5) == 12.5
    @test mask_building_height(180)  == 180.0

    # The warp writes the no-data gap as NaN; negatives are defensively masked too.
    @test isnan(mask_building_height(NaN))
    @test isnan(mask_building_height(-1))
    @test isnan(mask_building_height(-200))

    # Broadcasts elementwise, as used post-warp.
    raw = Float64[0, 8, NaN, 25, -1]
    masked = mask_building_height.(raw)
    @test masked[1] == 0 && masked[2] == 8 && masked[4] == 25
    @test isnan(masked[3]) && isnan(masked[5])
end

@testset "GHSL built-surface → fraction" begin
    # m² of buildings per native cell ÷ native cell area, clamped to [0, 1].
    @test built_surface_to_fraction(0.0, 10_000.0)     == 0.0    # non-built, valid
    @test built_surface_to_fraction(2_500.0, 10_000.0) == 0.25
    @test built_surface_to_fraction(10_000.0, 10_000.0) == 1.0
    @test built_surface_to_fraction(12_000.0, 10_000.0) == 1.0   # clamped
    @test built_surface_to_fraction(50.0, 100.0)       == 0.5    # 10 m cell area

    # No-data (negative / non-finite) → NaN.
    @test isnan(built_surface_to_fraction(-1.0, 100.0))
    @test isnan(built_surface_to_fraction(NaN, 100.0))
end

#####
##### Dataset / metadatum interface.
#####

@testset "GHSL dataset interface" begin
    region = BoundingBox(longitude = (-0.2, 0.1), latitude = (51.4, 51.6))

    for dataset in (GHSBuiltH(), GHSBuiltS(), GHSBuiltS(resolution = 10))
        @test longitude_interfaces(dataset) == (-180, 180)
        @test latitude_interfaces(dataset)  == (-90, 90)
        Nx, Ny, Nz = size(dataset, :building_height)
        @test Nz == 1
        @test Nx > Ny > 0
    end

    # Native resolution sets the finer 10 m grid ~10× denser than the 100 m grid.
    @test native_resolution(GHSBuiltH()) == 100
    @test native_resolution(GHSBuiltS(resolution = 10)) == 10
    @test size(GHSBuiltS(resolution = 10), :built_up_fraction)[1] ==
          10 * size(GHSBuiltS(resolution = 100), :built_up_fraction)[1]

    mdH = Metadatum(:building_height;   dataset = GHSBuiltH(), region)
    mdS = Metadatum(:built_up_fraction; dataset = GHSBuiltS(), region)

    @test dataset_variable_name(mdH) == "ANBH"
    @test dataset_variable_name(mdS) == "built_up_fraction"
    @test Set(keys(available_variables(GHSBuiltH()))) == Set((:building_height,))
    @test Set(keys(available_variables(GHSBuiltS()))) == Set((:built_up_fraction,))

    for md in (mdH, mdS)
        @test is_three_dimensional(md) == false
        @test default_inpainting(md) === nothing
        @test location(md) == (Center, Center, Center)
    end

    # Region- and product-keyed filenames disambiguate windows / resolutions / epochs.
    region_b = BoundingBox(longitude = (2, 3), latitude = (48, 49))
    @test metadata_filename(GHSBuiltH(), :building_height, nothing, region) !=
          metadata_filename(GHSBuiltH(), :building_height, nothing, region_b)
    @test metadata_filename(GHSBuiltS(resolution = 10), :built_up_fraction, nothing, region) !=
          metadata_filename(GHSBuiltS(resolution = 100), :built_up_fraction, nothing, region)
    @test occursin("2018", dataset_prefix(GHSBuiltS(resolution = 10)))
end

@testset "GHSBuiltS constructor" begin
    @test GHSBuiltS().resolution == 100
    @test GHSBuiltS().epoch == 2020
    @test GHSBuiltS(resolution = 10).epoch == 2018
    @test_throws ArgumentError GHSBuiltS(resolution = 30)

    # Epoch must match the published product matrix.
    @test GHSBuiltS(resolution = 100, epoch = 1975) isa GHSBuiltS   # valid endpoint
    @test_throws ArgumentError GHSBuiltS(resolution = 10, epoch = 2020)   # 10 m is 2018-only
    @test_throws ArgumentError GHSBuiltS(resolution = 100, epoch = 1999)  # not a 5-year step
end

#####
##### Windowed-read URL construction (JRC open-data host).
#####

@testset "GHSL tile URLs" begin
    urlH = ghsl_tile_url(GHSBuiltH(), 3, 19)
    @test startswith(urlH, "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/")
    @test occursin("GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100", urlH)
    @test endswith(urlH, "_R3_C19.zip")

    urlS = ghsl_tile_url(GHSBuiltS(resolution = 10), 3, 19)
    @test occursin("GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10", urlS)
    @test endswith(urlS, "_R3_C19.zip")

    # The GeoTIFF inside a tile archive matches the archive stem with a `.tif` suffix.
    @test ghsl_tile_tif_name(GHSBuiltH(), 3, 19) ==
          "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0_R3_C19.tif"

    region = BoundingBox(longitude = (-0.2, 0.1), latitude = (51.4, 51.6))
    urls = ghsl_tile_urls(GHSBuiltH(), region)
    @test all(startswith.(urls, "https://"))
    @test all(endswith.(urls, ".zip"))
end

#####
##### Coverage validation requires a bounded region.
#####

@testset "GHSL requires a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (8, 8),
                                 longitude = (-0.2, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))

    meta_global = Metadatum(:building_height; dataset = GHSBuiltH())
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)

    region = BoundingBox(longitude = (-0.2, 0.1), latitude = (51.4, 51.6))
    meta_region = Metadatum(:building_height; dataset = GHSBuiltH(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end

#####
##### The real Mollweide warp is gated behind the ArchGDAL extension.
#####

@testset "GHSL read is extension-gated" begin
    region = BoundingBox(longitude = (-0.2, 0.1), latitude = (51.4, 51.6))
    meta = Metadatum(:building_height; dataset = GHSBuiltH(), region)
    if isnothing(Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt))
        @test_throws ErrorException ghsl_tiles_to_netcdf(meta, tempname() * ".nc")
    end
end
