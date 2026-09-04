include("runtests_setup.jl")

using NumericalEarth.DataWrangling.GHSL
using NumericalEarth.DataWrangling.GHSL: ghsl_tile_index, ghsl_tiles_in_bbox,
                                         ghsl_tile_url, ghsl_tile_urls, ghsl_tile_tif_name,
                                         longitude_latitude_to_mollweide,
                                         mask_building_height, built_surface_to_fraction,
                                         dataset_prefix, native_resolution, ghsl_tiles_to_netcdf,
                                         ghsl_regional_raster
using NumericalEarth.Lands: MorphometricRoughness, aerodynamic_parameters, urban_roughness
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, native_grid,
                                    longitude_interfaces, latitude_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, metadata_path, available_variables,
                                    is_three_dimensional, default_inpainting

using NCDatasets: NCDataset, defDim, defVar

using Oceananigans.Fields: location
using Oceananigans.Grids: x_domain, y_domain, λnodes, φnodes

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

    # Regression: a window spanning many tiles must not silently drop any — every tile
    # hit by a dense point sampling of the window must be returned.
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

    for dataset in (GHSBuiltH(), GHSBuiltS(), GHSBuiltS(resolution = GHSBuiltS10m))
        @test longitude_interfaces(dataset) == (-180, 180)
        @test latitude_interfaces(dataset)  == (-90, 90)
        Nx, Ny, Nz = size(dataset, :building_height)
        @test Nz == 1
        @test Nx > Ny > 0
    end

    # Native resolution sets the finer 10 m grid ~10× denser than the 100 m grid.
    @test native_resolution(GHSBuiltH()) == 100
    @test native_resolution(GHSBuiltS(resolution = GHSBuiltS10m)) == 10
    @test size(GHSBuiltS(resolution = GHSBuiltS10m), :built_up_fraction)[1] ==
          10 * size(GHSBuiltS(resolution = GHSBuiltS100m), :built_up_fraction)[1]

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
    @test metadata_filename(GHSBuiltS(resolution = GHSBuiltS10m), :built_up_fraction, nothing, region) !=
          metadata_filename(GHSBuiltS(resolution = GHSBuiltS100m), :built_up_fraction, nothing, region)
    @test occursin("2018", dataset_prefix(GHSBuiltS(resolution = GHSBuiltS10m)))

    # The resolution appears in the cache filename as its size in meters, so a cached
    # file stays addressable no matter how the resolution is spelled in the API.
    @test dataset_prefix(GHSBuiltS(resolution = GHSBuiltS10m))  == "GHSBuiltS_10m_2018"
    @test dataset_prefix(GHSBuiltS(resolution = GHSBuiltS100m)) == "GHSBuiltS_100m_2020"
end

@testset "GHSBuiltS constructor" begin
    @test GHSBuiltS().resolution === GHSBuiltS100m
    @test GHSBuiltS().epoch == 2020
    @test GHSBuiltS(resolution = GHSBuiltS10m).epoch == 2018

    # Only a published resolution is representable.
    @test_throws MethodError GHSBuiltS(resolution = 30)

    # Epoch must match the published product matrix.
    @test GHSBuiltS(resolution = GHSBuiltS100m, epoch = 1975) isa GHSBuiltS   # valid endpoint
    @test_throws ArgumentError GHSBuiltS(resolution = GHSBuiltS10m, epoch = 2020)   # 10 m is 2018-only
    @test_throws ArgumentError GHSBuiltS(resolution = GHSBuiltS100m, epoch = 1999)  # not a 5-year step
end

#####
##### Windowed-read URL construction (JRC open-data host).
#####

@testset "GHSL tile URLs" begin
    urlH = ghsl_tile_url(GHSBuiltH(), 3, 19)
    @test startswith(urlH, "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/")
    @test occursin("GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100", urlH)
    @test endswith(urlH, "_R3_C19.zip")

    urlS = ghsl_tile_url(GHSBuiltS(resolution = GHSBuiltS10m), 3, 19)
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
##### The raster the warp materializes is the grid the read path indexes.
#####

@testset "GHSL regional raster geometry" begin
    region = BoundingBox(longitude = (-0.11, -0.07), latitude = (51.505, 51.525))

    for (name, dataset) in ((:building_height, GHSBuiltH()), (:built_up_fraction, GHSBuiltS()))
        metadatum = Metadatum(name; dataset, region)
        grid = native_grid(metadatum)
        raster = ghsl_regional_raster(metadatum)

        @test (raster.Nx, raster.Ny) == size(grid)[1:2]
        @test raster.region.longitude == x_domain(grid)
        @test raster.region.latitude  == y_domain(grid)
        @test raster.longitude == collect(λnodes(grid, Center()))
        @test raster.latitude  == collect(φnodes(grid, Center()))

        # The window covers the request — the native grid brackets it and pads up to a
        # cell per side — so the tiles it needs include every tile the request touches.
        @test raster.region.longitude[1] ≤ region.longitude[1]
        @test raster.region.longitude[2] ≥ region.longitude[2]
        @test raster.region.latitude[1]  ≤ region.latitude[1]
        @test raster.region.latitude[2]  ≥ region.latitude[2]
        @test ghsl_tiles_in_bbox(region) ⊆ ghsl_tiles_in_bbox(raster.region)
    end
end

#####
##### Urban roughness on a model grid, from synthetic rasters in the GHSL cache layout.
#####

function write_ghsl_raster(metadatum, values)
    raster = ghsl_regional_raster(metadatum)
    NCDataset(metadata_path(metadatum), "c") do ds
        defDim(ds, "lon", raster.Nx)
        defDim(ds, "lat", raster.Ny)
        defVar(ds, "lon", Float64, ("lon",))[:] = raster.longitude
        defVar(ds, "lat", Float64, ("lat",))[:] = raster.latitude
        defVar(ds, dataset_variable_name(metadatum), Float64, ("lon", "lat"))[:, :] =
            [values(λ, φ) for λ in raster.longitude, φ in raster.latitude]
    end
    return nothing
end

@testset "GHSL urban roughness on a grid" begin
    # A 2 × 2 grid near the equator whose 2.4 km lattice cells hold 24 × 24 native pixels
    # each, aligned with the pixel faces so the conservative means are exact.
    Nλ, Nφ, _ = size(GHSBuiltH(), :building_height)
    Δλ, Δφ = 360 / Nλ, 180 / Nφ
    refinement, pixels = 4, 24
    λ₁ = -180 + 200_000Δλ
    φ₁ = -90 + 100_000Δφ
    grid = LatitudeLongitudeGrid(CPU(); size = (2, 2),
                                 longitude = (λ₁, λ₁ + 2refinement * pixels * Δλ),
                                 latitude  = (φ₁, φ₁ + 2refinement * pixels * Δφ),
                                 topology = (Bounded, Bounded, Flat))
    lattice_cell(λ, φ) = (floor(Int, (λ - λ₁) / (pixels * Δλ)) + 1,
                          floor(Int, (φ - φ₁) / (pixels * Δφ)) + 1)
    odd_column(λ) = isodd(floor(Int, (λ - λ₁) / Δλ))

    # Lattice cell (2, 2) is a dense core in an otherwise empty grid cell (1, 1); grid cell
    # (2, 2) is uniformly built; lattice cell (2, 6) alternates two densities by pixel
    # column; grid cell (2, 1) is no-data.
    function built_fraction(λ, φ)
        a, b = lattice_cell(λ, φ)
        (a, b) == (2, 2) && return 0.5
        (a, b) == (2, 6) && return odd_column(λ) ? 0.2 : 0.6
        a > 4 && return b > 4 ? 0.5 : NaN
        return 0.0
    end
    function building_height(λ, φ)
        a, b = lattice_cell(λ, φ)
        (a, b) == (2, 6) && return odd_column(λ) ? 10.0 : 30.0
        a > 4 && b <= 4 && return NaN
        return 20.0
    end

    default_cache = GHSL.download_GHSL_cache
    setglobal!(GHSL, :download_GHSL_cache, mktempdir())
    try
        region = BoundingBox(grid)
        write_ghsl_raster(Metadatum(:built_up_fraction; dataset = GHSBuiltS(), region), built_fraction)
        write_ghsl_raster(Metadatum(:building_height; dataset = GHSBuiltH(), region), building_height)

        closure = MorphometricRoughness()
        fields = urban_roughness(grid, GHSBuilt(); closure, neighborhood = 2400)
        @test keys(fields) == (:momentum_roughness_length, :zero_plane_displacement, :urban_fraction, :building_height)

        # The core alone sets its cell's roughness.
        core_roughness, core_displacement = aerodynamic_parameters(closure, 0.5, 20.0)
        @test fields.momentum_roughness_length[1, 1, 1] ≈ core_roughness rtol = 1e-4
        @test fields.zero_plane_displacement[1, 1, 1] ≈ core_displacement rtol = 1e-4
        @test fields.building_height[1, 1, 1] ≈ 20 rtol = 1e-4
        @test fields.urban_fraction[1, 1, 1] == 1 / 16

        @test fields.momentum_roughness_length[2, 2, 1] ≈ core_roughness rtol = 1e-4
        @test fields.zero_plane_displacement[2, 2, 1] ≈ core_displacement rtol = 1e-4
        @test fields.building_height[2, 2, 1] ≈ 20 rtol = 1e-4
        @test fields.urban_fraction[2, 2, 1] == 1

        # Built-area-weighted height, (0.2 · 10 + 0.6 · 30) / 0.8, at plan-area index 0.4.
        mixed_roughness, mixed_displacement = aerodynamic_parameters(closure, 0.4, 25.0)
        @test fields.building_height[1, 2, 1] ≈ 25 rtol = 1e-4
        @test fields.momentum_roughness_length[1, 2, 1] ≈ mixed_roughness rtol = 1e-4
        @test fields.zero_plane_displacement[1, 2, 1] ≈ mixed_displacement rtol = 1e-4
        @test fields.urban_fraction[1, 2, 1] == 1 / 16

        # No-data pixels count as unbuilt.
        bare_roughness, _ = aerodynamic_parameters(closure, 0, 0)
        @test fields.momentum_roughness_length[2, 1, 1] == bare_roughness
        @test fields.zero_plane_displacement[2, 1, 1] == 0
        @test fields.urban_fraction[2, 1, 1] == 0
        @test fields.building_height[2, 1, 1] == 0

        # The 10 m built-up product regrids onto the 100 m height pixels.
        fine = GHSBuiltS(resolution = GHSBuiltS10m)
        write_ghsl_raster(Metadatum(:built_up_fraction; dataset = fine, region), built_fraction)
        fields = urban_roughness(grid, GHSBuilt(surface = fine); closure, neighborhood = 2400)
        @test fields.momentum_roughness_length[2, 2, 1] ≈ core_roughness rtol = 1e-4
        @test fields.building_height[1, 1, 1] ≈ 20 rtol = 1e-4
        @test fields.urban_fraction[1, 1, 1] == 1 / 16
        @test fields.momentum_roughness_length[2, 1, 1] == bare_roughness
    finally
        setglobal!(GHSL, :download_GHSL_cache, default_cache)
    end
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
