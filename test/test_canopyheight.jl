include("runtests_setup.jl")

using NumericalEarth.DataWrangling.CanopyHeight
using NumericalEarth.DataWrangling.CanopyHeight: mask_glad, mask_eth, coarsen_canopy_height,
                                                 roughness_length, displacement_height,
                                                 eth_tile_token, eth_tiles_in_bbox, eth_tile_urls,
                                                 canopy_height_cog_to_netcdf
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, available_variables,
                                    is_three_dimensional, default_inpainting, missing_value

using Oceananigans.Fields: location

#####
##### Pure no-data masking: GLAD categorical fill codes and ETH no-data byte.
#####

@testset "Canopy-height no-data masking" begin
    # Non-forest height 0 is a *valid* value and must be kept, never masked.
    @test mask_glad(0)  == 0.0
    @test mask_glad(37) == 37.0
    @test mask_glad(60) == 60.0            # top of GLAD's valid range

    # GLAD fill codes 101 (water), 102 (snow/ice), 103 (no-data) → NaN.
    @test isnan(mask_glad(101))
    @test isnan(mask_glad(102))
    @test isnan(mask_glad(103))

    # ETH: no-data byte (255) → NaN, valid heights (incl. 0) kept.
    @test mask_eth(0)   == 0.0
    @test mask_eth(23)  == 23.0
    @test isnan(mask_eth(255))
    @test isnan(mask_eth(200, 200))        # configurable no-data

    # Masking is elementwise-broadcastable (as used in the COG read).
    codes = Float32[0, 12, 101, 60, 103]
    masked = mask_glad.(codes)
    @test masked[1] == 0
    @test masked[2] == 12
    @test masked[4] == 60
    @test isnan(masked[3]) && isnan(masked[5])
end

#####
##### Antialiased downsampling: mask-before-average, plausible range preserved.
#####

@testset "Canopy-height coarsening" begin
    # 4×4 of a constant height coarsened 2× → 2×2 of the same height.
    fine = fill(15.0, 4, 4)
    coarse = coarsen_canopy_height(fine, 2)
    @test size(coarse) == (2, 2)
    @test all(coarse .== 15.0)

    # NaN fill cells are dropped from the mean, not propagated.
    block = [10.0 NaN; 20.0 30.0]        # one 2×2 block; mean of finite = 20
    @test coarsen_canopy_height(block, 2)[1, 1] == 20.0

    # A block that is all NaN stays NaN.
    @test isnan(coarsen_canopy_height(fill(NaN, 2, 2), 2)[1, 1])

    # Coarsening a plausible 10 m field keeps the result in a plausible [0, ~40 m] range.
    fine_realistic = clamp.(20 .+ 5 .* randn(40, 40), 0, 40)
    coarse_realistic = coarsen_canopy_height(fine_realistic, 10)
    @test size(coarse_realistic) == (4, 4)
    finite = filter(isfinite, coarse_realistic)
    @test all(0 .<= finite .<= 40)

    @test_throws ArgumentError coarsen_canopy_height(fine, 0)
end

#####
##### Roughness / displacement derivation endpoints (exposed coefficients).
#####

@testset "Roughness from canopy height" begin
    # Bare, endpoint: zero canopy height → zero roughness and displacement.
    @test roughness_length(0.0, 0.10)    == 0.0
    @test displacement_height(0.0, 0.70) == 0.0

    # Standard 20 m canopy.
    @test roughness_length(20.0, 0.10)    == 2.0
    @test displacement_height(20.0, 0.70) == 14.0

    # ClimaLand alternative momentum coefficient (0.13).
    @test roughness_length(10.0, 0.13) ≈ 1.3

    roughness = RoughnessFromCanopyHeight()
    @test roughness.momentum_roughness_coefficient == 0.10
    @test roughness.displacement_coefficient == 0.70

    props = roughness(20.0)
    @test props.momentum_roughness_length == 2.0
    @test props.displacement_height == 14.0

    # Coefficients are exposed and configurable.
    climaland = RoughnessFromCanopyHeight(; momentum_roughness_coefficient = 0.13)
    @test climaland(10.0).momentum_roughness_length ≈ 1.3

    # The closure broadcasts elementwise over a field of heights.
    heights = [0.0, 5.0, 30.0]
    ℓ = [roughness(h).momentum_roughness_length for h in heights]
    @test ℓ == [0.0, 0.5, 3.0]
end

#####
##### ETH tile tokens and windowed-COG URL construction.
#####

@testset "ETH tile tokens" begin
    # 3° lattice: the SW-corner token snaps each coordinate down to a multiple of 3.
    @test eth_tile_token(4.2, 51.7)    == "N51E003"
    @test eth_tile_token(-0.5, 0.5)    == "N00W003"
    @test eth_tile_token(-120.3, -3.2) == "S06W123"

    region = BoundingBox(longitude = (4.0, 5.0), latitude = (51.0, 52.0))
    tokens = eth_tiles_in_bbox(region)
    @test "N51E003" in tokens

    # A bbox straddling a 3° boundary needs more than one tile.
    wide = BoundingBox(longitude = (2.0, 7.0), latitude = (50.0, 53.0))
    @test length(eth_tiles_in_bbox(wide)) >= 2

    urls = eth_tile_urls(region, "Map")
    @test all(startswith.(urls, "/vsicurl/"))
    @test all(endswith.(urls, "_Map.tif"))
end

#####
##### Dataset / metadatum interface.
#####

@testset "Canopy-height dataset interface" begin
    for dataset in (ETHCanopyHeight(), GLADCanopyHeight())
        @test longitude_interfaces(dataset) == (-180, 180)
        @test latitude_interfaces(dataset)  == (-90, 90)
        Nx, Ny, Nz = size(dataset, :canopy_height)
        @test Nz == 1
        @test Nx > Ny > 0

        region = BoundingBox(longitude = (4, 5), latitude = (51, 52))
        meta = Metadatum(:canopy_height; dataset, region)

        @test dataset_variable_name(meta) == "Map"
        @test is_three_dimensional(meta) == false
        @test default_inpainting(meta) === nothing
        @test location(meta) == (Center, Center, Center)

        filename = metadata_filename(dataset, :canopy_height, nothing, region)
        @test endswith(filename, ".nc")
        @test occursin("lon_", filename) && occursin("lat_", filename)
    end

    # ETH exposes both the height map and its uncertainty layer; GLAD only the map.
    eth = ETHCanopyHeight()
    @test Set(keys(available_variables(eth))) == Set((:canopy_height, :canopy_height_uncertainty))
    sd = Metadatum(:canopy_height_uncertainty; dataset = eth,
                   region = BoundingBox(longitude = (4, 5), latitude = (51, 52)))
    @test dataset_variable_name(sd) == "SD"
    @test missing_value(Metadatum(:canopy_height; dataset = eth,
                                  region = BoundingBox(longitude = (4, 5), latitude = (51, 52)))) == 255

    @test Set(keys(available_variables(GLADCanopyHeight()))) == Set((:canopy_height,))

    # Region-keyed filenames disambiguate different windows.
    region_a = BoundingBox(longitude = (4, 5), latitude = (51, 52))
    region_b = BoundingBox(longitude = (0, 2), latitude = (45, 47))
    @test metadata_filename(eth, :canopy_height, nothing, region_a) !=
          metadata_filename(eth, :canopy_height, nothing, region_b)
end

#####
##### Coverage validation requires a bounded region.
#####

@testset "Canopy-height requires a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (10, 10),
                                 longitude = (4, 5),
                                 latitude = (51, 52),
                                 topology = (Bounded, Bounded, Flat))

    # No region → must error.
    meta_global = Metadatum(:canopy_height; dataset = ETHCanopyHeight())
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)

    # Bounded region → passes validation.
    region = BoundingBox(longitude = (4, 5), latitude = (51, 52))
    meta_region = Metadatum(:canopy_height; dataset = ETHCanopyHeight(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end

#####
##### The real COG read is gated behind the ArchGDAL extension: without it, the
##### module entry point errors clearly (mirrors CopernicusDEM.zarr_to_netcdf).
#####

@testset "Canopy-height COG read is extension-gated" begin
    region = BoundingBox(longitude = (4, 5), latitude = (51, 52))
    meta = Metadatum(:canopy_height; dataset = ETHCanopyHeight(), region)
    if isnothing(Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt))
        @test_throws ErrorException canopy_height_cog_to_netcdf(meta, tempname() * ".nc")
    end
end
