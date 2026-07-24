include("runtests_setup.jl")

using NumericalEarth.DataWrangling.ETHSentinel2Canopy
using NumericalEarth.DataWrangling.ETHSentinel2Canopy: mask_eth,
                                              eth_tile_token, eth_tiles_in_bbox, eth_tile_urls,
                                              canopy_height_cog_to_netcdf, canopy_height_field
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, available_variables,
                                    is_three_dimensional, default_inpainting

using Oceananigans.Fields: location, interior
using ArchGDAL   # loads NumericalEarthArchGDALExt so the real COG read path is exercised

#####
##### Pure no-data masking: the ETH no-data byte.
#####

@testset "Canopy-height no-data masking" begin
    # ETH: no-data byte (255) → NaN, valid heights (incl. the non-forest value 0) kept.
    @test mask_eth(0)   == 0.0
    @test mask_eth(23)  == 23.0
    @test isnan(mask_eth(255))
    @test isnan(mask_eth(200, 200))        # configurable no-data

    # Masking is elementwise-broadcastable (as used in the COG read).
    codes = Float32[0, 12, 255, 60, 255]
    masked = mask_eth.(codes)
    @test masked[1] == 0
    @test masked[2] == 12
    @test masked[4] == 60
    @test isnan(masked[3]) && isnan(masked[5])
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

    # An upper bound exactly on a 3° line only touches the next tile at its edge, so that
    # tile is excluded: (3, 6) × (51, 54) is one tile, not four.
    @test eth_tiles_in_bbox(BoundingBox(longitude = (3.0, 6.0), latitude = (51.0, 54.0))) == ["N51E003"]

    urls = eth_tile_urls(region, :canopy_height)
    @test all(startswith.(urls, "/vsicurl/"))
    @test all(occursin.("libdrive.ethz.ch/public.php/webdav", urls))
    @test all(endswith.(urls, "_Map.tif"))

    # The uncertainty layer is the `_Map_SD` sibling of each tile.
    sd_urls = eth_tile_urls(region, :canopy_height_uncertainty)
    @test all(endswith.(sd_urls, "_Map_SD.tif"))
end

#####
##### Dataset / metadatum interface.
#####

@testset "Canopy-height dataset interface" begin
    dataset = ETHSentinel2CanopyHeight()
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

    # ETH exposes both the height map and its uncertainty layer.
    @test Set(keys(available_variables(dataset))) == Set((:canopy_height, :canopy_height_uncertainty))
    sd = Metadatum(:canopy_height_uncertainty; dataset,
                   region = BoundingBox(longitude = (4, 5), latitude = (51, 52)))
    @test dataset_variable_name(sd) == "SD"

    # Region-keyed filenames disambiguate different windows.
    region_a = BoundingBox(longitude = (4, 5), latitude = (51, 52))
    region_b = BoundingBox(longitude = (0, 2), latitude = (45, 47))
    @test metadata_filename(dataset, :canopy_height, nothing, region_a) !=
          metadata_filename(dataset, :canopy_height, nothing, region_b)
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
    meta_global = Metadatum(:canopy_height; dataset = ETHSentinel2CanopyHeight())
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)

    # Bounded region → passes validation.
    region = BoundingBox(longitude = (4, 5), latitude = (51, 52))
    meta_region = Metadatum(:canopy_height; dataset = ETHSentinel2CanopyHeight(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end

#####
##### The real COG read is gated behind the ArchGDAL extension: without it, the
##### module entry point errors clearly (mirrors CopernicusDEM.zarr_to_netcdf).
#####

@testset "Canopy-height COG read is extension-gated" begin
    region = BoundingBox(longitude = (4, 5), latitude = (51, 52))
    meta = Metadatum(:canopy_height; dataset = ETHSentinel2CanopyHeight(), region)
    grid = LatitudeLongitudeGrid(CPU(); size = (4, 4), longitude = (4, 5), latitude = (51, 52),
                                 topology = (Bounded, Bounded, Flat))
    if isnothing(Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt))
        @test_throws ErrorException canopy_height_cog_to_netcdf(meta, tempname() * ".nc")
        @test_throws ErrorException canopy_height_field(grid, ETHSentinel2CanopyHeight())
    end
end

#####
##### The real windowed-COG read path, driven by a synthetic local GeoTIFF (no network):
##### mosaic/window via gdalwarp, the north→south orientation flip, geotransform-derived
##### coordinates, and no-data masking.
#####

@testset "Windowed COG read on a synthetic GeoTIFF" begin
    ext = Base.get_extension(NumericalEarth, :NumericalEarthArchGDALExt)
    @test !isnothing(ext)

    # 6×4 EPSG:4326 Byte raster over lon (0, 6), lat (50, 54) at 1° pixels, north-up.
    # The pixel value encodes its own center latitude; the no-data byte (255) sits at the
    # north-west corner so we can check both orientation and masking.
    nx, ny = 6, 4
    band = UInt8[round(UInt8, 54.0 - (row - 0.5)) for col in 1:nx, row in 1:ny]
    band[1, 1] = 255

    tif = tempname() * ".tif"
    ArchGDAL.create(tif; driver = ArchGDAL.getdriver("GTiff"),
                    width = nx, height = ny, nbands = 1, dtype = UInt8) do ds
        ArchGDAL.setgeotransform!(ds, [0.0, 1.0, 0.0, 54.0, 0.0, -1.0])
        ArchGDAL.setproj!(ds, ArchGDAL.toWKT(ArchGDAL.importEPSG(4326)))
        ArchGDAL.write!(ArchGDAL.getband(ds, 1), band)
    end

    warped = ext.warp_canopy_onto_grid([tif], (0.0, 6.0), (50.0, 54.0), nx, ny;
                                       resampling = "near", nodata = 255)

    # Cell centers come straight off the warped geotransform.
    @test warped.longitude ≈ [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    @test warped.latitude  ≈ [50.5, 51.5, 52.5, 53.5]

    # Rows run south→north after the flip: j = 1 is the southernmost row.
    @test warped.data[1, 1]   ≈ 50            # south row center latitude 50.5 → byte 50
    @test warped.data[end, end] ≈ 54          # north row
    @test warped.data[1, end] == 255f0        # north-west no-data byte preserved (near resampling)

    masked = mask_eth.(warped.data, 255)
    @test isnan(masked[1, end])               # no-data → NaN
    @test all(isfinite, masked[:, 1])         # valid south row kept, zeros are not masked

    # canopy_field drops the masked array onto a matching grid, oriented south→north.
    grid = LatitudeLongitudeGrid(CPU(); size = (nx, ny),
                                 longitude = (0, 6), latitude = (50, 54),
                                 topology = (Bounded, Bounded, Flat))
    h = ext.canopy_field(grid, masked)
    H = Array(interior(h))
    @test size(H) == (nx, ny, 1)
    @test H[1, 1, 1] ≈ 50                      # south
    @test isnan(H[1, ny, 1])                   # north-west gap
end
