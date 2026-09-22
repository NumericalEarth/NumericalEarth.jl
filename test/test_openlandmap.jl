include("runtests_setup.jl")

using NumericalEarth.DataWrangling.OpenLandMap
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces, z_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, conversion_units, convert_units,
                                    default_inpainting, is_three_dimensional,
                                    WeightPercent, GramPerCubicCentimeter
using NumericalEarth.DataWrangling.OpenLandMap: assemble_cog_window, cog_window_indices,
                                                cog_window_to_netcdf, validate_epsg4326,
                                                validate_geographic_northup

using ArchGDAL
using NCDatasets: NCDataset

# The real /vsicurl reads require network access, so they are exercised manually /
# in the docs build. Here we test the dataset-interface, unit-conversion, and
# coverage logic, plus the windowed-COG reader on a synthetic on-disk tile.

@testset "OpenLandMapSoilDB metadata interfaces" begin
    ds = OpenLandMapSoilDB()

    @test longitude_interfaces(ds) == (-180.0005, 180.0005)
    @test latitude_interfaces(ds) == (-56.0005, 76.0005)
    @test z_interfaces(ds) == [-1.0, -0.6, -0.3, 0.0]

    Nx, Ny, Nz = size(ds, :clay_fraction)
    @test (Nx, Ny, Nz) == (1440004, 528004, 3)

    region = BoundingBox(longitude = (-112.3, -111.9), latitude = (36.0, 36.4))
    for (name, short) in (:sand_fraction => "sand", :silt_fraction => "silt",
                          :clay_fraction => "clay", :bulk_density => "bd")
        meta = Metadatum(name; dataset = ds, region)
        @test dataset_variable_name(meta) == short
        @test is_three_dimensional(meta)
        @test default_inpainting(meta) === nothing
    end

    fname = metadata_filename(ds, :clay_fraction, nothing, region)
    @test fname == "OpenLandMap_clay_fraction_lon_-112.3_-111.9_lat_36.0_36.4.nc"

    # Region-keyed filenames are distinct.
    region_b = BoundingBox(longitude = (0, 2), latitude = (50, 52))
    @test metadata_filename(ds, :clay_fraction, nothing, region) !=
          metadata_filename(ds, :clay_fraction, nothing, region_b)
end

@testset "OpenLandMapSoilDB unit conversions" begin
    # Texture: percent → kg/kg.
    @test conversion_units(Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB())) isa WeightPercent
    @test convert_units(19.0f0, WeightPercent()) ≈ 0.19f0

    # Bulk density: g/cm³ → kg/m³.
    @test conversion_units(Metadatum(:bulk_density; dataset = OpenLandMapSoilDB())) isa GramPerCubicCentimeter
    @test convert_units(1.34f0, GramPerCubicCentimeter()) ≈ 1340.0f0
end

@testset "OpenLandMapSoilDB requires a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (10, 10, 3),
                                 longitude = (-112.3, -111.9), latitude = (36.0, 36.4),
                                 z = [-1.0, -0.6, -0.3, 0.0])

    meta_global = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB())
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)
    @test_throws ErrorException download(meta_global)

    region = BoundingBox(longitude = (-112.3, -111.9), latitude = (36.0, 36.4))
    meta_region = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end

@testset "OpenLandMapSoilDB COG windowing and decoding" begin
    x0, y0, dx, dy = -5.0, 4.0, 0.1, -0.1
    geotransform = [x0, dx, 0.0, y0, 0.0, dy]
    width, height = 10, 8

    # An interior request: the window must strictly contain it on all four sides.
    bbox = BoundingBox(longitude = (-4.5, -4.2), latitude = (3.5, 3.8))
    xoff, yoff, xsize, ysize = cog_window_indices(geotransform, width, height, bbox)

    @test x0 + xoff * dx < bbox.longitude[1]
    @test x0 + (xoff + xsize) * dx > bbox.longitude[2]
    @test y0 + (yoff + ysize) * dy < bbox.latitude[1]
    @test y0 + yoff * dy > bbox.latitude[2]

    # A request overhanging every edge clamps to the raster instead of running off it.
    huge = BoundingBox(longitude = (x0 - 1, x0 + width * dx + 1),
                       latitude  = (y0 + height * dy - 1, y0 + 1))
    @test cog_window_indices(geotransform, width, height, huge) == (0, 0, width, height)

    scale, offset, nodata = 0.5, 2.0, 255
    raw = UInt8[i + 10 * (j - 1) for i in 1:4, j in 1:3]  # (lon, lat), north-first
    raw[2, 1] = nodata
    longitude, latitude, data = assemble_cog_window(raw, geotransform, 2, 1, scale, offset, nodata)

    # Cell centers, half a pixel in from the window's west and north faces.
    @test longitude[1] ≈ x0 + 2 * dx + dx / 2
    @test latitude[end] ≈ y0 + 1 * dy + dy / 2
    @test issorted(latitude)

    # Latitude ascends, so raw row 1 (north) becomes the last latitude index.
    @test eltype(data) == Float32
    @test data[1, end] ≈ raw[1, 1] * scale + offset
    @test data[1, 1] ≈ raw[1, 3] * scale + offset

    # The fill is masked before scale/offset; scaling it would give a finite 129.5.
    @test isnan(data[2, end])
    @test count(isnan, data) == 1

    @test validate_geographic_northup(geotransform) === nothing
    @test_throws ErrorException validate_geographic_northup([x0, dx, 0.01, y0, 0.0, dy])
    @test_throws ErrorException validate_geographic_northup([x0, dx, 0.0, y0, 0.0, -dy])

    @test validate_epsg4326(nothing) === nothing
    @test validate_epsg4326(4326) === nothing
    @test_throws ErrorException validate_epsg4326(3857)
end

# Build a small GeoTIFF with a known CRS/scale/offset/nodata; row 0 is north.
function write_synthetic_tile(path; nx, ny, x0, y0, dx, dy, scale, offset, nodata, raw,
                              epsg = 4326, dtype = UInt8)
    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"),
                    width = nx, height = ny, nbands = 1, dtype) do ds
        ArchGDAL.setgeotransform!(ds, [x0, dx, 0.0, y0, 0.0, dy])
        ArchGDAL.setproj!(ds, ArchGDAL.toWKT(ArchGDAL.importEPSG(epsg)))
        band = ArchGDAL.getband(ds, 1)
        ArchGDAL.setnodatavalue!(band, Float64(nodata))
        ArchGDAL.GDAL.gdalsetrasterscale(band.ptr, Float64(scale))
        ArchGDAL.GDAL.gdalsetrasteroffset(band.ptr, Float64(offset))
        ArchGDAL.write!(band, raw)
    end
    return path
end

@testset "OpenLandMapSoilDB windowed COG reader (synthetic tile)" begin
    dir = mktempdir()
    nx, ny = 10, 8
    x0, y0, dx, dy = -5.0, 4.0, 0.1, -0.1
    scale, offset, nodata = 0.5, 2.0, 255

    raw = UInt8[i + 10 * (j - 1) for i in 1:nx, j in 1:ny]  # (lon, lat), north-first
    raw[3, 2] = nodata

    tif = write_synthetic_tile(joinpath(dir, "tile.tif");
                               nx, ny, x0, y0, dx, dy, scale, offset, nodata, raw)

    nc = joinpath(dir, "out.nc")
    bbox = BoundingBox(longitude = (x0, x0 + nx * dx), latitude = (y0 + ny * dy, y0))
    cog_window_to_netcdf([tif], nc, "clay", bbox)

    NCDataset(nc) do ds
        lon = ds["lon"][:]
        lat = ds["lat"][:]
        data = ds["clay"][:, :, 1]

        @test size(data) == (nx, ny)

        # Coordinates are cell centers, ascending in both axes.
        @test lon[1] ≈ x0 + dx / 2
        @test issorted(lon) && issorted(lat)

        # Orientation: latitude ascending ⇒ north (raw row 1) is the last lat index.
        # decode = raw * scale + offset.
        @test data[1, end] ≈ raw[1, 1] * scale + offset          # north-west corner
        @test data[1, 1] ≈ raw[1, ny] * scale + offset            # south-west corner

        # Nodata is masked to NaN *before* scale/offset (a scaled fill would be finite).
        @test isnan(data[3, ny - 2 + 1])
    end

    # Stacking: three depth sources → a (lon, lat, 3) array.
    nc3 = joinpath(dir, "out3.nc")
    cog_window_to_netcdf([tif, tif, tif], nc3, "clay", bbox)
    NCDataset(nc3) do ds
        data = ds["clay"][:, :, :]
        @test size(data, 3) == 3
        @test isequal(data[:, :, 1], data[:, :, 3])   # isequal: NaN == NaN elementwise
    end
end

@testset "OpenLandMapSoilDB windowed COG reader rejects a non-EPSG:4326 source" begin
    dir = mktempdir()
    nx, ny = 10, 8
    x0, y0, dx, dy = -5.0, 4.0, 0.1, -0.1
    raw = UInt8[i for i in 1:nx, j in 1:ny]

    # A source declaring a projected CRS must fail loudly: windowing is done in degrees.
    tif = write_synthetic_tile(joinpath(dir, "mercator.tif");
                               nx, ny, x0, y0, dx, dy, scale = 1.0, offset = 0.0,
                               nodata = 255, raw, epsg = 3857)
    bbox = BoundingBox(longitude = (x0, x0 + nx * dx), latitude = (y0 + ny * dy, y0))
    @test_throws ErrorException cog_window_to_netcdf([tif], joinpath(dir, "bad.nc"), "clay", bbox)
end

@testset "OpenLandMapSoilDB tiled regrid reproduces the whole-window regrid" begin
    dir = mktempdir()

    # A tile on the dataset's global lattice, carrying structure at every scale so that a
    # misregistered tile boundary could not hide in a smooth field.
    nx, ny = 512, 512
    x0, y0, dx, dy = -112.0005, 36.0005, 0.00025, -0.00025
    raw = Float32[30 + 10 * sinpi(i / 64) * cospi(j / 48) + (i % 7) for i in 1:nx, j in 1:ny]
    raw[100:140, 60:90] .= -1.0   # a masked patch straddling tile interiors

    tif = write_synthetic_tile(joinpath(dir, "tiled.tif");
                               nx, ny, x0, y0, dx, dy, scale = 1.0, offset = 0.0,
                               nodata = -1.0, raw, dtype = Float32)

    # The window spans the masked patch, so tile interiors have to straddle it.
    grid = LatitudeLongitudeGrid(CPU(); size = (10, 10, 3),
                                 longitude = (-111.98, -111.96), latitude = (35.97, 35.99),
                                 z = [-1.0, -0.6, -0.3, 0.0])
    region = BoundingBox(grid)

    metadatum = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region, dir)
    cog_window_to_netcdf(fill(tif, 3), metadata_path(metadatum), "clay", region)

    # The path the regrid took before tiling: materialize the whole window, then interpolate.
    native = Field(metadatum, CPU())
    untiled = Field{Center, Center, Center}(grid)
    NumericalEarth.DataWrangling.interpolate_physical!(untiled, native, metadatum)

    # Tiling changes where the data comes from, never the arithmetic done on it: each tile is a
    # windowed field over the native grid interpolated into a window of the target, so it runs
    # the same regrid over the same node coordinates. Every budget reproduces it bitwise.
    reference = Array(interior(untiled))

    for tile_bytes in (typemax(Int), 20_000, 5_000)
        tiled = Array(interior(Field(metadatum, grid; tile_bytes)))
        @test size(tiled) == size(reference)
        @test isequal(tiled, reference)
    end
end

@testset "windowed retrieval matches the whole-file read" begin
    dir = mktempdir()
    nx, ny = 256, 256
    x0, y0, dx, dy = -112.0005, 36.0005, 0.00025, -0.00025
    raw = Float32[(i % 11) + 3 * (j % 5) for i in 1:nx, j in 1:ny]

    tif = write_synthetic_tile(joinpath(dir, "window.tif");
                               nx, ny, x0, y0, dx, dy, scale = 1.0, offset = 0.0,
                               nodata = -1.0, raw, dtype = Float32)

    region = BoundingBox(longitude = (x0 + 0.01, x0 + 0.05), latitude = (y0 - 0.05, y0 - 0.01))
    metadatum = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region, dir)
    cog_window_to_netcdf(fill(tif, 3), metadata_path(metadatum), "clay", region)

    whole = NumericalEarth.DataWrangling.retrieve_data(metadatum)
    λ, φ = NumericalEarth.DataWrangling.read_file_coords(metadatum)

    for (longitude_indices, latitude_indices) in ((1:3, 1:4), (2:5, 3:6),
                                                  (1:size(whole, 1), 1:size(whole, 2)),
                                                  (:, :))
        data, window_longitude, window_latitude =
            NumericalEarth.DataWrangling.retrieve_window(metadatum, longitude_indices, latitude_indices)
        @test isequal(Array(data), whole[longitude_indices, latitude_indices, :])
        @test Array(window_longitude) ≈ λ[longitude_indices]
        @test Array(window_latitude) ≈ φ[latitude_indices]
    end
end
