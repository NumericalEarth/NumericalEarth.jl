include("runtests_setup.jl")

using NumericalEarth.DataWrangling.OpenLandMap
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces, z_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, conversion_units, convert_units,
                                    default_inpainting, is_three_dimensional,
                                    matching_resolution_dataset, minimum_horizontal_spacing,
                                    target_matched_metadata,
                                    WeightPercent, GramPerCubicCentimeter
using NumericalEarth.DataWrangling.OpenLandMap: cog_window_to_netcdf, aggregation_factor, read_step

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

# Build a small GeoTIFF with a known CRS/scale/offset/nodata; row 0 is north.
# `overviews` builds an average-resampled pyramid at the given decimation levels, the way the
# published cloud-optimized GeoTIFFs ship.
function write_synthetic_tile(path; nx, ny, x0, y0, dx, dy, scale, offset, nodata, raw,
                              epsg = 4326, dtype = UInt8, overviews = Cint[])
    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"),
                    width = nx, height = ny, nbands = 1, dtype,
                    options = ["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"]) do ds
        ArchGDAL.setgeotransform!(ds, [x0, dx, 0.0, y0, 0.0, dy])
        ArchGDAL.setproj!(ds, ArchGDAL.toWKT(ArchGDAL.importEPSG(epsg)))
        band = ArchGDAL.getband(ds, 1)
        ArchGDAL.setnodatavalue!(band, Float64(nodata))
        ArchGDAL.GDAL.gdalsetrasterscale(band.ptr, Float64(scale))
        ArchGDAL.GDAL.gdalsetrasteroffset(band.ptr, Float64(offset))
        ArchGDAL.write!(band, raw)
    end

    if !isempty(overviews)
        ArchGDAL.read(path, flags = ArchGDAL.OF_UPDATE) do ds
            ArchGDAL.GDAL.gdalbuildoverviews(ds.ptr, "AVERAGE", length(overviews), overviews,
                                             0, C_NULL, C_NULL, C_NULL)
        end
    end

    return path
end

# Mean of every `factor × factor` block of `raw`, the value a decimated read returns.
block_means(raw, factor) =
    [sum(Float64.(raw[factor*(i-1)+1 : factor*i, factor*(j-1)+1 : factor*j])) / factor^2
     for i in 1:size(raw, 1) ÷ factor, j in 1:size(raw, 2) ÷ factor]

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

@testset "OpenLandMapSoilDB read lattice at an aggregation factor" begin
    native = OpenLandMapSoilDB()
    coarse = OpenLandMapSoilDB(aggregation_factor = 8)

    @test aggregation_factor(native) == 1
    @test read_step(native) ≈ 0.00025
    @test read_step(coarse) ≈ 8 * 0.00025

    # Read cells are whole blocks of native pixels: 1440004 and 528004 leave a remainder of 4,
    # dropped at the far edge from the file origin (east in longitude, south in latitude).
    @test size(native, :clay_fraction) == (1440004, 528004, 3)
    @test size(coarse, :clay_fraction) == (180000, 66000, 3)

    @test longitude_interfaces(coarse)[1] == longitude_interfaces(native)[1]
    @test latitude_interfaces(coarse)[2]  == latitude_interfaces(native)[2]
    @test longitude_interfaces(coarse)[2] < longitude_interfaces(native)[2]
    @test latitude_interfaces(coarse)[1]  > latitude_interfaces(native)[1]

    # The lattice is exactly the factor times the native step.
    Nx, Ny, _ = size(coarse, :clay_fraction)
    Δλ = (longitude_interfaces(coarse)[2] - longitude_interfaces(coarse)[1]) / Nx
    Δφ = (latitude_interfaces(coarse)[2] - latitude_interfaces(coarse)[1]) / Ny
    @test Δλ ≈ read_step(coarse)
    @test Δφ ≈ read_step(coarse)

    @test_throws ArgumentError OpenLandMapSoilDB(aggregation_factor = 0)
end

@testset "OpenLandMapSoilDB matches the read resolution to the target grid" begin
    z = [-1.0, -0.6, -0.3, 0.0]

    # A 0.08° target needs nothing finer than half a cell — 160 native pixels — so the read
    # drops to the largest power of two below that, 128.
    coarse_grid = LatitudeLongitudeGrid(CPU(); size = (10, 10, 3),
                                        longitude = (-112.4, -111.6), latitude = (36.0, 36.8), z)
    @test minimum_horizontal_spacing(coarse_grid) ≈ 0.08
    @test aggregation_factor(matching_resolution_dataset(OpenLandMapSoilDB(), coarse_grid)) == 128

    # A target finer than twice the native step reads at full resolution.
    fine_grid = LatitudeLongitudeGrid(CPU(); size = (10, 10, 3),
                                      longitude = (-112.002, -111.998), latitude = (36.0, 36.004), z)
    @test aggregation_factor(matching_resolution_dataset(OpenLandMapSoilDB(), fine_grid)) == 1

    # An explicit factor pins the read lattice and the target does not override it.
    pinned = OpenLandMapSoilDB(aggregation_factor = 4)
    @test matching_resolution_dataset(pinned, coarse_grid) === pinned

    # The factor keys the cache: a coarse read never shares a file with a finer one, and a
    # full-resolution read keeps the name it has always had.
    region = BoundingBox(longitude = (-112.3, -111.9), latitude = (36.0, 36.4))
    filename(dataset) = metadata_filename(dataset, :clay_fraction, nothing, region)
    @test filename(OpenLandMapSoilDB()) == filename(OpenLandMapSoilDB(aggregation_factor = 1))
    @test filename(OpenLandMapSoilDB(aggregation_factor = 128)) ==
          "OpenLandMap_clay_fraction_f128_lon_-112.3_-111.9_lat_36.0_36.4.nc"

    # The metadatum a target rebuilds carries the matched dataset and its own cache file.
    metadatum = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region)
    matched = target_matched_metadata(metadatum, coarse_grid)
    @test aggregation_factor(matched.dataset) == 128
    @test matched.filename == filename(OpenLandMapSoilDB(aggregation_factor = 128))
    @test matched.name == metadatum.name && matched.region === metadatum.region
end

@testset "OpenLandMapSoilDB windowed COG reader at an aggregation factor" begin
    dir = mktempdir()
    nx, ny = 64, 64
    x0, y0, dx, dy = -112.0005, 36.0005, 0.00025, -0.00025
    scale, offset, nodata = 1.0, 0.0, -1.0

    raw = Float32[(i % 13) + 2 * (j % 7) for i in 1:nx, j in 1:ny]
    raw[1:4, 1:4] .= nodata   # one whole read cell of no-data at the north-west corner

    factor = 4
    bbox = BoundingBox(longitude = (x0, x0 + nx * dx), latitude = (y0 + ny * dy, y0))
    expected = reverse(block_means(raw, factor), dims = 2)  # file rows are north-first

    # With and without a pyramid: GDAL serves the read from the overviews when they exist and
    # decimates the full-resolution pixels when they do not, and both are block means.
    for (label, overviews) in ("with overviews" => Cint[2, 4], "without overviews" => Cint[])
        @testset "$label" begin
            tif = write_synthetic_tile(joinpath(dir, "tile_$(length(overviews)).tif");
                                       nx, ny, x0, y0, dx, dy, scale, offset, nodata, raw,
                                       dtype = Float32, overviews)

            nc = joinpath(dir, "window_$(length(overviews)).nc")
            cog_window_to_netcdf([tif], nc, "clay", bbox, factor)

            NCDataset(nc) do ds
                lon = ds["lon"][:]
                lat = ds["lat"][:]
                data = ds["clay"][:, :, 1]

                @test size(data) == (nx ÷ factor, ny ÷ factor)

                # Coordinates are the centers of the coarsened cells, ascending in both axes.
                @test lon[1] ≈ x0 + factor * dx / 2
                @test lat[end] ≈ y0 + factor * dy / 2
                @test lon[2] - lon[1] ≈ factor * dx
                @test issorted(lon) && issorted(lat)

                # A cell entirely of no-data stays masked; the rest are means of their blocks.
                @test isnan(data[1, end])
                valid = .!isnan.(data)
                @test count(valid) == length(data) - 1
                @test data[valid] ≈ expected[valid]
            end
        end
    end
end

@testset "OpenLandMapSoilDB regrids a target-matched read onto the target grid" begin
    dir = mktempdir()

    # A tile on the dataset's own global lattice, so its read blocks coincide with the cells of
    # the coarsened native grid. Values ramp linearly with the pixel indices, a field that
    # survives both block-averaging and bilinear interpolation exactly.
    nx, ny = 512, 512
    x0, y0, dx, dy = -112.0005, 36.0005, 0.00025, -0.00025
    raw = Float32[(i - 0.5) + (j - 0.5) for i in 1:nx, j in 1:ny]
    tif = write_synthetic_tile(joinpath(dir, "ramp.tif");
                               nx, ny, x0, y0, dx, dy, scale = 1.0, offset = 0.0,
                               nodata = -1.0, raw, dtype = Float32, overviews = Cint[2, 4, 8, 16])

    # 0.01° cells: half of that is 20 native pixels, so the read is matched at factor 16.
    grid = LatitudeLongitudeGrid(CPU(); size = (10, 10, 3),
                                 longitude = (-111.98, -111.88), latitude = (35.89, 35.99),
                                 z = [-1.0, -0.6, -0.3, 0.0])
    region = BoundingBox(grid)

    metadatum = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region, dir)
    matched = target_matched_metadata(metadatum, grid)
    @test aggregation_factor(matched.dataset) == 16

    # Materialize the window the matched metadatum names, so `Field` reads it instead of
    # downloading. All three depths carry the same ramp.
    cog_window_to_netcdf(fill(tif, 3), metadata_path(matched), "clay", region,
                         aggregation_factor(matched.dataset))

    field = Field(metadatum, grid)

    # `:clay_fraction` is a weight percent, so the stored ramp comes back divided by 100.
    λ = λnodes(grid, Center())
    φ = φnodes(grid, Center())
    expected = [((λ[i] - x0) / dx + (y0 - φ[j]) / abs(dy)) / 100 for i in 1:10, j in 1:10]

    @test !any(isnan, interior(field))
    for k in 1:3
        @test interior(field, :, :, k) ≈ expected rtol = 1e-4
    end
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
                               nodata = -1.0, raw, dtype = Float32, overviews = Cint[2, 4, 8, 16])

    grid = LatitudeLongitudeGrid(CPU(); size = (10, 10, 3),
                                 longitude = (-111.98, -111.88), latitude = (35.89, 35.99),
                                 z = [-1.0, -0.6, -0.3, 0.0])
    region = BoundingBox(grid)

    metadatum = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region, dir)
    matched = target_matched_metadata(metadatum, grid)
    cog_window_to_netcdf(fill(tif, 3), metadata_path(matched), "clay", region,
                         aggregation_factor(matched.dataset))

    # The path the regrid took before tiling: materialize the whole window, then interpolate.
    native = Field(matched, CPU())
    untiled = Field{Center, Center, Center}(grid)
    NumericalEarth.DataWrangling.interpolate_physical!(untiled, native, matched)

    # Tiling changes where the data comes from, never the arithmetic done on it: each tile is a
    # windowed field over the native grid, so it interpolates from the same node coordinates the
    # whole field carries. Every budget must therefore reproduce the untiled answer bitwise.
    reference = Array(interior(untiled))

    for tile_bytes in (typemax(Int), 1_000_000, 10_000, 512)
        tiled = Array(interior(Field(metadatum, grid; tile_bytes)))
        @test size(tiled) == size(reference)
        @test isequal(tiled, reference)
    end

    # The smallest budget really does split the target, otherwise the loop proves nothing.
    @test NumericalEarth.DataWrangling.tile_count(size(native.grid)[1:2], 3, 512) > 1
end

@testset "tiled regrid declines where it would not be equivalent" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (10, 10, 3),
                                 longitude = (-111.98, -111.88), latitude = (35.89, 35.99),
                                 z = [-1.0, -0.6, -0.3, 0.0])
    field = Field{Center, Center, Center}(grid)
    region = BoundingBox(grid)
    tiled_native_grid = NumericalEarth.DataWrangling.tiled_native_grid

    matched = target_matched_metadata(Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region), grid)
    @test !isnothing(tiled_native_grid(field, matched, nothing))

    # Inpainting is an iterative fill over the whole field; no tiling of it reproduces that.
    @test isnothing(tiled_native_grid(field, matched,
                                      NumericalEarth.DataWrangling.NearestNeighborInpainting(2)))

    # A dataset that cannot read a window would re-read the whole file per tile.
    unwindowed = Metadatum(:temperature; dataset = ECCO4Monthly(),
                           date = DateTime(1993, 1, 1), region)
    @test isnothing(tiled_native_grid(field, unwindowed, nothing))
end

@testset "windowed NetCDF retrieval matches the whole-file read" begin
    dir = mktempdir()
    nx, ny = 256, 256
    x0, y0, dx, dy = -112.0005, 36.0005, 0.00025, -0.00025
    raw = Float32[(i % 11) + 3 * (j % 5) for i in 1:nx, j in 1:ny]

    tif = write_synthetic_tile(joinpath(dir, "window.tif");
                               nx, ny, x0, y0, dx, dy, scale = 1.0, offset = 0.0,
                               nodata = -1.0, raw, dtype = Float32, overviews = Cint[2, 4])

    region = BoundingBox(longitude = (x0 + 0.01, x0 + 0.05), latitude = (y0 - 0.05, y0 - 0.01))
    metadatum = Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(aggregation_factor = 4),
                          region, dir)
    cog_window_to_netcdf(fill(tif, 3), metadata_path(metadatum), "clay", region, 4)

    whole = NumericalEarth.DataWrangling.retrieve_data(metadatum)
    λ, φ = NumericalEarth.DataWrangling.read_file_coords(metadatum)

    # Both the dataset's own hyperslab reader and the shared NetCDF one must reproduce a view of
    # the whole-file read, coordinates included.
    for reader in (NumericalEarth.DataWrangling.retrieve_window,
                   NumericalEarth.DataWrangling.netcdf_retrieve_window)
        for (longitude_indices, latitude_indices) in ((1:3, 1:4), (2:5, 3:6), (1:size(whole, 1), 1:size(whole, 2)))
            data, λw, φw = reader(metadatum, longitude_indices, latitude_indices)
            @test isequal(Array(data), whole[longitude_indices, latitude_indices, :])
            @test Array(λw) ≈ λ[longitude_indices]
            @test Array(φw) ≈ φ[latitude_indices]
        end
    end
end

@testset "north-first files map their window rows back before slicing" begin
    file_latitude_rows = NumericalEarth.DataWrangling.file_latitude_rows

    # An ascending file passes its rows through untouched.
    @test file_latitude_rows(10, 3:5, false) === 3:5

    # A north-first file of n rows holds ascending row j at file row n - j + 1, so a window maps
    # to the mirrored range and is flipped after slicing.
    @test file_latitude_rows(10, 3:5, true) == 6:8
    @test file_latitude_rows(10, 1:10, true) == 1:10

    # The mapping must satisfy: reversing the whole file then slicing equals slicing the mirrored
    # rows then reversing — the identity the windowed reader relies on.
    stored = [10i + j for i in 1:4, j in 1:10]
    ascending = reverse(stored, dims = 2)
    for latitude_indices in (1:3, 4:7, 2:2, 1:10)
        rows = file_latitude_rows(10, latitude_indices, true)
        @test ascending[:, latitude_indices] == reverse(stored[:, rows], dims = 2)
    end
end

@testset "a dataset regridded by something other than bilinear declines tiling" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (8, 8), longitude = (5.0, 5.4), latitude = (50.0, 50.4),
                                 topology = (Bounded, Bounded, Flat))
    field = CenterField(grid)
    region = BoundingBox(grid)

    # WorldCover counts class codes rather than interpolating them.
    worldcover = Metadatum(:vegetation_fraction; dataset = ESAWorldCover(), region)
    @test isnothing(NumericalEarth.DataWrangling.tiled_native_grid(field, worldcover, nothing))
end
