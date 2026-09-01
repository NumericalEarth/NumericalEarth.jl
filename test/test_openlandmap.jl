include("runtests_setup.jl")

using NumericalEarth.DataWrangling.OpenLandMap
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces, z_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, conversion_units, convert_units,
                                    default_inpainting, is_three_dimensional,
                                    WeightPercent, GramPerCubicCentimeter,
                                    metadata_path, native_grid
using NumericalEarth.DataWrangling.OpenLandMap: cog_window_to_netcdf

using ArchGDAL
using NCDatasets: NCDataset, defDim, defVar

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
function write_synthetic_tile(path; nx, ny, x0, y0, dx, dy, scale, offset, nodata, raw, epsg = 4326)
    ArchGDAL.create(path; driver = ArchGDAL.getdriver("GTiff"),
                    width = nx, height = ny, nbands = 1, dtype = UInt8) do ds
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

@testset "soil_hydraulic_properties from OpenLandMapSoilDB" begin
    dataset = OpenLandMapSoilDB()
    dir = mktempdir()

    grid = LatitudeLongitudeGrid(CPU(); size = (4, 3),
                                 longitude = (0.0, 0.02), latitude = (40.0, 40.015),
                                 topology = (Bounded, Bounded, Flat))

    # The regional files a download would leave behind: uniform texture in the raw
    # units (percent, g/cm3), with a NaN hole under the south-west grid cell.
    values = (sand_fraction = 40.0f0, silt_fraction = 35.0f0,
              clay_fraction = 25.0f0, bulk_density = 1.4f0)
    for (name, value) in pairs(values)
        metadatum = Metadatum(name; dataset, region = BoundingBox(grid), dir)
        native = native_grid(metadatum, CPU())
        Nx, Ny, Nz = size(native)
        data = fill(value, Nx, Ny, Nz)
        name == :sand_fraction && (data[1:Nx÷4, 1:Ny÷3, :] .= NaN32)
        NCDataset(metadata_path(metadatum), "c") do ds
            defDim(ds, "lon", Nx); defDim(ds, "lat", Ny); defDim(ds, "depth", Nz)
            defVar(ds, "lon", Array(λnodes(native, Center())), ("lon",))
            defVar(ds, "lat", Array(φnodes(native, Center())), ("lat",))
            defVar(ds, dataset_variable_name(metadatum), data, ("lon", "lat", "depth"))
        end
    end

    hydraulics = soil_hydraulic_properties(grid, dataset; slab_depth = 0.5, dir)

    # Uniform texture reduces to the same parameters as hand-built lattice fields,
    # with the hole inpainted from its (identical) neighbors.
    lattice = LatitudeLongitudeGrid(CPU(); size = (4, 3, 3),
                                    longitude = (0.0, 0.02), latitude = (40.0, 40.015),
                                    z = [-1.0, -0.6, -0.3, 0.0],
                                    topology = (Bounded, Bounded, Bounded))
    texture = map((0.4f0, 0.35f0, 0.25f0, 1400.0f0)) do value
        field = CenterField(lattice)
        set!(field, value)
        return field
    end
    reference = soil_hydraulic_properties(texture...; slab_depth = 0.5)

    for name in keys(reference)
        parameter = hydraulics[name]
        @test parameter.grid === grid
        @test interior(parameter, :, :, 1) ≈ interior(reference[name], :, :, 1) rtol=1e-6
    end
end
