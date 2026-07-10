module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth

const OpenLandMap = NumericalEarth.DataWrangling.OpenLandMap
const BoundingBox = NumericalEarth.DataWrangling.BoundingBox

function NumericalEarth.DataWrangling.IBCAO.reproject_ibcao_to_netcdf(tiff_path, nc_path)
    ArchGDAL.read(tiff_path) do src
        # Warp from EPSG:3996 (Polar Stereographic) to EPSG:4326 (WGS84)
        # at 0.01° resolution, clipping to 64–90°N
        ArchGDAL.gdalwarp([src],
            ["-t_srs", "EPSG:4326",
             "-te",    "-180", "64", "180", "90",  # xmin ymin xmax ymax
             "-tr",    "0.01", "0.01",             # target resolution (degrees)
             "-r",     "bilinear",                 # resampling method
             "-ot",    "Float32"]) do warped

            # ArchGDAL returns data as (Nx, Ny) with y from north to south (GDAL convention)
            data = Float32.(ArchGDAL.read(warped, 1))
            data = reverse(data, dims=2)

            Nx, Ny = size(data)  # expected: (36000, 2600)

            NCDataset(nc_path, "c") do ds
                defDim(ds, "lon", Nx)
                defDim(ds, "lat", Ny)

                lon_var = defVar(ds, "lon", Float64, ("lon",);
                                attrib = ["units" => "degrees_east",
                                          "long_name" => "longitude"])
                lat_var = defVar(ds, "lat", Float64, ("lat",);
                                attrib = ["units" => "degrees_north",
                                          "long_name" => "latitude"])
                z_var   = defVar(ds, "z",   Float32, ("lon", "lat");
                                attrib = ["long_name" => "elevation",
                                          "units"     => "m"])

                lon_var[:] = range(-180 + 0.005, 180 - 0.005; length=Nx)
                lat_var[:] = range(64 + 0.005, 90 - 0.005; length=Ny)
                z_var[:, :] = data
            end
        end
    end

    return nothing
end

#####
##### OpenLandMap-soilDB windowed COG reader
#####

function configure_vsicurl!()
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    ArchGDAL.setconfigoption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    ArchGDAL.setconfigoption("GDAL_HTTP_MULTIRANGE", "YES")

    # GDAL's bundled libcurl reads its CA bundle from CURL_CA_BUNDLE and does not
    # reliably fall back to a system trust store, which breaks https /vsicurl reads.
    # Point it at Julia's cross-platform CA roots unless the user already set it, so
    # an explicit override wins.
    if !haskey(ENV, "CURL_CA_BUNDLE")
        ENV["CURL_CA_BUNDLE"] = NetworkOptions.ca_roots_path()
    end
    return nothing
end

# Decode raw COG integers to Float32 physical values. Order matters: mask nodata
# to NaN first, then apply the band scale/offset (a scaled fill is a spurious value).
function decode_cog_window(raw, scale, offset, nodata)
    decoded = Array{Float32}(undef, size(raw))
    @inbounds for idx in eachindex(raw)
        value = Float64(raw[idx])
        is_nodata = !isnothing(nodata) && value == nodata
        decoded[idx] = is_nodata ? NaN32 : Float32(value * scale + offset)
    end
    return decoded
end

# The windowing math and the north→south row reversal below assume a north-up,
# axis-aligned geographic (EPSG:4326, degrees) grid; fail loudly on anything else.
function validate_geographic_northup(dataset, geotransform)
    _, dx, rx, _, ry, dy = geotransform
    (rx == 0 && ry == 0) ||
        error("Windowed COG reader requires an axis-aligned grid (no rotation/shear); " *
              "got geotransform $geotransform.")
    (dx > 0 && dy < 0) ||
        error("Windowed COG reader assumes west→east (Δλ > 0) and north→south (Δφ < 0) " *
              "pixel order; got Δλ = $dx, Δφ = $dy.")

    # If the source declares a CRS, require EPSG:4326 — the windowing is done in
    # degrees, so a projected grid would silently land the window in the wrong place.
    wkt = ArchGDAL.getproj(dataset)
    if !isempty(wkt)
        epsg = try
            ArchGDAL.toEPSG(ArchGDAL.importWKT(wkt))
        catch  # WKT without an EPSG authority tag: rely on the geometry checks above.
            nothing
        end
        isnothing(epsg) || epsg == 4326 ||
            error("Windowed COG reader expects EPSG:4326 lon/lat in degrees; " *
                  "the source declares EPSG:$epsg.")
    end
    return nothing
end

# Dispatch on `bbox::BoundingBox` (more specific than the module fallback) so this
# adds a specialization rather than overwriting a method during precompilation.
function OpenLandMap.read_cog_window(source, bbox::BoundingBox)
    configure_vsicurl!()

    W, E = bbox.longitude
    S, N = bbox.latitude

    return ArchGDAL.read(source) do ds
        geotransform = ArchGDAL.getgeotransform(ds)  # [x₀, Δλ, 0, y₀, 0, Δφ]
        validate_geographic_northup(ds, geotransform)
        x0, dx, _, y0, _, dy = geotransform
        width  = ArchGDAL.width(ds)
        height = ArchGDAL.height(ds)

        xoff  = clamp(floor(Int, (W - x0) / dx), 0, width - 1)
        yoff  = clamp(floor(Int, (N - y0) / dy), 0, height - 1)
        xsize = clamp(ceil(Int, (E - x0) / dx) - xoff, 1, width - xoff)
        ysize = clamp(ceil(Int, (S - y0) / dy) - yoff, 1, height - yoff)

        band   = ArchGDAL.getband(ds, 1)
        scale  = ArchGDAL.getscale(band)
        offset = ArchGDAL.getoffset(band)
        nodata = ArchGDAL.getnodatavalue(band)

        raw = ArchGDAL.read(ds, 1, xoff, yoff, xsize, ysize)  # (lon, lat), north-first
        # Cell centers: shift the pixel-corner origin x₀ by half a pixel (+0.5·Δ).
        longitude = [x0 + (xoff + i - 0.5) * dx for i in 1:xsize]
        # COGs store rows north-first (Δφ < 0); reverse latitude and data so both
        # come out ascending (south-to-north), per CF convention.
        latitude  = reverse([y0 + (yoff + j - 0.5) * dy for j in 1:ysize])
        data = reverse(decode_cog_window(raw, scale, offset, nodata), dims = 2)
        return (longitude, latitude, data)
    end
end

end # module NumericalEarthArchGDALExt
