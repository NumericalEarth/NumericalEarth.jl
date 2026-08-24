#####
##### Shared GDAL helpers for the windowed cloud-optimized GeoTIFF readers
#####

const vsicurl_configured = Ref(false)

function configure_vsicurl!()
    vsicurl_configured[] && return nothing
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    ArchGDAL.setconfigoption("GDAL_HTTP_MULTIRANGE", "YES")

    if !haskey(ENV, "CURL_CA_BUNDLE")
        ENV["CURL_CA_BUNDLE"] = NetworkOptions.ca_roots_path()
    end
    vsicurl_configured[] = true
    return nothing
end

# Decode raw COG integers to Float32 physical values. Order matters: mask nodata
# to NaN first, then apply the band scale/offset (a scaled fill is a spurious value).
function decode_cog_window(raw, scale, offset, nodata)
    decoded = Array{Float32}(undef, size(raw))
    @inbounds for idx in eachindex(raw)
        value = Float64(raw[idx])
        is_nodata = !isnothing(nodata) && isequal(value, nodata)
        decoded[idx] = is_nodata ? NaN32 : Float32(value * scale + offset)
    end
    return decoded
end

"""
    cog_window(geotransform, raster_size, bbox, factor = 1)

Pixel window and output lattice for reading `bbox` from a north-up EPSG:4326 raster of
`raster_size` pixels at `factor` native pixels per output cell side, returned as
`(; xoff, yoff, xsize, ysize, nx, ny, factor, longitude, latitude)`.

The window is snapped outward to whole `factor`-pixel blocks of the raster's own lattice and
padded by one block on each side, so every output cell is an exact block of native pixels — the
lattice a `factor`-decimated read of the whole raster would produce, which is what lets the
framework's coarsened native grid and the window agree cell for cell. The padding keeps the
window a strict superset of that grid, whose center-bracketing can reach one cell past `bbox`.

`longitude` and `latitude` are output cell centers, ascending; GDAL returns rows north-first, so
the data still has to be reversed to match `latitude`.
"""
function cog_window(geotransform, raster_size, bbox, factor = 1)
    x0, dx, _, y0, _, dy = geotransform
    width, height = raster_size
    factor = clamp(factor, 1, min(width, height))

    W, E = bbox.longitude
    S, N = bbox.latitude

    xoff, xsize = block_aligned_range(W, E, x0, dx, width, factor)
    yoff, ysize = block_aligned_range(N, S, y0, dy, height, factor)  # Δφ < 0: north comes first
    nx, ny = xsize ÷ factor, ysize ÷ factor

    # The origin is the outer face of pixel 0, so output cell i starts at pixel
    # xoff + (i - 1) * factor and its center sits half a block further in.
    longitude = [x0 + (xoff + (i - 0.5) * factor) * dx for i in 1:nx]
    latitude  = reverse([y0 + (yoff + (j - 0.5) * factor) * dy for j in 1:ny])

    return (; xoff, yoff, xsize, ysize, nx, ny, factor, longitude, latitude)
end

# Pixel range `[offset, offset + size)` spanning `first_coordinate` through `last_coordinate`
# along an axis of `n` pixels, padded by one `factor`-pixel block on each side and trimmed to
# whole blocks. The trim drops the `n % factor` pixels at the far edge that cannot fill a block.
function block_aligned_range(first_coordinate, last_coordinate, origin, spacing, n, factor)
    i⁻ = factor * (fld(floor(Int, (first_coordinate - origin) / spacing), factor) - 1)
    i⁺ = factor * (cld(ceil( Int, (last_coordinate  - origin) / spacing), factor) + 1)
    last_face = factor * fld(n, factor)
    i⁻ = clamp(i⁻, 0, last_face - factor)
    i⁺ = clamp(i⁺, i⁻ + factor, last_face)
    return i⁻, i⁺ - i⁻
end

# Read one band over `window`. Above `factor` 1 the destination buffer is smaller than the
# window, and that mismatch is what makes GDAL serve the read from the coarsest overview level
# that still resolves it instead of from full-resolution pixels. `AVERAGE` keeps the values means
# of the pixels underneath even when the factor falls between two levels of the pyramid.
function read_cog_band(dataset, band_index, window)
    window.factor == 1 &&
        return ArchGDAL.read(dataset, band_index, window.xoff, window.yoff, window.xsize, window.ysize)

    buffer = Array{Float32}(undef, window.nx, window.ny)
    return ArchGDAL.environment(globalconfig = ["GDAL_RASTERIO_RESAMPLING" => "AVERAGE"]) do
        ArchGDAL.read!(dataset, buffer, band_index,
                       window.xoff, window.yoff, window.xsize, window.ysize)
    end
end

# The windowing math and the north→south row reversal assume a north-up,
# axis-aligned geographic (EPSG:4326, degrees) grid.
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
