#####
##### OpenLandMap-soilDB windowed COG reader
#####

function NumericalEarth.DataWrangling.OpenLandMap.read_cog_window(source, bbox::BoundingBox)
    configure_vsicurl!()

    W, E = bbox.longitude
    S, N = bbox.latitude

    return ArchGDAL.read(source) do ds
        geotransform = ArchGDAL.getgeotransform(ds)  # [x₀, Δλ, 0, y₀, 0, Δφ]
        validate_geographic_northup(ds, geotransform)
        x0, dx, _, y0, _, dy = geotransform
        width  = ArchGDAL.width(ds)
        height = ArchGDAL.height(ds)

        # Pad one native cell on each side so the window is a strict superset of the
        # framework's center-bracketed native grid; otherwise the grid can hold one
        # more cell than the file, forcing a clamped read that shifts the whole
        # window by a pixel and duplicates the outermost row/column.
        xoff  = clamp(floor(Int, (W - x0) / dx) - 1, 0, width - 1)
        yoff  = clamp(floor(Int, (N - y0) / dy) - 1, 0, height - 1)
        xsize = clamp(ceil(Int, (E - x0) / dx) + 1 - xoff, 1, width - xoff)
        ysize = clamp(ceil(Int, (S - y0) / dy) + 1 - yoff, 1, height - yoff)

        band   = ArchGDAL.getband(ds, 1)
        scale  = ArchGDAL.getscale(band)
        offset = ArchGDAL.getoffset(band)
        nodata = ArchGDAL.getnodatavalue(band)

        raw = ArchGDAL.read(ds, 1, xoff, yoff, xsize, ysize)  # (lon, lat), north-first
        # Pixel centers: x₀ is the corner of pixel 0, so the 0-based column
        # (xoff + i - 1) plus half a pixel (+0.5) gives the center: xoff + i - 0.5.
        longitude = [x0 + (xoff + i - 0.5) * dx for i in 1:xsize]
        # COGs store rows north-first (Δφ < 0); reverse latitude and data so both
        # come out ascending (south-to-north), per CF convention.
        latitude  = reverse([y0 + (yoff + j - 0.5) * dy for j in 1:ysize])
        data = reverse(decode_cog_window(raw, scale, offset, nodata), dims = 2)
        return (longitude, latitude, data)
    end
end
