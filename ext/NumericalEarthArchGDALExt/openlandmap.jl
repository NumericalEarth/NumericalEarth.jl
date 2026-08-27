#####
##### OpenLandMap-soilDB windowed COG reader
#####

function NumericalEarth.DataWrangling.OpenLandMap.read_cog_window(source, bbox::BoundingBox, factor = 1)
    configure_vsicurl!()

    return ArchGDAL.read(source) do ds
        geotransform = ArchGDAL.getgeotransform(ds)  # [x₀, Δλ, 0, y₀, 0, Δφ]
        validate_geographic_northup(geotransform)
        validate_epsg4326(source_epsg(ds))
        window = cog_window(geotransform, (ArchGDAL.width(ds), ArchGDAL.height(ds)), bbox, factor)

        band   = ArchGDAL.getband(ds, 1)
        scale  = ArchGDAL.getscale(band)
        offset = ArchGDAL.getoffset(band)
        nodata = ArchGDAL.getnodatavalue(band)

        raw = read_cog_band(ds, 1, window)  # (lon, lat), north-first
        # The window's latitude ascends but the rows do not; reverse the data so both come out
        # south-to-north, per CF convention.
        data = reverse(decode_cog_window(raw, scale, offset, nodata), dims = 2)
        return (window.longitude, window.latitude, data)
    end
end
