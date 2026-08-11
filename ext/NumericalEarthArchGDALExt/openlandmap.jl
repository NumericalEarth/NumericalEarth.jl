#####
##### OpenLandMap-soilDB windowed COG reader
#####

function NumericalEarth.DataWrangling.OpenLandMap.read_cog_window(source, bbox::BoundingBox)
    configure_vsicurl!()

    return ArchGDAL.read(source) do ds
        geotransform = ArchGDAL.getgeotransform(ds)  # [x₀, Δλ, 0, y₀, 0, Δφ]
        validate_geographic_northup(geotransform)
        validate_epsg4326(source_epsg(ds))

        width  = ArchGDAL.width(ds)
        height = ArchGDAL.height(ds)
        xoff, yoff, xsize, ysize = cog_window_indices(geotransform, width, height, bbox)

        band   = ArchGDAL.getband(ds, 1)
        scale  = ArchGDAL.getscale(band)
        offset = ArchGDAL.getoffset(band)
        nodata = ArchGDAL.getnodatavalue(band)

        raw = ArchGDAL.read(ds, 1, xoff, yoff, xsize, ysize)  # (lon, lat), north-first
        return assemble_cog_window(raw, geotransform, xoff, yoff, scale, offset, nodata)
    end
end
