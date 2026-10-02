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
        column_offset, row_offset, Nx, Ny = raster_window_indices(geotransform, width, height, bbox)

        band          = ArchGDAL.getband(ds, 1)
        value_scale   = ArchGDAL.getscale(band)
        value_offset  = ArchGDAL.getoffset(band)
        missing_value = ArchGDAL.getnodatavalue(band)

        raw = ArchGDAL.read(ds, 1, column_offset, row_offset, Nx, Ny)  # (lon, lat), north-first
        return assemble_raster_window(raw, geotransform, column_offset, row_offset, value_scale, value_offset, missing_value)
    end
end
