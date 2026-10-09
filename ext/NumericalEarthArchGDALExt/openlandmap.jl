#####
##### OpenLandMap-soilDB windowed COG reader
#####

function NumericalEarth.DataWrangling.OpenLandMap.read_cog_window(source, bbox::BoundingBox, factor = 1)
    configure_vsicurl!()

    return ArchGDAL.read(source) do ds
        geotransform = ArchGDAL.getgeotransform(ds)  # [x₀, Δλ, 0, y₀, 0, Δφ]
        validate_geographic_northup(geotransform)
        validate_wgs84_longitude_latitude(source_coordinate_system_code(ds))

        width  = ArchGDAL.width(ds)
        height = ArchGDAL.height(ds)
        column_offset, row_offset, Nx, Ny = raster_window_indices(geotransform, width, height, bbox, factor)

        band          = ArchGDAL.getband(ds, 1)
        value_scale   = ArchGDAL.getscale(band)
        value_offset  = ArchGDAL.getoffset(band)
        missing_value = ArchGDAL.getnodatavalue(band)

        raw = read_raster_band(ds, 1, column_offset, row_offset, Nx, Ny, factor)  # (lon, lat), north-first
        return assemble_raster_window(raw, geotransform, column_offset, row_offset, value_scale, value_offset, missing_value, factor)
    end
end
