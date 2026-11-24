#####
##### ITS_LIVE windowed COG reader, on the native polar stereographic lattice
#####

function NumericalEarth.DataWrangling.ITSLIVE.read_cog_window(source, window::NamedTuple)
    configure_vsicurl!()

    return ArchGDAL.read(source) do ds
        geotransform           = ArchGDAL.getgeotransform(ds)
        coordinate_system_code = source_coordinate_system_code(ds)

        band          = ArchGDAL.getband(ds, 1)
        value_scale   = ArchGDAL.getscale(band)
        value_offset  = ArchGDAL.getoffset(band)
        missing_value = ArchGDAL.getnodatavalue(band)

        raw  = ArchGDAL.read(ds, 1, window.column_offset, window.row_offset, window.Nx, window.Ny)
        data = reverse(decode_raster_values(raw, value_scale, value_offset, missing_value), dims = 2)
        return (; geotransform, coordinate_system_code, data)
    end
end
