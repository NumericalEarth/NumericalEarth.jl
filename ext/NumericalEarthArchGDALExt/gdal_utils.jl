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

# Return the EPSG catalog code identifying the dataset's coordinate system,
# or `nothing` if its coordinate-system metadata cannot supply one.
function source_coordinate_system_code(dataset)
    wkt = ArchGDAL.getproj(dataset)
    isempty(wkt) && return nothing
    return try
        ArchGDAL.toEPSG(ArchGDAL.importWKT(wkt))
    catch
        nothing
    end
end

# Read one band over the native-pixel window into a `factor`-times-coarser buffer. A destination
# buffer smaller than the window makes GDAL serve the read from the coarsest overview level that
# resolves it; `AVERAGE` keeps the values means of the pixels underneath even when the factor
# falls between two levels of the pyramid.
function read_raster_band(dataset, band_index, column_offset, row_offset, Nx, Ny, factor)
    factor == 1 && return ArchGDAL.read(dataset, band_index, column_offset, row_offset, Nx, Ny)

    buffer = Array{Float32}(undef, Nx ÷ factor, Ny ÷ factor)
    return ArchGDAL.environment(globalconfig = ["GDAL_RASTERIO_RESAMPLING" => "AVERAGE"]) do
        ArchGDAL.read!(dataset, buffer, band_index, column_offset, row_offset, Nx, Ny)
    end
end
