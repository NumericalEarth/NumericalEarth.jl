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

# `nothing` when the source declares no CRS, or a WKT carrying no EPSG authority tag.
function source_epsg(dataset)
    wkt = ArchGDAL.getproj(dataset)
    isempty(wkt) && return nothing
    return try
        ArchGDAL.toEPSG(ArchGDAL.importWKT(wkt))
    catch
        nothing
    end
end
