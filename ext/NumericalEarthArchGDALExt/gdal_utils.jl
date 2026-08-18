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

# Sentinel returned by `cplgetconfigoption` when an option is unset, so it can be told
# apart from an option the caller genuinely set (GDAL forbids embedded NULs, so this is a
# plain improbable string rather than a NUL-guarded one).
const GDAL_CONFIG_UNSET = "__numericalearth_gdal_config_unset_sentinel__"

# Set GDAL config `options` (key => value pairs) for the duration of `f`, then restore each
# to its prior value or unset it afterwards. GDAL config is process-global, so leaving
# per-host basic-auth credentials set would leak them into any later, unrelated `/vsicurl/`
# read in the same session; scoping keeps them local to the read.
function with_gdal_config(f, options)
    saved = map(options) do (key, _)
        prior = cplgetconfigoption(key, GDAL_CONFIG_UNSET)
        key => (prior == GDAL_CONFIG_UNSET ? C_NULL : prior)
    end
    for (key, value) in options
        cplsetconfigoption(key, value)
    end
    try
        return f()
    finally
        for (key, value) in saved
            cplsetconfigoption(key, value)
        end
    end
end
