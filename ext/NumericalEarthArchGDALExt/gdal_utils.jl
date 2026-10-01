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
