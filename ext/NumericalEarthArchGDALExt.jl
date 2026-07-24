module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using ArchGDAL.GDAL: cplsetconfigoption, cplgetconfigoption
using NCDatasets: NCDataset, defDim, defVar
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth
using Oceananigans: Center
using Oceananigans.Fields: Field, interior

const ETHSentinel2Canopy = NumericalEarth.DataWrangling.ETHSentinel2Canopy
const BoundingBox = NumericalEarth.DataWrangling.BoundingBox

# Configure GDAL's /vsicurl HTTP driver for anonymous COG reads, once per session.
# GDAL_jll's bundled libcurl has no CA store on some platforms (notably macOS), so an
# https open fails with "HTTP response code ... : 0" (a transport-layer TLS failure,
# not a 404). Point libcurl at a CA bundle via `CURL_CA_BUNDLE` (respected at request
# time) unless the caller already set one; `NetworkOptions.ca_roots_path()` resolves
# the platform's trust store — Julia's bundled `cert.pem` by default, or whatever
# `SSL_CERT_FILE`/`JULIA_SSL_CA_ROOTS_PATH` point at.
const vsicurl_configured = Ref(false)

function configure_vsicurl!()
    vsicurl_configured[] && return nothing
    if !haskey(ENV, "CURL_CA_BUNDLE")
        ENV["CURL_CA_BUNDLE"] = NetworkOptions.ca_roots_path()
    end
    cplsetconfigoption("GDAL_HTTP_MAX_RETRY", "3")
    cplsetconfigoption("GDAL_HTTP_RETRY_DELAY", "1")
    vsicurl_configured[] = true
    return nothing
end

# Sentinel returned by `cplgetconfigoption` when an option is unset, so it can be told
# apart from an option the caller genuinely set (GDAL forbids embedded NULs, so this is a
# plain improbable string rather than a NUL-guarded one).
const GDAL_CONFIG_UNSET = "__numericalearth_gdal_config_unset_sentinel__"

# Set GDAL config `options` (key => value pairs) for the duration of `f`, then restore each
# to its prior value — or unset it — afterwards. GDAL config is process-global, so leaving
# per-host basic-auth credentials (the ETH share token) set would leak them into any later,
# unrelated `/vsicurl/` read in the same session; scoping keeps them local to the read.
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

# Credentials for the ETH libdrive WebDAV share: a browser User-Agent and the public
# read-only share token as basic-auth, plus the /vsicurl directory-listing suppression.
eth_http_config() =
    ["GDAL_HTTP_USERAGENT"          => ETHSentinel2Canopy.ETH_BROWSER_USER_AGENT,
     "GDAL_HTTP_USERPWD"            => ETHSentinel2Canopy.ETH_LIBDRIVE_TOKEN * ":",
     "GDAL_DISABLE_READDIR_ON_OPEN" => "EMPTY_DIR"]

function NumericalEarth.DataWrangling.IBCAO.reproject_ibcao_to_netcdf(tiff_path, nc_path)
    ArchGDAL.read(tiff_path) do src
        # Warp from EPSG:3996 (Polar Stereographic) to EPSG:4326 (WGS84)
        # at 0.01° resolution, clipping to 64–90°N
        ArchGDAL.gdalwarp([src],
            ["-t_srs", "EPSG:4326",
             "-te",    "-180", "64", "180", "90",  # xmin ymin xmax ymax
             "-tr",    "0.01", "0.01",             # target resolution (degrees)
             "-r",     "bilinear",                 # resampling method
             "-ot",    "Float32"]) do warped

            # ArchGDAL returns data as (Nx, Ny) with y from north to south (GDAL convention)
            data = Float32.(ArchGDAL.read(warped, 1))
            data = reverse(data, dims=2)

            Nx, Ny = size(data)  # expected: (36000, 2600)

            NCDataset(nc_path, "c") do ds
                defDim(ds, "lon", Nx)
                defDim(ds, "lat", Ny)

                lon_var = defVar(ds, "lon", Float64, ("lon",);
                                attrib = ["units" => "degrees_east",
                                          "long_name" => "longitude"])
                lat_var = defVar(ds, "lat", Float64, ("lat",);
                                attrib = ["units" => "degrees_north",
                                          "long_name" => "latitude"])
                z_var   = defVar(ds, "z",   Float32, ("lon", "lat");
                                attrib = ["long_name" => "elevation",
                                          "units"     => "m"])

                lon_var[:] = range(-180 + 0.005, 180 - 0.005; length=Nx)
                lat_var[:] = range(64 + 0.005, 90 - 0.005; length=Ny)
                z_var[:, :] = data
            end
        end
    end

    return nothing
end

#####
##### Canopy-height COG (ETH) → regional NetCDF
#####

# Mosaic + window the intersecting COG tiles under `geometry` (the gdalwarp `-te`/`-tr`
# or `-te`/`-ts` options), returning `(; data, longitude, latitude)` — the height array in
# (Nx, Ny) order with latitude increasing south→north, and the cell-center coordinates read
# straight off the warped geotransform. A canopy product tiles only over land, so a 3° cell
# with no published tile is a legitimate miss (open ocean), not an error — skip the ones that
# fail to open and mosaic the rest, but surface every read error if *all* fail (an all-ocean
# region and a network/TLS/credential failure both leave nothing opened). `identity.` narrows
# the collected vector back to the concrete dataset type so `gdalwarp`'s
# `Vector{<:AbstractDataset}` method still dispatches.
function warp_canopy_sources(sources, geometry; resampling, nodata = nothing)
    configure_vsicurl!()
    opened = []
    read_errors = Pair{Any, Any}[]
    for source in sources
        try
            push!(opened, ArchGDAL.read(source))
        catch err
            err isa InterruptException && rethrow()
            push!(read_errors, source => err)
        end
    end
    isempty(opened) && error(
        "No canopy-height tiles could be read for the requested region. The product " *
        "publishes no tile over open ocean, so an all-ocean region legitimately yields " *
        "nothing — but a network, TLS, or credential failure produces the same empty " *
        "result. Underlying read errors:\n" *
        join(("  $source: $(sprint(showerror, err))" for (source, err) in read_errors), "\n"))
    datasets = identity.(opened)
    # Declare the categorical no-data byte so `-r average` drops it from cell means rather
    # than blending it in; all-no-data cells then come out as `nodata` for the caller to mask.
    nodata_options = isnothing(nodata) ? String[] :
                     String["-srcnodata", string(nodata), "-dstnodata", string(nodata)]
    options = vcat(String["-t_srs", "EPSG:4326"], geometry,
                   String["-r", resampling, "-ot", "Float32"], nodata_options)
    try
        return ArchGDAL.gdalwarp(datasets, options) do warped
            gt = ArchGDAL.getgeotransform(warped)   # [x₀, Δλ, 0, y₀, 0, -Δφ]
            band = Float32.(ArchGDAL.read(warped, 1))
            Nx, Ny = size(band)
            # GDAL rows run north→south; flip to south→north. Cell centers come from the
            # geotransform, so they stay exact even when `-tr` snaps the extent to whole pixels.
            data = reverse(band, dims = 2)
            longitude = [gt[1] + (i - 0.5) * gt[2] for i in 1:Nx]
            latitude  = [gt[4] + (Ny - j + 0.5) * gt[6] for j in 1:Ny]
            return (; data, longitude, latitude)
        end
    finally
        for dataset in datasets
            ArchGDAL.destroy(dataset)
        end
    end
end

# Window the tiles onto the region bbox at a fixed resolution (native-resolution read).
warp_canopy_layer(sources, longitude, latitude, resolution; resampling = "near", nodata = nothing) =
    warp_canopy_sources(sources,
        String["-te", string(longitude[1]), string(latitude[1]),
               string(longitude[2]), string(latitude[2]),
               "-tr", string(resolution), string(resolution)]; resampling, nodata)

# Area-average the tiles onto an explicit (Nx, Ny) grid over the bbox. `-ts` pins the
# output to the grid's cell count (so it drops straight into a grid Field) and `-r average`
# coarse-grains the native pixels within each cell — not point interpolation. `nodata`
# excludes the product no-data byte from those means (see `warp_canopy_sources`).
warp_canopy_onto_grid(sources, longitude, latitude, Nx, Ny; resampling = "average", nodata = nothing) =
    warp_canopy_sources(sources,
        String["-te", string(longitude[1]), string(latitude[1]),
               string(longitude[2]), string(latitude[2]),
               "-ts", string(Nx), string(Ny)]; resampling, nodata)

# `longitude`/`latitude` are the cell-center coordinate vectors from `warp_canopy_sources`.
function write_canopy_netcdf(nc_path, longitude, latitude, layers)
    Nx = length(longitude)
    Ny = length(latitude)

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)

        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])

        lon_var[:] = longitude
        lat_var[:] = latitude

        for (name, data) in layers
            var = defVar(ds, name, Float32, ("lon", "lat");
                         attrib = ["long_name" => "canopy height", "units" => "m"])
            var[:, :] = data
        end
    end

    return nothing
end

# ETH: window the intersecting 3° 10 m COG tiles (libdrive WebDAV) for the requested
# layer at the native resolution, mask the no-data byte (255) to NaN — keeping non-forest
# zeros — and write one regional NetCDF. The WebDAV endpoint needs a browser User-Agent
# and the public read-only share token as basic-auth credentials, and it honors HTTP
# range requests, so `/vsicurl/` fetches only the windowed COG blocks rather than whole
# 415 MB tiles. Nearest-neighbor resampling keeps the categorical 255 no-data byte exact
# so `mask_eth` catches it (bilinear would blend 255 into a valid neighbor).
function ETHSentinel2Canopy.canopy_height_cog_to_netcdf(metadatum::ETHSentinel2Canopy.ETHSentinel2CanopyHeightMetadatum, nc_path)
    region = metadatum.region
    resolution = ETHSentinel2Canopy.ETH_TILE_RESOLUTION
    sources = ETHSentinel2Canopy.eth_tile_urls(region, metadatum.name)

    warped = with_gdal_config(eth_http_config()) do
        warp_canopy_layer(sources, region.longitude, region.latitude, resolution;
                          resampling = "near")
    end

    layer = NumericalEarth.DataWrangling.dataset_variable_name(metadatum)   # "Map" or "SD"
    layers = Dict(layer => ETHSentinel2Canopy.mask_eth.(warped.data, 255))
    write_canopy_netcdf(nc_path, warped.longitude, warped.latitude, layers)
    return nothing
end

#####
##### Direct area-averaged read onto a model grid (coarse-graining, no NetCDF)
#####

# Drop a masked (Nx, Ny) canopy array straight into a grid Field; `-ts` guarantees the
# array matches the grid's cell count.
function canopy_field(grid, data)
    h = Field{Center, Center, Nothing}(grid)
    interior(h, :, :, 1) .= data
    return h
end

function ETHSentinel2Canopy.canopy_height_field(grid, ::ETHSentinel2Canopy.ETHSentinel2CanopyHeight;
                                          name = :canopy_height, resampling = "average")
    region = BoundingBox(grid)
    sources = ETHSentinel2Canopy.eth_tile_urls(region, name)

    warped = with_gdal_config(eth_http_config()) do
        warp_canopy_onto_grid(sources, region.longitude, region.latitude,
                              size(grid, 1), size(grid, 2); resampling, nodata = 255)
    end

    return canopy_field(grid, ETHSentinel2Canopy.mask_eth.(warped.data, 255))
end

end # module NumericalEarthArchGDALExt
