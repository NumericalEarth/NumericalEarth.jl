module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using ArchGDAL.GDAL: cplsetconfigoption
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth
using Oceananigans: Center
using Oceananigans.Fields: Field, interior

const CanopyHeight = NumericalEarth.DataWrangling.CanopyHeight
const BoundingBox = NumericalEarth.DataWrangling.BoundingBox

# Candidate CA-certificate bundles, most portable first: Julia's own bundled
# `cert.pem`, then the common system locations (macOS/BSD, Debian/Ubuntu, RHEL).
_ca_bundle_candidates() = (
    normpath(Sys.BINDIR, "..", "share", "julia", "cert.pem"),
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
)

# Configure GDAL's /vsicurl HTTP driver for anonymous COG reads. GDAL_jll's
# bundled libcurl has no CA store on some platforms (notably macOS), so an https
# open fails with "HTTP response code ... : 0" (a transport-layer TLS failure,
# not a 404). Point libcurl at a CA bundle via `CURL_CA_BUNDLE` (respected at
# request time) unless the caller already set one; if no bundle is found, fall
# back to skipping verification so anonymous public COGs still load. Restricting
# the allowed extensions to `.tif` avoids probing sidecar files that don't exist.
function configure_vsicurl!()
    if !haskey(ENV, "CURL_CA_BUNDLE") && !haskey(ENV, "SSL_CERT_FILE")
        ca = findfirst(isfile, _ca_bundle_candidates())
        if ca === nothing
            cplsetconfigoption("GDAL_HTTP_UNSAFESSL", "YES")
        else
            ENV["CURL_CA_BUNDLE"] = _ca_bundle_candidates()[ca]
        end
    end
    cplsetconfigoption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    cplsetconfigoption("GDAL_HTTP_MAX_RETRY", "3")
    cplsetconfigoption("GDAL_HTTP_RETRY_DELAY", "1")
    return nothing
end

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
##### Canopy-height COG (ETH / GLAD) → regional NetCDF
#####

# Mosaic + window the intersecting COG tiles under `geometry` (the gdalwarp `-te`/`-tr`
# or `-te`/`-ts` options), returning the height array in (Nx, Ny) order with latitude
# increasing south→north. A canopy product tiles only over land, so a 3° cell with no
# published tile is a legitimate miss (open ocean), not an error — skip the ones that
# fail to open and mosaic the rest. `identity.` narrows the collected vector back to the
# concrete dataset type so `gdalwarp`'s `Vector{<:AbstractDataset}` method still dispatches.
function warp_canopy_sources(sources, geometry; resampling)
    configure_vsicurl!()
    opened = []
    for source in sources
        try
            push!(opened, ArchGDAL.read(source))
        catch
        end
    end
    isempty(opened) && error("No canopy-height tiles cover the requested region; " *
                             "it may lie entirely over ocean.")
    datasets = identity.(opened)
    options = vcat(String["-t_srs", "EPSG:4326"], geometry,
                   String["-r", resampling, "-ot", "Float32"])
    try
        return ArchGDAL.gdalwarp(datasets, options) do warped
            # GDAL returns (Nx, Ny) with latitude north→south; flip to south→north.
            data = Float32.(ArchGDAL.read(warped, 1))
            return reverse(data, dims = 2)
        end
    finally
        for dataset in datasets
            ArchGDAL.destroy(dataset)
        end
    end
end

# Window the tiles onto the region bbox at a fixed resolution (native-resolution read).
warp_canopy_layer(sources, longitude, latitude, resolution; resampling = "bilinear") =
    warp_canopy_sources(sources,
        String["-te", string(longitude[1]), string(latitude[1]),
               string(longitude[2]), string(latitude[2]),
               "-tr", string(resolution), string(resolution)]; resampling)

# Area-average the tiles onto an explicit (Nx, Ny) grid over the bbox. `-ts` pins the
# output to the grid's cell count (so it drops straight into a grid Field) and `-r average`
# coarse-grains the native pixels within each cell — not point interpolation.
warp_canopy_onto_grid(sources, longitude, latitude, Nx, Ny; resampling = "average") =
    warp_canopy_sources(sources,
        String["-te", string(longitude[1]), string(latitude[1]),
               string(longitude[2]), string(latitude[2]),
               "-ts", string(Nx), string(Ny)]; resampling)

function write_canopy_netcdf(nc_path, longitude, latitude, layers)
    λ₁, λ₂ = longitude
    φ₁, φ₂ = latitude
    # Every layer shares the same window/shape.
    Nx, Ny = size(first(values(layers)))

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)

        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])

        Δλ = (λ₂ - λ₁) / Nx
        Δφ = (φ₂ - φ₁) / Ny
        lon_var[:] = range(λ₁ + Δλ / 2, λ₂ - Δλ / 2; length = Nx)
        lat_var[:] = range(φ₁ + Δφ / 2, φ₂ - Δφ / 2; length = Ny)

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
# and the public read-only share token as basic-auth credentials, and it honours HTTP
# range requests, so `/vsicurl/` fetches only the windowed COG blocks rather than whole
# 415 MB tiles. Nearest-neighbour resampling keeps the categorical 255 no-data byte exact
# so `mask_eth` catches it (bilinear would blend 255 into a valid neighbour).
function CanopyHeight.canopy_height_cog_to_netcdf(metadatum::CanopyHeight.ETHCanopyHeightMetadatum, nc_path)
    cplsetconfigoption("GDAL_HTTP_USERAGENT", CanopyHeight.ETH_BROWSER_USER_AGENT)
    cplsetconfigoption("GDAL_HTTP_USERPWD", CanopyHeight.ETH_LIBDRIVE_TOKEN * ":")
    cplsetconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    region = metadatum.region
    resolution = CanopyHeight.ETH_TILE_RESOLUTION

    sources = CanopyHeight.eth_tile_urls(region, metadatum.name)
    raw = warp_canopy_layer(sources, region.longitude, region.latitude, resolution;
                            resampling = "near")
    layer = NumericalEarth.DataWrangling.dataset_variable_name(metadatum)   # "Map" or "SD"
    layers = Dict(layer => CanopyHeight.mask_eth.(raw, 255))

    write_canopy_netcdf(nc_path, region.longitude, region.latitude, layers)
    return nothing
end

# GLAD: window the intersecting continental forest-height mosaic(s), mask the
# categorical fill codes (101/102/103) to NaN *before* any averaging, keep
# non-forest zeros. Best-effort host (see `CanopyHeight.GLAD_COG_HOST`).
function CanopyHeight.canopy_height_cog_to_netcdf(metadatum::CanopyHeight.GLADCanopyHeightMetadatum, nc_path)
    dataset = metadatum.dataset
    region  = metadatum.region
    resolution = 360 / Base.size(dataset, :canopy_height)[1]

    sources = CanopyHeight.glad_tile_urls(region)
    # Nearest-neighbor read so the categorical fill codes (101/102/103) are never
    # blended before `mask_glad` converts them to NaN.
    raw = warp_canopy_layer(sources, region.longitude, region.latitude, resolution;
                            resampling = "near")
    layers = Dict("Map" => CanopyHeight.mask_glad.(raw))

    write_canopy_netcdf(nc_path, region.longitude, region.latitude, layers)
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

function CanopyHeight.canopy_height_field(grid, ::CanopyHeight.ETHCanopyHeight;
                                          name = :canopy_height, resampling = "average")
    cplsetconfigoption("GDAL_HTTP_USERAGENT", CanopyHeight.ETH_BROWSER_USER_AGENT)
    cplsetconfigoption("GDAL_HTTP_USERPWD", CanopyHeight.ETH_LIBDRIVE_TOKEN * ":")
    cplsetconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    region = BoundingBox(grid)
    sources = CanopyHeight.eth_tile_urls(region, name)
    raw = warp_canopy_onto_grid(sources, region.longitude, region.latitude,
                                size(grid, 1), size(grid, 2); resampling)
    return canopy_field(grid, CanopyHeight.mask_eth.(raw, 255))
end

function CanopyHeight.canopy_height_field(grid, ::CanopyHeight.GLADCanopyHeight; resampling = "average")
    region = BoundingBox(grid)
    sources = CanopyHeight.glad_tile_urls(region)
    raw = warp_canopy_onto_grid(sources, region.longitude, region.latitude,
                                size(grid, 1), size(grid, 2); resampling)
    return canopy_field(grid, CanopyHeight.mask_glad.(raw))
end

end # module NumericalEarthArchGDALExt
