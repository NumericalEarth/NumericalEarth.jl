module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth

const WorldCover = NumericalEarth.DataWrangling.WorldCover

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
##### ESA WorldCover: anonymous COG tiles → regional NetCDF
#####
##### The `Map` band is a UInt8 land-cover class code (no-data = 0). Class codes
##### must never be averaged, so we read the raw 10 m codes windowed to the bbox
##### with NEAREST resampling (which only clips/aligns — it never invents an
##### intermediate code), then aggregate onto the coarse lat/lon grid by an
##### INTEGER factor using the pure helpers in the WorldCover module
##### (`mode_aggregate`, `class_fraction`, `vegetation_fraction`). This keeps the
##### categorical field on its native EPSG:4326 grid — no reprojection.
#####

# S3 key for one 3°×3° tile named by its SW corner (e.g. "N51E006").
function worldcover_tile_url(dataset, tile)
    year = WorldCover.version_year(dataset)
    version = WorldCover.version_string(dataset)
    key = string("v", version[2:end], "/", year, "/map/",
                 "ESA_WorldCover_10m_", year, "_", version, "_", tile, "_Map.tif")
    return string("/vsis3/esa-worldcover/", key)
end

# SW-corner tile label for the 3° grid cell containing (longitude, latitude).
function worldcover_tile_label(longitude, latitude)
    tile_latitude  = 3 * fld(latitude, 3)
    tile_longitude = 3 * fld(longitude, 3)
    ns = tile_latitude  ≥ 0 ? "N" : "S"
    ew = tile_longitude ≥ 0 ? "E" : "W"
    return string(ns, lpad(abs(Int(tile_latitude)),  2, '0'),
                  ew, lpad(abs(Int(tile_longitude)), 3, '0'))
end

# The SW corners of every 3° tile intersecting the bbox.
function worldcover_tiles(longitude_bounds, latitude_bounds)
    λ₁, λ₂ = longitude_bounds
    φ₁, φ₂ = latitude_bounds
    tiles = String[]
    for tile_latitude in (3 * fld(φ₁, 3)):3:(3 * fld(φ₂, 3))
        for tile_longitude in (3 * fld(λ₁, 3)):3:(3 * fld(λ₂, 3))
            push!(tiles, worldcover_tile_label(tile_longitude, tile_latitude))
        end
    end
    return tiles
end

# GDAL_jll ships its own libcurl, which on some platforms (notably macOS) cannot
# locate a system CA bundle and then fails TLS verification against the HTTPS S3
# endpoint ("unable to get local issuer certificate"). If neither variable is
# already set, point curl at the CA bundle that ships with Julia so the anonymous
# HTTPS read can verify the endpoint. An explicit user configuration always wins,
# and TLS verification itself is never disabled.
function ensure_curl_ca_bundle!()
    (haskey(ENV, "CURL_CA_BUNDLE") || haskey(ENV, "SSL_CERT_FILE")) && return nothing
    bundled = normpath(joinpath(Sys.BINDIR, "..", "share", "julia", "cert.pem"))
    isfile(bundled) && (ENV["CURL_CA_BUNDLE"] = bundled)
    return nothing
end

function WorldCover.worldcover_cog_to_netcdf(metadatum::WorldCover.ESAWorldCoverMetadatum, nc_path)
    ensure_curl_ca_bundle!()

    dataset = metadatum.dataset
    region  = metadatum.region

    factor = dataset.aggregation_factor
    native_step = WorldCover.ESA_WORLDCOVER_NATIVE_STEP

    i₁, i₂, j₁, j₂ = WorldCover.worldcover_window(region.longitude, region.latitude, factor)
    west  = i₁ * native_step
    east  = i₂ * native_step
    south = j₁ * native_step
    north = j₂ * native_step

    tile_urls = [worldcover_tile_url(dataset, tile)
                 for tile in worldcover_tiles(region.longitude, region.latitude)]

    # Read the anonymous, unsigned public bucket; `environment` restores any prior
    # AWS/GDAL config afterwards, so a signed `/vsis3` read elsewhere in the same
    # session is not left with signing disabled.
    codes = ArchGDAL.environment(globalconfig = ["AWS_NO_SIGN_REQUEST" => "YES",
                                                 "AWS_REGION" => "eu-central-1"]) do
        # ESA WorldCover only publishes tiles that contain land; a 3° cell that is
        # entirely ocean (or outside coverage) has no tile, so skip a URL that
        # fails to open instead of aborting the whole read.
        sources = ArchGDAL.IDataset[]
        for url in tile_urls
            try
                push!(sources, ArchGDAL.read(url))
            catch tile_error
                @warn "Skipping unavailable ESA WorldCover tile" url exception = tile_error
            end
        end
        isempty(sources) && error("No ESA WorldCover tiles are published for the region " *
                                  "longitude $(region.longitude), latitude $(region.latitude); " *
                                  "it may be entirely ocean or outside the product's coverage " *
                                  "(land only, 60°S–84°N).")

        # Build a VRT mosaic over the available tiles, then read the raw codes on
        # the snapped window at native resolution with nearest resampling.
        try
            ArchGDAL.gdalbuildvrt(sources) do mosaic
                ArchGDAL.gdalwarp([mosaic],
                    ["-te", string(west), string(south), string(east), string(north),
                     "-tr", string(native_step), string(native_step),
                     "-r",  "near",
                     "-ot", "Byte"]) do windowed
                    # (Nx, Ny) with y north-to-south (GDAL convention).
                    data = UInt8.(ArchGDAL.read(windowed, 1))
                    reverse(data, dims = 2)  # flip to south-to-north
                end
            end
        finally
            foreach(ArchGDAL.destroy, sources)
        end
    end

    # Aggregate onto the coarse lat/lon grid by the integer factor in one pass.
    aggregated  = WorldCover.aggregate_landcover(codes, factor)
    class_field = aggregated.landcover_class
    vegetation  = aggregated.vegetation_fraction

    nx, ny = size(class_field)
    Δ = factor * native_step
    longitude = range(west  + Δ / 2, step = Δ, length = nx)
    latitude  = range(south + Δ / 2, step = Δ, length = ny)

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", nx)
        defDim(ds, "lat", ny)

        longitude_variable = defVar(ds, "lon", Float64, ("lon",);
                                    attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        latitude_variable  = defVar(ds, "lat", Float64, ("lat",);
                                    attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        longitude_variable[:] = collect(longitude)
        latitude_variable[:]  = collect(latitude)

        class_variable = defVar(ds, "landcover_class", Float32, ("lon", "lat");
                                attrib = ["long_name" => "majority land-cover class code",
                                          "missing_value" => 0])
        class_variable[:, :] = Float32.(class_field)

        vegetation_variable = defVar(ds, "vegetation_fraction", Float32, ("lon", "lat");
                                     attrib = ["long_name" => "vegetated area fraction",
                                               "units" => "1"])
        vegetation_variable[:, :] = Float32.(vegetation)

        for (name, fraction) in pairs(aggregated.class_fractions)
            band = string(WorldCover.class_fraction_variable_name(name))
            fraction_variable = defVar(ds, band, Float32, ("lon", "lat");
                                       attrib = ["long_name" => string(name, " area fraction"),
                                                 "units" => "1"])
            fraction_variable[:, :] = Float32.(fraction)
        end
    end

    return nothing
end

end # module NumericalEarthArchGDALExt
