module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using Downloads: Downloads
using NumericalEarth: NumericalEarth

using NumericalEarth.DataWrangling: BoundingBox
const GHSL = NumericalEarth.DataWrangling.GHSL

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
##### GHSL built-up raster ingest (World Mollweide ESRI:54009 → EPSG:4326)
#####
#####
##### Downloads the GHSL R2023A tile archives intersecting a BoundingBox from the JRC
##### open-data host, reads the Mollweide GeoTIFF inside each `.zip` in place with
##### GDAL's `/vsizip/`, mosaics + reprojects them to EPSG:4326 clipped to the region,
##### masks the no-data, converts built-up surface (m²/cell) to a plan-area fraction,
##### and writes a regional lat/lon NetCDF that `GHSL.retrieve_data` reads back.
#####

# Cache a GHSL tile archive next to the regional NetCDF, keyed by tile so it is reused
# across regions. Idempotent.
function ghsl_download_tile(dataset, row, column, cache_dir)
    mkpath(cache_dir)
    url = GHSL.ghsl_tile_url(dataset, row, column)
    zip_path = joinpath(cache_dir, basename(url))
    isfile(zip_path) || Downloads.download(url, zip_path)
    inner_tif = GHSL.ghsl_tile_tif_name(dataset, row, column)
    return string("/vsizip/", zip_path, "/", inner_tif)
end

function NumericalEarth.DataWrangling.GHSL.ghsl_tiles_to_netcdf(metadatum::GHSL.GHSLMetadatum, nc_path)
    dataset = metadatum.dataset
    region  = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("ghsl_tiles_to_netcdf requires a BoundingBox region.")

    name = GHSL.DataWrangling.dataset_variable_name(metadatum)
    resolution_m = GHSL.native_resolution(dataset)
    Δ = resolution_m / 111320  # target degree pixel size

    west, east = region.longitude
    south, north = region.latitude

    cache_dir = joinpath(dirname(nc_path), "tiles")
    tiles = GHSL.ghsl_tiles_in_bbox(region)
    sources = [ghsl_download_tile(dataset, row, column, cache_dir) for (row, column) in tiles]

    datasets = [ArchGDAL.read(source) for source in sources]
    raw, longitude, latitude = try
        ArchGDAL.gdalwarp(datasets,
            ["-s_srs", "ESRI:54009",
             "-t_srs", "EPSG:4326",
             "-te",    string(west), string(south), string(east), string(north),
             "-tr",    string(Δ), string(Δ),
             "-r",     "bilinear",
             "-ot",    "Float32"]) do warped
            data = Float64.(ArchGDAL.read(warped, 1))
            data = reverse(data, dims = 2)  # GDAL writes y north→south
            Nx, Ny = size(data)
            geotransform = ArchGDAL.getgeotransform(warped)
            Δλ = geotransform[2]
            Δφ = geotransform[6]  # negative
            lon = collect(range(geotransform[1] + Δλ / 2; step = Δλ, length = Nx))
            lat = collect(range(geotransform[4] + Δφ / 2; step = Δφ, length = Ny))
            reverse!(lat)  # match the reversed data
            return data, lon, lat
        end
    finally
        for d in datasets
            ArchGDAL.destroy(d)
        end
    end

    # Mask no-data and, for built-up surface, convert m²/cell → plan-area fraction.
    if dataset isa GHSL.GHSBuiltS
        cell_area = resolution_m^2
        field = GHSL.built_surface_to_fraction.(raw, cell_area)
    else
        field = GHSL.mask_building_height.(raw)
    end

    NCDataset(nc_path, "c") do ds
        Nx = length(longitude)
        Ny = length(latitude)
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        lon_var[:] = longitude
        lat_var[:] = latitude
        var = defVar(ds, name, Float64, ("lon", "lat"))
        var[:, :] = field
    end

    return nothing
end

end # module NumericalEarthArchGDALExt
