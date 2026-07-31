#####
##### Global Human Settlement Layer (GHSL) built-up raster ingest:
##### World Mollweide (ESRI:54009) → EPSG:4326.
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
    url = ghsl_tile_url(dataset, row, column)
    zip_path = joinpath(cache_dir, basename(url))
    # Download to a staging file and rename on success, so an interrupted transfer
    # never leaves a truncated archive that the `isfile` guard would keep forever.
    if !isfile(zip_path)
        staging = zip_path * ".part"
        try
            Downloads.download(url, staging)
            mv(staging, zip_path; force = true)
        finally
            rm(staging; force = true)
        end
    end
    inner_tif = ghsl_tile_tif_name(dataset, row, column)
    return string("/vsizip/", zip_path, "/", inner_tif)
end

function NumericalEarth.DataWrangling.GHSL.ghsl_tiles_to_netcdf(metadatum::GHSLMetadatum, nc_path)
    dataset = metadatum.dataset
    region  = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("ghsl_tiles_to_netcdf requires a BoundingBox region.")

    name = dataset_variable_name(metadatum)
    resolution_m = native_resolution(dataset)
    Δ = resolution_m / 111320  # target degree pixel size

    west, east = region.longitude
    south, north = region.latitude

    cache_dir = joinpath(dirname(nc_path), "tiles")
    tiles = ghsl_tiles_in_bbox(region)

    # JRC omits all-ocean tiles from the grid, so a coastal window can reference a tile
    # that 404s; skip those and mosaic the land tiles that do exist.
    sources = String[]
    for (row, column) in tiles
        source = try
            ghsl_download_tile(dataset, row, column, cache_dir)
        catch err
            (err isa Downloads.RequestError && err.response.status == 404) || rethrow()
            @warn "GHSL tile R$(row)_C$(column) is not published (likely all-ocean); skipping."
            nothing
        end
        isnothing(source) || push!(sources, source)
    end
    isempty(sources) &&
        error("No GHSL tiles are published for the requested region; it may be entirely ocean.")

    datasets = [ArchGDAL.read(source) for source in sources]
    raw, longitude, latitude = try
        ArchGDAL.gdalwarp(datasets,
            ["-s_srs", "ESRI:54009",
             "-t_srs", "EPSG:4326",
             "-te",    string(west), string(south), string(east), string(north),
             "-tr",    string(Δ), string(Δ),
             "-r",     "bilinear",
             "-dstnodata", "nan",  # honor the source no-data: keep it out of the bilinear blend and write NaN for gaps
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
    if dataset isa GHSBuiltS
        cell_area = resolution_m^2
        field = built_surface_to_fraction.(raw, cell_area)
    else
        field = mask_building_height.(raw)
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
