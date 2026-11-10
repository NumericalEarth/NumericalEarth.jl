#####
##### SoilGrids2 ISRIC ingest: warp each depth's remote Homolosine (IGH) VRT directly to
##### EPSG:4326 via GDAL's `/vsicurl` (no local intermediate GeoTIFF is downloaded — GDAL
##### streams the needed bytes via HTTP range requests during the warp itself), stacking the
##### six depths into the global or regional NetCDF the read path expects.
#####

# ISRIC depth-range URL fragments, deepest first (matching `z_interfaces(::SoilGrids2)`).
const soilgrids_depth_ranges = ("100-200cm", "60-100cm", "30-60cm", "15-30cm", "5-15cm", "0-5cm")

# Warp one depth's remote VRT directly onto `raster`'s EPSG:4326 grid.
function soilgrids_depth_window(source, raster)
    configure_vsicurl!()
    dataset = ArchGDAL.read(source)
    data = try
        ArchGDAL.gdalwarp([dataset],
            ["-s_srs", "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs",
             "-t_srs", "EPSG:4326",
             "-te",    string(raster.west), string(raster.south),
                       string(raster.east), string(raster.north),
             "-ts",    string(raster.Nx), string(raster.Ny),
             "-r",     "bilinear",
             "-dstnodata", "nan",
             "-ot",    "Float32"]) do warped
            raw = Float32.(ArchGDAL.read(warped, 1))
            return reverse(raw, dims = 2)  # GDAL writes y north→south
        end
    finally
        ArchGDAL.destroy(dataset)
    end
    return data
end

function NumericalEarth.DataWrangling.SoilGrids.soilgrids_variable_to_netcdf(metadatum::SoilGrids2Metadatum, nc_path)
    var = dataset_variable_name(metadatum)
    stat_name = soilgrids_statistic_url_name(metadatum.dataset.statistic)
    sources = [soilgrids_vsicurl_source(var, depth, stat_name) for depth in soilgrids_depth_ranges]
    raster = soilgrids_raster_geometry(metadatum)
    soilgrids_sources_to_netcdf(sources, var, raster, nc_path)
    return nothing
end

# Warp each depth's source (deepest first) onto `raster` and stack them into a
# `(lon, lat, depth)` NetCDF. Separate from the URL construction above so that it
# can be exercised on local rasters.
function soilgrids_sources_to_netcdf(sources, var, raster, nc_path)
    Nz = length(sources)

    data = Array{Float32}(undef, raster.Nx, raster.Ny, Nz)
    for (k, source) in enumerate(sources)
        data[:, :, k] = soilgrids_depth_window(source, raster)
    end

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", raster.Nx)
        defDim(ds, "lat", raster.Ny)
        defDim(ds, "depth", Nz)
        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        chunk    = [min(512, raster.Nx), min(512, raster.Ny), Nz]
        var_data = defVar(ds, var, Float32, ("lon", "lat", "depth"); chunksizes = chunk)

        lon_var[:]         = raster.longitude
        lat_var[:]         = raster.latitude
        var_data[:, :, :]  = data
    end

    return nothing
end
