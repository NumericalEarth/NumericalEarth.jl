module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth

using NumericalEarth.DataWrangling: BoundingBox
using NumericalEarth.DataWrangling.MODISLand: MODISLand, MCD43Albedo, MCD12Q1

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
##### MODIS land HDF-EOS ingest (sinusoidal → lat/lon)
#####
#####
##### Resolves the sinusoidal-grid HDF-EOS2 (HDF4) granules intersecting a
##### BoundingBox via CMR, downloads them with Earthdata credentials, warps the
##### requested Science Data Sets from SIN → EPSG:4326 clipped to the region, and
##### writes a regional NetCDF of *raw digital numbers* (no scale/offset applied —
##### the MODISLand module decodes/blends/masks on read). Categorical products are
##### resampled nearest-neighbour; continuous products bilinear.
#####
##### Requires GDAL_jll built with the HDF4 driver (GDAL.jl issue #84); if absent,
##### the subdataset open fails and the caller should fall back to a pre-reprojected
##### (e.g. AppEEARS) NetCDF.
#####

# Which raw SDS layers to warp for a given metadatum, paired with the GDAL
# resampler each layer needs. Continuous fields (albedo, LAI/FPAR) are bilinear;
# categorical or bit-packed fields (PFT class codes, the `FparLai_QC` bitfield)
# MUST be nearest-neighbour — averaging class codes or QA bits is meaningless.
function modis_layers_and_resamplers(metadatum)
    dataset = metadatum.dataset
    name = metadatum.name
    if dataset isa MCD43Albedo
        black_sky, white_sky = MODISLand.MCD43Albedo_variable_names[name]
        return [black_sky, white_sky], ["bilinear", "bilinear"]
    elseif dataset isa MCD12Q1
        return [MODISLand.MCD12Q1_variable_names[name]], ["near"]
    else # LAI / FPAR products, plus the QA layer for masking
        return [MODISLand.MODISLAI_variable_names[name], "FparLai_QC"], ["bilinear", "near"]
    end
end

# Open the `HDF4_EOS:EOS_GRID:"<hdf>":<Grid>:<layer>` subdataset whose name ends in
# `:<layer>`, without hardcoding the per-product grid name.
function modis_subdataset(hdf_path, layer)
    ArchGDAL.read(hdf_path) do dataset
        subdatasets = ArchGDAL.metadata(dataset; domain = "SUBDATASETS")
        for entry in subdatasets
            occursin("_NAME=", entry) || continue
            subdataset_name = split(entry, "_NAME="; limit = 2)[2]
            endswith(subdataset_name, ":" * layer) && return subdataset_name
        end
        error("Layer $(layer) not found among the HDF-EOS subdatasets of $(hdf_path). " *
              "GDAL_jll may lack the HDF4 driver (GDAL.jl issue #84).")
    end
end

function NumericalEarth.DataWrangling.MODISLand.modis_granules_to_netcdf(metadatum::MODISLand.MODISLandMetadatum, nc_path)
    bbox = metadatum.region
    (bbox isa BoundingBox && !isnothing(bbox.longitude) && !isnothing(bbox.latitude)) ||
        error("modis_granules_to_netcdf requires a BoundingBox region.")

    short_name = MODISLand.modis_short_name(metadatum.dataset)
    version = MODISLand.modis_version(metadatum.dataset)
    layers, resamplers = modis_layers_and_resamplers(metadatum)

    temporal = isnothing(metadatum.dates) ? nothing :
               begin
                   date = string(metadatum.dates)
                   string(date, ",", date)
               end

    granule_urls = MODISLand.earthdata_cmr_granules(short_name, version, bbox; temporal)
    isempty(granule_urls) && error("CMR returned no $(short_name).$(version) granules for region $(bbox).")

    # A MODIS granule's temporal extent spans its full multi-day retrieval window, so a
    # single-date search also returns neighbouring granules whose window overlaps that
    # date. Keep only the granule acquired on the requested day (the `.A<yyyy><doy>.`
    # token) so we download one product per tile instead of the whole window.
    if !isnothing(metadatum.dates)
        date = MODISLand.Dates.DateTime(metadatum.dates)
        token = string(".A", MODISLand.Dates.year(date),
                       lpad(MODISLand.Dates.dayofyear(date), 3, '0'), ".")
        matched = filter(url -> occursin(token, url), granule_urls)
        isempty(matched) || (granule_urls = matched)
    end

    west, east = bbox.longitude
    south, north = bbox.latitude

    mktempdir() do tmp
        # Download the intersecting tiles.
        hdf_paths = String[]
        for (n, url) in enumerate(granule_urls)
            hdf_path = joinpath(tmp, string(short_name, "_tile_", n, ".hdf"))
            MODISLand.earthdata_download(url, hdf_path)
            push!(hdf_paths, hdf_path)
        end

        longitude = nothing
        latitude = nothing
        layer_data = Dict{String, Matrix{Float64}}()

        for (layer, resampler) in zip(layers, resamplers)
            # Mosaic the tiles for this layer, then warp SIN → EPSG:4326 clipped to bbox.
            subdatasets = [modis_subdataset(hdf_path, layer) for hdf_path in hdf_paths]
            sources = [ArchGDAL.read(name) for name in subdatasets]
            ArchGDAL.gdalwarp(sources,
                ["-t_srs", "EPSG:4326",
                 "-te", string(west), string(south), string(east), string(north),
                 "-tr", "0.0045", "0.0045",
                 "-r", resampler]) do warped
                data = Float64.(ArchGDAL.read(warped, 1))
                data = reverse(data, dims = 2)  # GDAL writes y north→south
                layer_data[layer] = data
                if isnothing(longitude)
                    Nx, Ny = size(data)
                    geotransform = ArchGDAL.getgeotransform(warped)
                    Δλ = geotransform[2]
                    Δφ = geotransform[6]  # negative
                    longitude = collect(range(geotransform[1] + Δλ / 2; step = Δλ, length = Nx))
                    latitude = collect(range(geotransform[4] + Δφ / 2; step = Δφ, length = Ny))
                    reverse!(latitude)  # match the reversed data
                end
            end
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
            for layer in layers
                # Raw DN, no CF scale/offset/_FillValue — MODISLand decodes on read.
                layer_var = defVar(ds, layer, Float64, ("lon", "lat"))
                layer_var[:, :] = layer_data[layer]
            end
        end
    end

    return nothing
end

end # module NumericalEarthArchGDALExt
