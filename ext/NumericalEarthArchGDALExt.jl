module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox
using NumericalEarth.DataWrangling.MODISLand: MODISLand, earthdata_download, granule_urls,
                                              regional_lattice, stored_granule_layers

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
##### MODIS land products — Earthdata granule fetch + sinusoidal HDF-EOS reprojection
#####
##### A region spans several sinusoidal tiles, so every granule covering it is fetched and
##### the layers are mosaicked and warped to the region's latitude-longitude window in one
##### `gdalwarp` call each. The digital numbers are written through unchanged; masking,
##### quality screening, and scaling happen on read, where the product's decode rules live.
#####

# Locate the `HDF4_EOS:EOS_GRID:"…":<grid>:<layer>` subdataset without hardcoding the
# product's grid-group name.
function modis_subdataset(granule_path, layer)
    ArchGDAL.read(granule_path) do ds
        for entry in ArchGDAL.metadata(ds; domain = "SUBDATASETS")
            occursin("_NAME=", entry) || continue
            name = split(entry, "_NAME="; limit = 2)[2]
            endswith(name, ":" * layer) && return name
        end
        error("Layer $(layer) not found among the HDF-EOS subdatasets of $(granule_path).")
    end
end

# Mosaic one layer of `granule_paths` onto the regional lattice. Nearest-neighbor
# resampling throughout: the source and target cell sizes agree to within a percent, and
# every layer is either a bit-packed quality byte or a digital number whose out-of-range
# codes carry meaning — averaging either would invent values. Uncovered cells take the fill
# code 255, which the read path rejects on both the digital numbers and the quality bytes.
function warp_modis_layer(granule_paths, layer, lattice)
    sources = [ArchGDAL.read(modis_subdataset(path, layer)) for path in granule_paths]

    warped = try
        ArchGDAL.gdalwarp(sources,
            ["-t_srs", "EPSG:4326",
             "-te", string(lattice.west), string(lattice.south),
                    string(lattice.east), string(lattice.north),
             "-ts", string(lattice.Nx), string(lattice.Ny),
             "-r", "near",
             "-ot", "Byte",
             "-dstnodata", "255"]) do destination
            ArchGDAL.read(destination, 1)
        end
    finally
        foreach(ArchGDAL.destroy, sources)
    end

    # GDAL writes rows north→south; flip to the ascending latitude axis of the stored file.
    return reverse(warped, dims = 2)
end

function write_modis_netcdf(nc_path, layers, lattice)
    Δλ = (lattice.east - lattice.west) / lattice.Nx
    Δφ = (lattice.north - lattice.south) / lattice.Ny
    longitude = collect(range(lattice.west + Δλ / 2; step = Δλ, length = lattice.Nx))
    latitude  = collect(range(lattice.south + Δφ / 2; step = Δφ, length = lattice.Ny))

    # A staging name unique per writer, in the destination directory so the rename stays on one
    # filesystem. A shared `nc_path * ".tmp"` would let two processes materializing the same date
    # truncate each other's staging file and then rename a half-written result into place.
    staging_path = tempname(dirname(nc_path); cleanup = false) * ".nc"
    NCDataset(staging_path, "c") do ds
        defDim(ds, "lon", lattice.Nx)
        defDim(ds, "lat", lattice.Ny)
        defVar(ds, "lon", longitude, ("lon",);
               attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        defVar(ds, "lat", latitude, ("lat",);
               attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        for (layer, data) in pairs(layers)
            defVar(ds, String(layer), data, ("lon", "lat");
                   deflatelevel = 2, shuffle = true)
        end
    end
    mv(staging_path, nc_path; force = true)
    return nothing
end

function MODISLand.modis_granules_to_netcdf(metadatum::MODISLand.MODISLandMetadatum, nc_path)
    region = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("modis_granules_to_netcdf requires a bounded BoundingBox region.")

    lattice = regional_lattice(metadatum)
    urls = granule_urls(metadatum)

    mktempdir() do tmp
        @info string("Fetching ", length(urls), " MODIS granules for ", metadatum.dates, "...")
        granule_paths = [earthdata_download(url, joinpath(tmp, basename(url))) for url in urls]
        layers = Dict(layer => warp_modis_layer(granule_paths, layer, lattice)
                      for layer in stored_granule_layers(metadatum.dataset))
        write_modis_netcdf(nc_path, layers, lattice)
    end

    return nothing
end

end # module NumericalEarthArchGDALExt
