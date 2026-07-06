module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth

const CanopyHeight = NumericalEarth.DataWrangling.CanopyHeight

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

# Mosaic + window the intersecting COG tiles for `layer` ("Map"/"SD") onto the
# region bbox at the product's native resolution, returning the height array in
# (Nx, Ny) order with latitude increasing south→north.
function warp_canopy_layer(sources, longitude, latitude, resolution; resampling = "bilinear")
    λ₁, λ₂ = longitude
    φ₁, φ₂ = latitude
    return ArchGDAL.read(sources) do datasets
        ArchGDAL.gdalwarp(datasets,
            ["-t_srs", "EPSG:4326",
             "-te",    string(λ₁), string(φ₁), string(λ₂), string(φ₂),
             "-tr",    string(resolution), string(resolution),
             "-r",     resampling,
             "-ot",    "Float32"]) do warped
            # GDAL returns (Nx, Ny) with latitude north→south; flip to south→north.
            data = Float32.(ArchGDAL.read(warped, 1))
            return reverse(data, dims = 2)
        end
    end
end

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

# ETH: mosaic the intersecting 3° COG tiles for the Map (and SD) layers, mask the
# no-data byte to NaN (keeping non-forest zeros), and write one regional NetCDF.
function CanopyHeight.canopy_height_cog_to_netcdf(metadatum::CanopyHeight.ETHCanopyHeightMetadatum, nc_path)
    dataset = metadatum.dataset
    region  = metadatum.region
    resolution = 360 / Base.size(dataset, :canopy_height)[1]
    missing_value = 255

    layers = Dict{String, Matrix{Float32}}()
    for (variable, layer) in CanopyHeight.ETHCanopyHeight_variable_names
        sources = CanopyHeight.eth_tile_urls(region, layer)
        raw = warp_canopy_layer(sources, region.longitude, region.latitude, resolution)
        layers[layer] = CanopyHeight.mask_eth.(raw, missing_value)
    end

    write_canopy_netcdf(nc_path, region.longitude, region.latitude, layers)
    return nothing
end

# GLAD: window the forest-height mosaic, mask the categorical fill codes
# (101/102/103) to NaN *before* any averaging, keep non-forest zeros.
function CanopyHeight.canopy_height_cog_to_netcdf(metadatum::CanopyHeight.GLADCanopyHeightMetadatum, nc_path)
    dataset = metadatum.dataset
    region  = metadatum.region
    resolution = 360 / Base.size(dataset, :canopy_height)[1]

    sources = ["/vsicurl/" * CanopyHeight.GLAD_COG_HOST * ".tif"]
    # Nearest-neighbor read so the categorical fill codes (101/102/103) are never
    # blended before `mask_glad` converts them to NaN.
    raw = warp_canopy_layer(sources, region.longitude, region.latitude, resolution;
                            resampling = "near")
    layers = Dict("Map" => CanopyHeight.mask_glad.(raw))

    write_canopy_netcdf(nc_path, region.longitude, region.latitude, layers)
    return nothing
end

end # module NumericalEarthArchGDALExt
