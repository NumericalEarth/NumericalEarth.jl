module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth

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
##### Land surface temperature reads (gated behind ArchGDAL)
#####

# GOES-R ABI-L2-LSTC is stored on the geostationary fixed grid (scan-angle x/y
# radians + `goes_imager_projection`). GDAL understands the `+proj=geos` SRS from
# the netCDF metadata, so we warp the `LST` and `DQF` subdatasets to EPSG:4326,
# clip to the requested BoundingBox, then apply the pure `goes_lst` decode
# (scale/offset + fill/valid-range + DQF masking). Returns a (Nx, Ny) Float32
# array of Kelvin (NaN in the cloud/no-retrieval gaps).
const LST = NumericalEarth.DataWrangling.LandSurfaceTemperature

function LST.read_goes_lst_lonlat(metadatum, path)
    region = metadatum.region
    λ = region.longitude
    φ = region.latitude

    warp(subdataset) = ArchGDAL.read(subdataset) do src
        ArchGDAL.gdalwarp([src],
            ["-t_srs", "EPSG:4326",
             "-te",    string(λ[1]), string(φ[1]), string(λ[2]), string(φ[2]),
             "-r",     "near"]) do w
            ArchGDAL.read(w, 1)
        end
    end

    dn  = warp("NETCDF:\"$path\":LST")
    dqf = warp("NETCDF:\"$path\":DQF")

    K = LST.goes_lst.(round.(Int, dn), round.(Int, dqf))
    return Float32.(reverse(K, dims = 2)) # GDAL returns north→south; flip to south→north
end

# ECOSTRESS ECO_L2G_LSTE is HDF5 on a plain lat/lon grid. HDF5.jl is not a
# project dependency (AGENTS.md rule 10), so we route through GDAL's HDF5 driver
# and apply the pure `ecostress_lst` decode (float32 K passthrough, cloud → NaN).
function LST.read_ecostress_l2g(metadatum, path)
    read_band(field) = ArchGDAL.read("HDF5:\"$path\"://$field") do ds
        ArchGDAL.read(ds, 1)
    end

    lst   = read_band("LST")
    cloud = read_band("cloud")

    K = LST.ecostress_lst.(Float32.(lst), round.(Int, cloud))
    return Float32.(reverse(K, dims = 2))
end

end # module NumericalEarthArchGDALExt
