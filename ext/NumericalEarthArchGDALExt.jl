module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox
using NumericalEarth.DataWrangling.ECOSTRESS: ECOSTRESS, ecostress_lst,
                                              ecostress_cmr_granules, earthdata_download,
                                              ECOSTRESS_RESOLUTION

using Dates: Dates, Day, Second

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
##### ECOSTRESS ECO_L2G_LSTE — Earthdata CMR discovery + GDAL HDF5 read
#####
##### Fetch the overpass nearest the requested date, clip its (already EPSG:4326)
##### LST / LST_err / cloud subdatasets to the region, decode via the pure
##### `ecostress_lst` core, and write a clean regional lon/lat NetCDF so the generic
##### `Field` / `set_region_data!` machinery brackets it onto the native grid.
#####

# Write a regional lon/lat NetCDF of decoded Kelvin layers. Each `layers` entry maps
# a NetCDF variable name to a south→north-ordered `(Nx, Ny)` array; `gt` is the
# (north→south) GDAL geotransform of the clip.
function write_ecostress_netcdf(nc_path, layers, gt)
    first_layer = first(values(layers))
    Nx, Ny = size(first_layer)
    Δφ = gt[6]  # negative (north→south)
    longitude = collect(range(gt[1] + gt[2] / 2; step = gt[2], length = Nx))
    latitude  = collect(range(gt[4] + Δφ / 2; step = Δφ, length = Ny))
    reverse!(latitude)  # match the reversed (south→north) data

    staging_path = nc_path * ".tmp"
    NCDataset(staging_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        lon_var[:] = longitude
        lat_var[:] = latitude
        for (name, data) in pairs(layers)
            var = defVar(ds, String(name), Float32, ("lon", "lat");
                         attrib = ["units" => "K", "long_name" => "land surface temperature"],
                         deflatelevel = 2, shuffle = true)
            var[:, :] = data
        end
    end
    mv(staging_path, nc_path; force = true)
    return nothing
end

# Locate the `HDF5:"…"://…/<field>` subdataset without hardcoding the per-resolution
# grid group name.
function ecostress_subdataset(h5_path, field)
    ArchGDAL.read(h5_path) do ds
        for entry in ArchGDAL.metadata(ds; domain = "SUBDATASETS")
            occursin("_NAME=", entry) || continue
            name = split(entry, "_NAME="; limit = 2)[2]
            endswith(name, "/" * field) && return name
        end
        error("Field $(field) not found among the ECO_L2G_LSTE HDF5 subdatasets of $(h5_path).")
    end
end

function ECOSTRESS.ecostress_granule_to_netcdf(metadatum::ECOSTRESS.ECOSTRESSMetadatum, nc_path)
    region = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("ecostress_granule_to_netcdf requires a bounded BoundingBox region.")

    version = metadatum.dataset.version
    date = metadatum.dates

    # Search a ±1-day window around the requested overpass. Prefer granules that actually
    # cover the region (CMR's spatial search over-reports); among those, pick the nearest
    # in time. Fall back to all candidates only if none clear the coverage threshold.
    granules = ecostress_cmr_granules(version, region, date - Day(1), date + Day(1))
    isempty(granules) &&
        error("CMR returned no ECO_L2G_LSTE.$(version) granules for region $(region) near $(date).")
    covered = filter(g -> g.coverage ≥ ECOSTRESS.ECOSTRESS_MIN_COVERAGE, granules)
    candidates = isempty(covered) ? granules : covered
    gaps = [abs(Dates.value(Second(g.time - date))) for g in candidates]
    url = candidates[argmin(gaps)].url

    λ = region.longitude
    φ = region.latitude
    Δ = ECOSTRESS_RESOLUTION

    mktempdir() do tmp
        h5 = joinpath(tmp, "ecostress.h5")
        earthdata_download(url, h5)

        # Nearest-neighbour resampling so the 0 fill (off-swath / no-retrieval) is
        # never blended into a spurious intermediate temperature; source ≈ target Δ.
        clip(field) = ArchGDAL.read(ecostress_subdataset(h5, field)) do src
            ArchGDAL.gdalwarp([src],
                ["-t_srs", "EPSG:4326",
                 "-te", string(λ[1]), string(φ[1]), string(λ[2]), string(φ[2]),
                 "-tr", string(Δ), string(Δ),
                 "-r", "near"]) do w
                (ArchGDAL.read(w, 1), ArchGDAL.getgeotransform(w))
            end
        end

        lst,   gt = clip("LST")
        err,   _  = clip("LST_err")
        cloud, _  = clip("cloud")

        cloud_i = round.(Int, cloud)
        # Decode LST (cloud/fill → NaN); mask LST_err wherever LST is invalid.
        LST     = ecostress_lst.(Float32.(lst), cloud_i)
        LST_err = ifelse.(isnan.(LST) .| .!(Float32.(err) .> 0), NaN32, Float32.(err))

        # Flip GDAL's north→south rows to south→north to match the NetCDF lat axis.
        layers = (LST = reverse(LST, dims = 2), LST_err = reverse(LST_err, dims = 2))
        write_ecostress_netcdf(nc_path, layers, gt)
    end
    return nothing
end

end # module NumericalEarthArchGDALExt
