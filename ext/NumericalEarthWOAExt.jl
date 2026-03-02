module NumericalEarthWOAExt

using WorldOceanAtlasTools
using NCDatasets

import NumericalEarth.DataWrangling: download_dataset, metadata_path
using NumericalEarth.DataWrangling: Metadata, Metadatum
using NumericalEarth.WOA: WOAClimatology, WOA_variable_names, woa_period

using Oceananigans.DistributedComputations: @root

function download_dataset(metadata::Metadata{<:WOAClimatology}; skip_existing=true)
    @root for metadatum in metadata
        filepath = metadata_path(metadatum)

        if isfile(filepath) && skip_existing
            continue
        end

        woa_tracer = WOA_variable_names[metadatum.name]
        period = woa_period(metadatum.dataset, metadatum.dates)

        # Open WOA dataset to get coordinates and data directly
        woa_ds = WorldOceanAtlasTools.WOA_Dataset(woa_tracer;
                                                  product_year=2018,
                                                  period,
                                                  resolution=1)

        lon   = Float64.(woa_ds["lon"][:])
        lat   = Float64.(woa_ds["lat"][:])
        depth = Float64.(woa_ds["depth"][:])

        # Read the objectively analyzed field directly from WOA NetCDF
        # WOA NetCDF stores data as (lon, lat, depth, time)
        woa_varname = WorldOceanAtlasTools.WOA_varname(woa_tracer, "an")
        raw_data = woa_ds[woa_varname][:, :, :, 1]  # (lon, lat, depth)

        close(woa_ds)

        # Convert Missing to NaN for NetCDF storage
        Nlon, Nlat, Ndepth = size(raw_data)
        data = Array{Float32}(undef, Nlon, Nlat, Ndepth)
        for i in eachindex(raw_data)
            data[i] = ismissing(raw_data[i]) ? NaN32 : Float32(raw_data[i])
        end

        # Save as NetCDF for the core module to read
        varname = string(metadatum.name)
        ds = NCDataset(filepath, "c")

        defDim(ds, "lon", Nlon)
        defDim(ds, "lat", Nlat)
        defDim(ds, "depth", Ndepth)

        ds_lon = defVar(ds, "lon", Float64, ("lon",))
        ds_lat = defVar(ds, "lat", Float64, ("lat",))
        ds_dep = defVar(ds, "depth", Float64, ("depth",))

        ds_lon[:] = lon
        ds_lat[:] = lat
        ds_dep[:] = depth

        ds_var = defVar(ds, varname, Float32, ("lon", "lat", "depth"))
        ds_var[:, :, :] = data

        close(ds)
    end

    return nothing
end

end # module
