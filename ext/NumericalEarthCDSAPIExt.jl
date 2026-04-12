module NumericalEarthCDSAPIExt

using NumericalEarth
using CDSAPI

using Oceananigans.DistributedComputations: @root

using Dates
using NumericalEarth.DataWrangling.ERA5: ERA5Metadata, ERA5Metadatum, ERA5_dataset_variable_names

import NumericalEarth.DataWrangling: download_dataset

"""
    download_dataset(metadata::ERA5Metadata; kwargs...)

Download ERA5 data for each date in the metadata, returning paths to downloaded files.
Downloads all dates for the variable in a single CDS API request when possible.
"""
function download_dataset(metadata::ERA5Metadata; skip_existing=true)
    # Collect all metadatums and check which files are missing
    all_meta = [m for m in metadata]
    paths = [joinpath(m.dir, m.filename) for m in all_meta]

    missing_indices = findall(i -> !isfile(paths[i]), eachindex(paths))
    isempty(missing_indices) && return paths

    missing_meta = all_meta[missing_indices]
    mkpath(first(missing_meta).dir)

    # Group by unique (year, month, day) to batch hours within each day
    days = unique(Dates.Date.(m.dates for m in missing_meta))
    hours = unique(lpad(string(Dates.hour(m.dates)), 2, '0') * ":00" for m in missing_meta)
    years = unique(string(Dates.year(d)) for d in days)
    months = unique(lpad(string(Dates.month(d)), 2, '0') for d in days)
    day_strs = unique(lpad(string(Dates.day(d)), 2, '0') for d in days)

    variable_name = ERA5_dataset_variable_names[metadata.name]
    region = metadata.region

    # Single CDS request for all dates
    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => [variable_name],
        "year"            => collect(years),
        "month"           => collect(months),
        "day"             => collect(day_strs),
        "time"            => collect(hours),
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    area = build_era5_area(region)
    if !isnothing(area)
        request["area"] = area
    end

    # Download multi-time file, then split into per-time files
    dir = first(missing_meta).dir
    batch_file = joinpath(dir, "era5_batch_$(variable_name).nc")

    @root CDSAPI.retrieve("reanalysis-era5-single-levels", request, batch_file)

    # Split into individual files per time step
    _split_era5_batch(batch_file, missing_meta)
    rm(batch_file; force=true)

    return paths
end

using NCDatasets

function _split_era5_batch(batch_file, metadatums)
    ds = NCDataset(batch_file)

    # Read the time coordinate
    times = ds["valid_time"][:]

    for m in metadatums
        output_path = joinpath(m.dir, m.filename)
        isfile(output_path) && continue

        # Find the time index for this metadatum
        target_time = m.dates
        tidx = findfirst(t -> Dates.DateTime(t) == target_time, times)
        isnothing(tidx) && continue

        varname = ERA5_dataset_variable_names[m.name]

        NCDataset(output_path, "c") do out
            # Copy spatial dimensions
            for dimname in ("longitude", "latitude")
                if haskey(ds, dimname)
                    src = ds[dimname]
                    defDim(out, dimname, length(src))
                    defVar(out, dimname, Array(src), (dimname,); attrib=src.attrib)
                end
            end
            # Single time dimension
            defDim(out, "valid_time", 1)

            # Copy the variable at the target time
            if haskey(ds, varname)
                src = ds[varname]
                dims = NCDatasets.dimnames(src)
                spatial_dims = filter(d -> d != "valid_time", dims)
                out_dims = (spatial_dims..., "valid_time")

                # Select the time slice
                data = if ndims(src) == 3  # lon, lat, time
                    src[:, :, tidx:tidx]
                elseif ndims(src) == 2  # lat, time or lon, time
                    src[:, tidx:tidx]
                else
                    src[tidx:tidx]
                end

                defVar(out, varname, data, out_dims; attrib=src.attrib)
            end
        end
    end

    close(ds)
end

"""
    download_dataset(metadata_list::AbstractVector{<:ERA5Metadata})

Download ERA5 data for multiple Metadata objects.
"""
function download_dataset(metadata_list::AbstractVector{<:ERA5Metadata})
    for metadata in metadata_list
        download_dataset(metadata)
    end
end

"""
    download_dataset(meta::ERA5Metadatum; skip_existing=true, kwargs...)

Download ERA5 data for a single date/time using the CDSAPI package.

# Keyword Arguments
- `skip_existing`: Skip download if the file already exists (default: `true`).

# Environment Setup
Before downloading, you must:
1. Create an account at https://cds.climate.copernicus.eu/
2. Accept the Terms of Use for the ERA5 dataset on the dataset page
3. Set up your API credentials in `~/.cdsapirc`

See https://cds.climate.copernicus.eu/how-to-api for details.
"""
function download_dataset(meta::ERA5Metadatum; skip_existing=true)

    output_directory = meta.dir
    output_filename = meta.filename
    output_path = joinpath(output_directory, output_filename)

    # Skip if file already exists
    if skip_existing && isfile(output_path)
        return output_path
    end

    # Ensure output directory exists
    mkpath(output_directory)

    # Get the ERA5 variable name
    variable_name = ERA5_dataset_variable_names[meta.name]

    # Extract date information
    date = meta.dates
    year  = string(Dates.year(date))
    month = lpad(string(Dates.month(date)), 2, '0')
    day   = lpad(string(Dates.day(date)), 2, '0')
    hour  = lpad(string(Dates.hour(date)), 2, '0') * ":00"

    # Build request parameters
    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => [variable_name],
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => [hour],
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    # Add area constraint from region
    area = build_era5_area(meta.region)
    if !isnothing(area)
        request["area"] = area
    end

    # Perform the download using CDSAPI
    @root begin
        CDSAPI.retrieve("reanalysis-era5-single-levels", request, output_path)
    end

    return output_path
end

#####
##### Area/region utilities
#####

build_era5_area(::Nothing) = nothing

const BBOX = NumericalEarth.DataWrangling.BoundingBox
const COL  = NumericalEarth.DataWrangling.Column

function build_era5_area(bbox::BBOX)
    # CDS API uses [north, west, south, east] ordering
    # BoundingBox has longitude = (west, east), latitude = (south, north)

    lon = bbox.longitude
    lat = bbox.latitude

    if isnothing(lon) || isnothing(lat)
        return nothing
    end

    west  = lon[1]
    east  = lon[2]
    south = lat[1]
    north = lat[2]

    return [north, west, south, east]
end

function build_era5_area(col::COL)
    # ERA5 is 0.25°; expand by 0.5° (2 grid cells) for interpolation
    ε = 0.5
    lon, lat = col.longitude, col.latitude
    return [lat + ε, lon - ε, lat - ε, lon + ε]  # [N, W, S, E]
end

end # module NumericalEarthCDSAPIExt
