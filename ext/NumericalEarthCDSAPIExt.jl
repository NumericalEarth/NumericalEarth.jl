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
"""
function download_dataset(metadata::ERA5Metadata; kwargs...)
    paths = Array{String}(undef, length(metadata))
    for (m, metadatum) in enumerate(metadata)
        paths[m] = download_dataset(metadatum; kwargs...)
    end
    return paths
end

using NCDatasets

"""
    download_dataset(metadata_list::AbstractVector{<:ERA5Metadata})

Batch-download ERA5 data for multiple variables in a single CDS API request
per date, then split the result into per-variable files. This avoids making
N_variables separate API calls (each of which queues independently).
"""
function download_dataset(metadata_list::AbstractVector{<:ERA5Metadata})
    # Collect all unique dates across all metadata
    all_metadatums = [m for metadata in metadata_list for m in metadata]

    # Check which files already exist
    missing = filter(m -> !isfile(joinpath(m.dir, m.filename)), all_metadatums)
    isempty(missing) && return

    # Group missing metadatums by date (to batch variables per date)
    by_date = Dict{Any, Vector{eltype(missing)}}()
    for m in missing
        d = m.dates
        if !haskey(by_date, d)
            by_date[d] = eltype(missing)[]
        end
        push!(by_date[d], m)
    end

    for (date, metadatums) in by_date
        # All metadatums share the same date and region
        region = first(metadatums).region
        variable_names = [ERA5_dataset_variable_names[m.name] for m in metadatums]

        year  = string(Dates.year(date))
        month = lpad(string(Dates.month(date)), 2, '0')
        day   = lpad(string(Dates.day(date)), 2, '0')
        hour  = lpad(string(Dates.hour(date)), 2, '0') * ":00"

        request = Dict(
            "product_type"    => ["reanalysis"],
            "variable"        => variable_names,
            "year"            => [year],
            "month"           => [month],
            "day"             => [day],
            "time"            => [hour],
            "data_format"     => "netcdf",
            "download_format" => "unarchived",
        )

        area = build_era5_area(region)
        if !isnothing(area)
            request["area"] = area
        end

        # Download to a temp file, then split into per-variable files
        dir = first(metadatums).dir
        mkpath(dir)
        batch_path = joinpath(dir, "era5_batch_$(year)$(month)$(day)_$(hour[1:2]).nc")

        @root CDSAPI.retrieve("reanalysis-era5-single-levels", request, batch_path)

        # Split the multi-variable file into individual files
        ds = NCDataset(batch_path)
        for m in metadatums
            varname = ERA5_dataset_variable_names[m.name]
            output_path = joinpath(m.dir, m.filename)
            isfile(output_path) && continue

            NCDataset(output_path, "c") do out
                # Copy dimensions
                for (dimname, dim) in ds.dim
                    defDim(out, dimname, length(dim))
                end
                # Copy coordinate variables
                for dimname in keys(ds.dim)
                    if haskey(ds, dimname)
                        src = ds[dimname]
                        defVar(out, dimname, Array(src), (dimname,); attrib=src.attrib)
                    end
                end
                # Copy the target variable
                if haskey(ds, varname)
                    src = ds[varname]
                    defVar(out, varname, Array(src), NCDatasets.dimnames(src); attrib=src.attrib)
                end
            end
        end
        close(ds)
        rm(batch_path; force=true)
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
