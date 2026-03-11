module ERA5

# 2-D data
export ERA5HourlySingleLevel, ERA5MonthlySingleLevel

# 3-D data
export ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels, ERA5_all_pressure_levels, pressure_field, hPa
export standard_atmosphere_z_interfaces, mean_geopotential_z_interfaces

using NCDatasets
using Printf
using Scratch
using Statistics

using Oceananigans.Fields: Center
using Oceananigans: CenterField, interior, fill_halo_regions!, CPU
using NumericalEarth.DataWrangling: Metadata, Metadatum, metadata_path, native_grid, InverseGravity, download_dataset
using Dates
using Dates: DateTime, Day, Month, Hour

import Oceananigans.Fields: location

import NumericalEarth.DataWrangling:
    all_dates,
    dataset_variable_name,
    default_download_directory,
    default_inpainting,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    metadata_filename,
    inpainted_metadata_path,
    available_variables,
    retrieve_data,
    metadata_path,
    is_three_dimensional,
    reversed_vertical_axis,
    conversion_units

import Base: eltype

download_ERA5_cache::String = ""

function __init__()
    global download_ERA5_cache = @get_scratch!("ERA5")
end

#####
##### ERA5 Datasets
#####

abstract type ERA5Dataset end

default_download_directory(::ERA5Dataset) = download_ERA5_cache

const ERA5Metadata{D} = Metadata{<:ERA5Dataset, D}
const ERA5Metadatum = Metadatum{<:ERA5Dataset}

#####
##### Grid interfaces
#####

location(::ERA5Metadata) = (Center, Center, Center)

# ERA5 global coverage: 0-359.75 longitude, -90 to 90 latitude at 0.25 degree resolution
longitude_interfaces(::ERA5Metadata) = (-0.125, 359.875)
latitude_interfaces(::ERA5Metadata) = (-90, 90)

# ERA5 single-levels (2-D) data product
z_interfaces(::ERA5Metadata) = (0, 1)

# ERA5 data is stored as Float32
eltype(::ERA5Metadata) = Float32

#####
##### Shared filename utilities
#####

function date_str(date::DateTime)
    y = Dates.year(date)
    m = lpad(Dates.month(date), 2, '0')
    d = lpad(Dates.day(date),   2, '0')
    h = lpad(Dates.hour(date),  2, '0')
    return "$(y)-$(m)-$(d)T$(h)"
end

start_date_str(date::DateTime) = date_str(date)
end_date_str(date::DateTime) = date_str(date)
start_date_str(dates::AbstractVector) = date_str(first(dates))
end_date_str(dates::AbstractVector) = date_str(last(dates))

colon2dash(s::String) = replace(s, ":" => "-")
underscore_spaces(s::String) = replace(s, " " => "_")

function bbox_strs(::Nothing)
    return "_nothing", "_nothing"
end

function bbox_strs(c)
    first = @sprintf("_%.1f", c[1])
    second = @sprintf("_%.1f", c[2])
    return first, second
end

function metadata_filename(metadata::ERA5Metadatum)
    prefix = metadata_prefix(metadata)
    return string(prefix, ".nc")
end

function metadata_filename(metadata::ERA5Metadata)
    return [metadata_filename(metadatum) for metadatum in metadata]
end

function inpainted_metadata_filename(metadata::ERA5Metadata)
    original_filename = metadata_filename(metadata)
    without_extension = original_filename[1:end-3]
    return without_extension * "_inpainted.jld2"
end

inpainted_metadata_path(metadata::ERA5Metadata) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

#####
##### Single-level and pressure-level specifics
#####

include("ERA5_single_levels.jl")
include("ERA5_pressure_levels.jl")

end # module ERA5
