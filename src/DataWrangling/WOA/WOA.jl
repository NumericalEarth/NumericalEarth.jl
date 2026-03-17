module WOA

export WOAClimatology, WOAAnnual, WOAMonthly

using NumericalEarth
using Oceananigans
using NCDatasets
using JLD2
using Scratch
using Adapt
using Dates

using ..DataWrangling:
    Metadata,
    Metadatum,
    BoundingBox,
    inpaint_mask!,
    NearestNeighborInpainting,
    download_progress

using Oceananigans.DistributedComputations: @root

import NumericalEarth.DataWrangling:
    all_dates,
    first_date,
    last_date,
    metadata_filename,
    download_dataset,
    default_download_directory,
    metadata_path,
    dataset_variable_name,
    metaprefix,
    z_interfaces,
    longitude_interfaces,
    latitude_interfaces,
    is_three_dimensional,
    reversed_vertical_axis,
    inpainted_metadata_path,
    available_variables,
    retrieve_data

import Oceananigans.Fields: location

download_WOA_cache::String = ""
function __init__()
    global download_WOA_cache = @get_scratch!("WOA")
end

WOA_variable_names = Dict(
    :temperature      => "t",
    :salinity         => "s",
    :phosphate        => "p",
    :nitrate          => "n",
    :silicate         => "i",
    :dissolved_oxygen => "o",
)

# Dataset types
abstract type WOAClimatology end

struct WOAAnnual <: WOAClimatology
    product_year :: Int
end

struct WOAMonthly <: WOAClimatology
    product_year :: Int
end

WOAAnnual(;  product_year=2023) = WOAAnnual(product_year)
WOAMonthly(; product_year=2023) = WOAMonthly(product_year)

function default_download_directory(::WOAAnnual)
    return mkpath(joinpath(download_WOA_cache, "annual"))
end

function default_download_directory(::WOAMonthly)
    return mkpath(joinpath(download_WOA_cache, "monthly"))
end

# Annual: single snapshot, no date
all_dates(::WOAAnnual, args...) = nothing
first_date(::WOAAnnual, args...) = nothing
last_date(::WOAAnnual, args...) = nothing

# Monthly: 12 climatological months (year is arbitrary, month matters)
all_dates(::WOAMonthly, args...) = [DateTime(2018, m, 1) for m in 1:12]

# WOA stores depth as positive values, surface first (0 to 5500m)
reversed_vertical_axis(::WOAClimatology) = true

longitude_interfaces(::WOAClimatology) = (-180, 180)
latitude_interfaces(::WOAClimatology) = (-90, 90)
available_variables(::WOAClimatology) = WOA_variable_names

"""
    woa_z_interfaces_from_centers(depth_centers)

Compute cell interfaces (negative z, bottom-first) from WOA standard
depth centers (positive, surface-first).
"""
function woa_z_interfaces_from_centers(depth_centers)
    N = length(depth_centers)

    # Compute cell faces (positive depth, surface-first)
    faces = Vector{Float64}(undef, N + 1)
    faces[1] = 0.0 # surface
    for k in 1:N-1
        faces[k+1] = (depth_centers[k] + depth_centers[k+1]) / 2
    end
    # Bottom face: extrapolate below deepest center
    faces[N+1] = depth_centers[N] + (depth_centers[N] - depth_centers[N-1]) / 2

    # Convert to negative z, bottom-first
    return [-faces[N + 2 - k] for k in 1:N+1]
end

# Read size and z_interfaces from the actual WOA NetCDF file.
# This is necessary because the number of depth levels varies:
# annual climatologies have 102 levels, monthly have 57.
function Base.size(metadata::Metadata{<:WOAClimatology})
    path = metadata_path(first(metadata))
    ds = Dataset(path)
    Nlon = length(ds["lon"])
    Nlat = length(ds["lat"])
    Nz = length(ds["depth"])
    close(ds)

    Nt = metadata.dates isa AbstractArray ? length(metadata.dates) : 1
    return (Nlon, Nlat, Nz, Nt)
end

function z_interfaces(metadata::Metadata{<:WOAClimatology})
    path = metadata_path(first(metadata))
    ds = Dataset(path)
    depth_centers = Float64.(ds["depth"][:])
    close(ds)
    return woa_z_interfaces_from_centers(depth_centers)
end

# Type aliases
const WOAMetadata{D} = Metadata{<:WOAClimatology, D}
const WOAMetadatum   = Metadatum{<:WOAClimatology}

metaprefix(::WOAMetadata) = "WOAMetadata"
metaprefix(::WOAMetadatum) = "WOAMetadatum"

# Map from date to WOA period number (used by extension for download)
woa_period(::WOAAnnual, date) = 0
woa_period(::WOAMonthly, date) = Dates.month(date)

function metadata_filename(::WOAAnnual, name, date, bounding_box)
    varname = WOA_variable_names[name]
    return "woa_$(varname)_annual.nc"
end

function metadata_filename(::WOAMonthly, name, date, bounding_box)
    varname = WOA_variable_names[name]
    m = lpad(Dates.month(date), 2, '0')
    return "woa_$(varname)_monthly_$(m).nc"
end

# WOA NetCDF variables are named "{tracer}_an" for the objectively analyzed field
dataset_variable_name(data::WOAMetadata) = WOA_variable_names[data.name] * "_an"

location(::WOAMetadata) = (Center, Center, Center)
is_three_dimensional(::WOAMetadata) = true

function inpainted_metadata_filename(metadata::WOAMetadatum)
    without_extension = metadata.filename[1:end-3]
    var = string(metadata.name)
    return without_extension * "_" * var * "_inpainted.jld2"
end

inpainted_metadata_path(metadata::WOAMetadatum) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

# Custom retrieve_data: WOA NetCDF files contain Missing values (from _FillValue)
# which must be converted to NaN before the GPU kernel in set_metadata_field!.
function retrieve_data(metadata::Metadatum{<:WOAClimatology})
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)

    ds = Dataset(path)
    raw = ds[name][:, :, :, 1]
    close(ds)

    # Convert Union{Missing, Float32} → Float32 with NaN for missing
    data = Array{Float32}(undef, size(raw))
    for i in eachindex(raw)
        data[i] = ismissing(raw[i]) ? NaN32 : Float32(raw[i])
    end

    if reversed_vertical_axis(metadata.dataset)
        data = reverse(data, dims=3)
    end

    return data
end

end # module
