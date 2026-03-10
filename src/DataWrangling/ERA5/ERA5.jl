module ERA5

# 2-D data
export ERA5SingleLevelsHourly, ERA5SingleLevelsMonthly

# 3-D data
export ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels, ERA5_all_pressure_levels, pressure_field, hPa
export standard_atmosphere_z_interfaces, mean_geopotential_z_interfaces

using NCDatasets
using Printf
using Scratch
using Statistics

using Oceananigans.Fields: Center
using Oceananigans: CenterField, interior, fill_halo_regions!, CPU
using NumericalEarth.DataWrangling: Metadata, Metadatum, metadata_path, native_grid, InverseGravity
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

struct ERA5SingleLevelsHourly <: ERA5Dataset end
struct ERA5SingleLevelsMonthly <: ERA5Dataset end

dataset_name(::ERA5SingleLevelsHourly)  = "ERA5SingleLevelsHourly"
dataset_name(::ERA5SingleLevelsMonthly) = "ERA5SingleLevelsMonthly"

#####
##### ERA5 pressure-level datasets
#####

const hPa = 100.0
const ERA5_all_pressure_levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150,
    175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000]*hPa

abstract type ERA5PressureDataset <: ERA5Dataset end

struct ERA5HourlyPressureLevels <: ERA5PressureDataset
    levels :: Vector{Int}
end
ERA5HourlyPressureLevels(; levels=ERA5_all_pressure_levels) = ERA5HourlyPressureLevels(levels)

struct ERA5MonthlyPressureLevels <: ERA5PressureDataset
    levels :: Vector{Int}
end
ERA5MonthlyPressureLevels(; levels=ERA5_all_pressure_levels) = ERA5MonthlyPressureLevels(levels)

dataset_name(::ERA5HourlyPressureLevels)  = "ERA5HourlyPressureLevels"
dataset_name(::ERA5MonthlyPressureLevels) = "ERA5MonthlyPressureLevels"

Base.size(ds::ERA5PressureDataset, variable) = (1440, 720, length(ds.levels))

all_dates(::ERA5HourlyPressureLevels,  var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-31"), step=Hour(1))
all_dates(::ERA5MonthlyPressureLevels, var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-01"), step=Month(1))

const ERA5PressureMetadata{D} = Metadata{<:ERA5PressureDataset, D}
const ERA5PressureMetadatum   = Metadatum{<:ERA5PressureDataset}

# Wave variables are on a 0.5° grid (720×361), atmospheric variables on 0.25° (1440×721)
const ERA5_wave_variables = Set([
    :eastward_stokes_drift, :northward_stokes_drift,
    :significant_wave_height, :mean_wave_period, :mean_wave_direction,
])

function Base.size(::ERA5Dataset, variable)
    if variable in ERA5_wave_variables
        return (720, 361, 1)
    else
        return (1440, 721, 1)
    end
end

# ERA5 reanalysis data available from 1940 to present (we use a practical range here)
all_dates(::ERA5SingleLevelsHourly,  var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-31"), step=Hour(1))
all_dates(::ERA5SingleLevelsMonthly, var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-01"), step=Month(1))

const ERA5Metadata{D} = Metadata{<:ERA5Dataset, D}
const ERA5Metadatum = Metadatum{<:ERA5Dataset}

# ERA5 is a spatially 2D dataset (atmospheric surface variables)
is_three_dimensional(::ERA5Metadata) = false

# ERA5 pressure-level data is 3D
is_three_dimensional(::ERA5PressureMetadata) = true

# ERA5 stores pressure levels bottom-to-top
reversed_vertical_axis(::ERA5PressureDataset) = false

# Variable name mappings from NumericalEarth names to ERA5/CDS API variable names
ERA5_dataset_variable_names = Dict(
    :temperature                     => "2m_temperature",
    :dewpoint_temperature            => "2m_dewpoint_temperature",
    :eastward_velocity               => "10m_u_component_of_wind",
    :northward_velocity              => "10m_v_component_of_wind",
    :surface_pressure                => "surface_pressure",
    :mean_sea_level_pressure         => "mean_sea_level_pressure",
    :total_precipitation             => "total_precipitation",
    :sea_surface_temperature         => "sea_surface_temperature",
    :downwelling_shortwave_radiation => "surface_solar_radiation_downwards",
    :downwelling_longwave_radiation  => "surface_thermal_radiation_downwards",
    :total_cloud_cover               => "total_cloud_cover",
    :evaporation                     => "evaporation",
    :specific_humidity               => "specific_humidity",
    :eastward_stokes_drift           => "u_component_stokes_drift",
    :northward_stokes_drift          => "v_component_stokes_drift",
    :significant_wave_height         => "significant_height_of_combined_wind_waves_and_swell",
    :mean_wave_period                => "mean_wave_period",
    :mean_wave_direction             => "mean_wave_direction",
)

# Variables available for download
ERA5_variable_names = keys(ERA5_dataset_variable_names)

available_variables(::ERA5Dataset) = ERA5_dataset_variable_names

dataset_variable_name(metadata::ERA5Metadata) = ERA5_dataset_variable_names[metadata.name]

# NetCDF short variable names (what's actually in the downloaded files)
# These differ from the CDS API variable names above
ERA5_netcdf_variable_names = Dict(
    :temperature                     => "t2m",
    :dewpoint_temperature            => "d2m",
    :eastward_velocity               => "u10",
    :northward_velocity              => "v10",
    :surface_pressure                => "sp",
    :mean_sea_level_pressure         => "msl",
    :total_precipitation             => "tp",
    :sea_surface_temperature         => "sst",
    :downwelling_shortwave_radiation => "ssrd",
    :downwelling_longwave_radiation  => "strd",
    :total_cloud_cover               => "tcc",
    :evaporation                     => "e",
    :specific_humidity               => "q",
    :eastward_stokes_drift           => "ust",
    :northward_stokes_drift          => "vst",
    :significant_wave_height         => "swh",
    :mean_wave_period                => "mwp",
    :mean_wave_direction             => "mwd",
)

netcdf_variable_name(metadata::ERA5Metadata) = ERA5_netcdf_variable_names[metadata.name]

#####
##### ERA5 pressure-level variable name mappings
#####

ERA5PL_dataset_variable_names = Dict(
    :temperature                         => "temperature",
    :eastward_velocity                   => "u_component_of_wind",
    :northward_velocity                  => "v_component_of_wind",
    :vertical_velocity                   => "vertical_velocity",
    :geopotential                        => "geopotential",
    :geopotential_height                 => "geopotential",
    :specific_humidity                   => "specific_humidity",
    :relative_humidity                   => "relative_humidity",
    :vorticity                           => "vorticity",
    :divergence                          => "divergence",
    :potential_vorticity                 => "potential_vorticity",
    :ozone_mass_mixing_ratio             => "ozone_mass_mixing_ratio",
    :fraction_of_cloud_cover             => "fraction_of_cloud_cover",
    :specific_cloud_liquid_water_content => "specific_cloud_liquid_water_content",
    :specific_cloud_ice_water_content    => "specific_cloud_ice_water_content",
)

ERA5PL_netcdf_variable_names = Dict(
    :temperature                         => "t",
    :eastward_velocity                   => "u",
    :northward_velocity                  => "v",
    :vertical_velocity                   => "w",
    :geopotential                        => "z",
    :geopotential_height                 => "z",
    :specific_humidity                   => "q",
    :relative_humidity                   => "r",
    :vorticity                           => "vo",
    :divergence                          => "d",
    :potential_vorticity                 => "pv",
    :ozone_mass_mixing_ratio             => "o3",
    :fraction_of_cloud_cover             => "cc",
    :specific_cloud_liquid_water_content => "clwc",
    :specific_cloud_ice_water_content    => "ciwc",
)

available_variables(::ERA5PressureDataset) = ERA5PL_dataset_variable_names
dataset_variable_name(md::ERA5PressureMetadata) = ERA5PL_dataset_variable_names[md.name]
netcdf_variable_name(md::ERA5PressureMetadata)  = ERA5PL_netcdf_variable_names[md.name]

conversion_units(md::ERA5PressureMetadatum) =
    md.name == :geopotential_height ? InverseGravity() : nothing

default_inpainting(md::ERA5Metadatum) = nothing
default_inpainting(md::ERA5PressureMetadatum) = nothing

"""
    retrieve_data(metadata::ERA5Metadatum)

Retrieve ERA5 data from NetCDF file according to `metadata`.
ERA5 is 2D surface data, so we return a 2D array with an added singleton z-dimension.
"""
function retrieve_data(metadata::ERA5Metadatum)
    path = metadata_path(metadata)
    name = netcdf_variable_name(metadata)
    
    ds = NCDatasets.Dataset(path)
    
    # ERA5 is 2D + time, we take the first time step
    # Data shape is typically (lon, lat) or (lon, lat, time)
    raw_data = ds[name]
    ndim = ndims(raw_data)
    
    if ndim == 2
        data_2d = raw_data[:, :]
    elseif ndim == 3
        data_2d = raw_data[:, :, 1]
    else
        error("Unexpected ERA5 data dimensions: $ndim")
    end
    
    close(ds)

    # Latitude is stored from 90°N → 90°S
    data_2d = reverse(data_2d, dims=2)
    
    # Add singleton z-dimension for 3D field compatibility
    # Return as (Nx, Ny, 1)
    return reshape(data_2d, size(data_2d, 1), size(data_2d, 2), 1)
end

"""
    retrieve_data(metadata::ERA5PressureMetadatum)

Retrieve ERA5 pressure-level data from a NetCDF file.
Returns a 3D array (lon, lat, level) with levels ordered bottom-to-top
(highest pressure at k=1, lowest pressure at k=Nz).
"""
function retrieve_data(metadata::ERA5PressureMetadatum)
    path = metadata_path(metadata)
    name = netcdf_variable_name(metadata)
    ds   = NCDatasets.Dataset(path)
    data = ds[name][:, :, :, 1]   # (lon, lat, pressure_level, time=1)
    close(ds)
    return reverse(data, dims=2)  # Latitude is stored from 90°N → 90°S
end

#####
##### Metadata filename construction
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

function metadata_prefix(metadata::ERA5PressureMetadata)
    var = ERA5PL_dataset_variable_names[metadata.name]
    dataset = dataset_name(metadata.dataset)
    start_date = start_date_str(metadata.dates)
    end_date = end_date_str(metadata.dates)
    bbox = metadata.bounding_box

    if !isnothing(bbox)
        w, e = bbox_strs(bbox.longitude)
        s, n = bbox_strs(bbox.latitude)
        suffix = string(w, e, s, n)
    else
        suffix = ""
    end

    if start_date == end_date
        prefix = string(var, "_", dataset, "_", start_date, suffix)
    else
        prefix = string(var, "_", dataset, "_", start_date, "_", end_date, suffix)
    end
    prefix = colon2dash(prefix)
    prefix = underscore_spaces(prefix)
    return prefix
end

function metadata_prefix(metadata::ERA5Metadata)
    var = ERA5_dataset_variable_names[metadata.name]
    dataset = dataset_name(metadata.dataset)
    start_date = start_date_str(metadata.dates)
    end_date = end_date_str(metadata.dates)
    bbox = metadata.bounding_box

    if !isnothing(bbox)
        w, e = bbox_strs(bbox.longitude)
        s, n = bbox_strs(bbox.latitude)
        suffix = string(w, e, s, n)
    else
        suffix = ""
    end

    if start_date == end_date
        prefix = string(var, "_", dataset, "_", start_date, suffix)
    else
        prefix = string(var, "_", dataset, "_", start_date, "_", end_date, suffix)
    end
    prefix = colon2dash(prefix)
    prefix = underscore_spaces(prefix)
    return prefix
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
##### Pressure-level vertical coordinate
#####

const ERA5_gravitational_acceleration = 9.80665

# International Standard Atmosphere height (m) for a given pressure (hPa)
function standard_atmosphere_geopotential_height(p)
    g = ERA5_gravitational_acceleration
    T⁰ = 288.15 # K
    p⁰ = 1013.25 * hPa
    Rᵈ = 287.0528 # J/(kg-K)

    return (Rᵈ * T⁰ / g) * log(p⁰ / p)
end

# Build z-interfaces (Nz+1 values) from pressure levels.
# Levels may be in any order; output is sorted so k=1 is highest pressure (lowest altitude).
function standard_atmosphere_z_interfaces(levels)
    @info """
    Calculating z-interfaces based on International Standard Atmosphere...
    For greater accuracy, use `mean_geopotential_heights`!
    """
    sorted_levels = sort(levels, rev=true)   # highest pressure first → k=1 is bottom
    heights = standard_atmosphere_geopotential_height.(Float64.(sorted_levels))
    Nz = length(heights)

    interfaces = Vector{Float64}(undef, Nz + 1)

    if Nz == 1
        interfaces[1] = heights[1] - 0.5
        interfaces[2] = heights[1] + 0.5
    else
        interfaces[1] = heights[1] - (heights[2] - heights[1]) / 2
        for k in 2:Nz
            interfaces[k] = (heights[k-1] + heights[k]) / 2
        end
        interfaces[Nz+1] = heights[Nz] + (heights[Nz] - heights[Nz-1]) / 2
    end

    return interfaces
end

# ERA5 pressure-levels (3-D) data product
z_interfaces(metadata::ERA5PressureMetadata) = standard_atmosphere_z_interfaces(metadata.dataset.levels)

#####
##### pressure_field — synthetic pressure coordinate field
#####

"""
    pressure_field(metadata::ERA5PressureMetadatum, arch=CPU(); halo=(3,3,3))

Return a `CenterField` on the native grid of `metadata` filled with the pressure
value (hPa) at each vertical level. Levels are ordered bottom-to-top (k=1 is the
highest pressure level).
"""
function pressure_field(metadata::ERA5PressureMetadatum, arch=CPU(); halo=(3,3,3))
    grid = native_grid(metadata, arch; halo)
    field = CenterField(grid)
    reversed_levels = sort(metadata.dataset.levels, rev=true)   # highest pressure → k=1
    for (k, p) in enumerate(reversed_levels)
        interior(field)[:, :, k] .= Float32(p)
    end
    fill_halo_regions!(field)
    return field
end

#####
##### mean_geopotential_heights — data-derived static z-coordinate
#####

"""
    mean_geopotential_heights(metadata::ERA5PressureMetadata; arch=CPU())

Compute spatially and temporally averaged geopotential heights (m) for each
pressure level in `metadata`. This provides more accurate z-coordinates than
the standard-atmosphere fallback used by `z_interfaces`.

Downloads the `:geopotential` field for every date in `metadata`, divides by g,
averages over the horizontal domain and all dates, and returns one representative
height per pressure level in bottom-to-top order (k=1 is highest pressure).
"""
function mean_geopotential_heights(metadata::ERA5PressureMetadata; arch=CPU())
    geo_metadata = Metadata(:geopotential; dataset=metadata.dataset,
                            dates=metadata.dates, bounding_box=metadata.bounding_box,
                            dir=metadata.dir)
    heights = zeros(length(metadata.dataset.levels))
    for geo_datum in geo_metadata
        data = retrieve_data(geo_datum) ./ Float32(ERA5_gravitational_acceleration)   # Φ → Z (m)
        heights .+= dropdims(mean(data; dims=(1, 2)); dims=(1, 2))
    end
    heights ./= length(geo_metadata)
    return heights
end

end # module ERA5

