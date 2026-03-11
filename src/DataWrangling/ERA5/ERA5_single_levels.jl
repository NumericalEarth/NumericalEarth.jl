struct ERA5SingleLevelsHourly <: ERA5Dataset end
struct ERA5SingleLevelsMonthly <: ERA5Dataset end

dataset_name(::ERA5SingleLevelsHourly)  = "ERA5SingleLevelsHourly"
dataset_name(::ERA5SingleLevelsMonthly) = "ERA5SingleLevelsMonthly"

# Wave variables are on a 0.5° grid (720×361), atmospheric variables on 0.25° (1440×721)
const ERA5_wave_variables = Set([
    :eastward_stokes_drift, :northward_stokes_drift,
    :significant_wave_height, :mean_wave_period, :mean_wave_direction,
])

#####
##### ERA5 single-level data availability
#####

# ERA5 reanalysis data available from 1940 to present (we use a practical range here)
all_dates(::ERA5SingleLevelsHourly,  var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-31"), step=Hour(1))
all_dates(::ERA5SingleLevelsMonthly, var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-01"), step=Month(1))

# ERA5 single-level data is a spatially 2-D dataset
is_three_dimensional(::ERA5Metadata) = false

function Base.size(::ERA5Dataset, variable)
    if variable in ERA5_wave_variables
        return (720, 361, 1)
    else
        return (1440, 721, 1)
    end
end

#####
##### ERA5 single-level variable name mappings
#####

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

# Variables available for download
available_variables(::ERA5Dataset) = ERA5_dataset_variable_names
dataset_variable_name(md::ERA5Metadata) = ERA5_dataset_variable_names[md.name]
netcdf_variable_name(md::ERA5Metadata) = ERA5_netcdf_variable_names[md.name]

default_inpainting(md::ERA5Metadata) = nothing

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

#####
##### Metadata filename construction
#####

function metadata_prefix(md::ERA5Metadata)
    var = ERA5_dataset_variable_names[md.name]
    dataset = dataset_name(md.dataset)
    start_date = start_date_str(md.dates)
    end_date = end_date_str(md.dates)
    bbox = md.bounding_box

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
