struct ERA5HourlySingleLevel <: ERA5Dataset end
struct ERA5MonthlySingleLevel <: ERA5Dataset end

dataset_name(::ERA5HourlySingleLevel)  = "ERA5HourlySingleLevel"
dataset_name(::ERA5MonthlySingleLevel) = "ERA5MonthlySingleLevel"

# Wave variables are on a 0.5° grid (720×361), atmospheric variables on 0.25° (1440×721)
const ERA5_wave_variables = Set([
    :eastward_stokes_drift, :northward_stokes_drift,
    :significant_wave_height, :mean_wave_period, :mean_wave_direction,
])

# Mean rate / accumulated variables (CDS "step type" = accum).
# All other single-level variables are instantaneous.
# See ECMWF ERA5 documentation, Tables 3 and 4:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations
const ERA5_single_level_accumulated_variables = Set([
    :total_precipitation,
    :mean_surface_sensible_heat_flux,
    :mean_surface_latent_heat_flux,
    :mean_surface_momentum_flux_x,
    :mean_surface_momentum_flux_y,
    :downwelling_shortwave_radiation,
    :downwelling_longwave_radiation,
    :evaporation,
    :mean_evaporation_rate,
])

#####
##### ERA5 single-level data availability
#####

# ERA5 reanalysis data available from 1940 to present (we use a practical range here)
all_dates(::ERA5HourlySingleLevel,  var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-31"), step=Hour(1))
all_dates(::ERA5MonthlySingleLevel, var) = range(DateTime("1940-01-01"), stop=DateTime("2024-12-01"), step=Month(1))

# ERA5 single-level data is a spatially 2-D dataset
is_three_dimensional(::ERA5Metadata) = false

function Base.size(::ERA5Dataset, variable)
    if variable in ERA5_wave_variables
        return (720, 360, 1)
    else
        return (1440, 720, 1)
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
    :mean_surface_momentum_flux_x    => "mean_eastward_turbulent_surface_stress",
    :mean_surface_momentum_flux_y    => "mean_northward_turbulent_surface_stress",
    :sea_surface_temperature         => "sea_surface_temperature",
    :mean_surface_sensible_heat_flux => "mean_surface_sensible_heat_flux",
    :mean_surface_latent_heat_flux   => "mean_surface_latent_heat_flux",
    :downwelling_shortwave_radiation => "surface_solar_radiation_downwards",
    :downwelling_longwave_radiation  => "surface_thermal_radiation_downwards",
    :total_cloud_cover               => "total_cloud_cover",
    :evaporation                     => "evaporation",
    :mean_evaporation_rate           => "mean_evaporation_rate",
    :specific_humidity               => "specific_humidity",
    :eastward_stokes_drift           => "u_component_stokes_drift",
    :northward_stokes_drift          => "v_component_stokes_drift",
    :significant_wave_height         => "significant_height_of_combined_wind_waves_and_swell",
    :mean_wave_period                => "mean_wave_period",
    :mean_wave_direction             => "mean_wave_direction",
)

# NetCDF short variable names (what's actually in the downloaded files)
# - These differ from the CDS API variable names above
# - The expected "shortName" (see https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#heading-Parameterlistings)
#   did not always match the netcdf variable names?! In those cases, the longName and units were manually verified.
ERA5_netcdf_variable_names = Dict(
    :temperature                     => "t2m",
    :dewpoint_temperature            => "d2m",
    :eastward_velocity               => "u10",
    :northward_velocity              => "v10",
    :surface_pressure                => "sp",
    :mean_sea_level_pressure         => "msl",
    :total_precipitation             => "tp",
    :mean_surface_momentum_flux_x    => "avg_iews", # shortName: metss
    :mean_surface_momentum_flux_y    => "avg_inss", # shortName: mntss
    :sea_surface_temperature         => "sst",
    :mean_surface_sensible_heat_flux => "avg_ishf", # shortName: msshf
    :mean_surface_latent_heat_flux   => "avg_slhtf", # shortName: mslhf
    :downwelling_shortwave_radiation => "ssrd",
    :downwelling_longwave_radiation  => "strd",
    :total_cloud_cover               => "tcc",
    :evaporation                     => "e",
    :mean_evaporation_rate           => "avg_ie", # shortName: mer
    :specific_humidity               => "q",
    :eastward_stokes_drift           => "ust",
    :northward_stokes_drift          => "vst",
    :significant_wave_height         => "swh",
    :mean_wave_period                => "mwp",
    :mean_wave_direction             => "mwd",
)

# Variables available for download
available_variables(::ERA5Dataset) = ERA5_dataset_variable_names

# `dataset_variable_name` returns the short name as stored in the NetCDF file
# (e.g. "t2m"). The CDS API catalog name (e.g. "2m_temperature") used in
# download requests is accessed via the `ERA5_dataset_variable_names` dict
# directly in `NumericalEarthCDSAPIExt`.
dataset_variable_name(md::ERA5Metadata) = ERA5_netcdf_variable_names[md.name]

default_inpainting(md::ERA5Metadata) = nothing

"""
    retrieve_data(metadata::ERA5Metadatum)

Retrieve ERA5 data from NetCDF file according to `metadata`.
ERA5 is 2D surface data, so we return a 2D array with an added singleton z-dimension.
"""
function retrieve_data(metadata::ERA5Metadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)

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
    bbox = md.region

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
