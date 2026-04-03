using NumericalEarth.Atmospheres: PrescribedAtmosphere, TwoBandDownwellingRadiation
using Oceananigans.Architectures: AbstractArchitecture, CPU
using Oceananigans.OutputReaders: Cyclical

using NumericalEarth.DataWrangling: FieldTimeSeries,
                                    Metadata,
                                    first_date,
                                    last_date,
                                    all_dates,
                                    compute_native_date_range

"""
    ERA5PrescribedAtmosphere([architecture = CPU(), FT = Float32];
                             dataset = ERA5Hourly(),
                             start_date = first_date(dataset, :temperature),
                             end_date = last_date(dataset, :temperature),
                             region = nothing,
                             time_indices_in_memory = 2,
                             time_indexing = Cyclical(),
                             surface_layer_height = 10)

Return a `PrescribedAtmosphere` constructed from ERA5 reanalysis data.

The atmosphere includes 10-meter winds, 2-meter temperature, specific humidity,
surface pressure, and downwelling shortwave and longwave radiation.

Keyword Arguments
=================

- `dataset`: ERA5 dataset type. Default: `ERA5Hourly()`.
- `start_date`, `end_date`: date range to load.
- `region`: spatial region (`BoundingBox`, `Column`, or `nothing` for global).
- `time_indices_in_memory`: number of time snapshots held in memory. Default: 2.
- `time_indexing`: time interpolation scheme. Default: `Cyclical()`.
- `surface_layer_height`: height of the atmospheric surface layer in meters. Default: 10.
"""
function ERA5PrescribedAtmosphere(architecture::AbstractArchitecture = CPU(), FT = Float32;
                                  dataset = ERA5Hourly(),
                                  start_date = first_date(dataset, :temperature),
                                  end_date = last_date(dataset, :temperature),
                                  region = nothing,
                                  time_indices_in_memory = 2,
                                  time_indexing = Cyclical(),
                                  surface_layer_height = 10)

    kw = (; time_indices_in_memory, time_indexing)

    variables = (:eastward_velocity, :northward_velocity,
                 :temperature, :specific_humidity,
                 :surface_pressure, :total_precipitation,
                 :downwelling_longwave_radiation, :downwelling_shortwave_radiation)

    # Pre-download all variables in a single batch request to avoid
    # 8 separate CDS API calls (each of which queues independently).
    native_dates = all_dates(dataset, :temperature)
    dates = compute_native_date_range(native_dates, start_date, end_date)
    all_metadata = [Metadata(v; dataset, dates, region) for v in variables]
    download_dataset(all_metadata)

    function era5_field_time_series(variable_name)
        metadata = Metadata(variable_name; dataset, dates, region)
        return FieldTimeSeries(metadata, architecture; kw...)
    end

    ua   = era5_field_time_series(:eastward_velocity)
    va   = era5_field_time_series(:northward_velocity)
    Ta   = era5_field_time_series(:temperature)
    qa   = era5_field_time_series(:specific_humidity)
    pa   = era5_field_time_series(:surface_pressure)
    Fra  = era5_field_time_series(:total_precipitation)
    ℐꜜˡʷ = era5_field_time_series(:downwelling_longwave_radiation)
    ℐꜜˢʷ = era5_field_time_series(:downwelling_shortwave_radiation)

    times = ua.times
    grid  = ua.grid

    velocities = (u = ua, v = va)
    tracers    = (T = Ta, q = qa)
    pressure   = pa

    freshwater_flux = (precipitation = Fra, )  # ERA5 only has total_precipitation

    downwelling_radiation = TwoBandDownwellingRadiation(shortwave = ℐꜜˢʷ, longwave = ℐꜜˡʷ)

    FT = eltype(ua)
    surface_layer_height = convert(FT, surface_layer_height)

    atmosphere = PrescribedAtmosphere(grid, times;
                                      velocities,
                                      freshwater_flux,
                                      tracers,
                                      downwelling_radiation,
                                      surface_layer_height,
                                      pressure)

    return atmosphere
end
