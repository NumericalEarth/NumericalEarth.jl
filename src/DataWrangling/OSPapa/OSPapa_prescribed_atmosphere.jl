"""
    relative_humidity_to_specific_humidity(RH, T_celsius, P_hPa)

Convert relative humidity (%) to specific humidity (kg/kg) using the
August-Roche-Magnus formula for saturation vapor pressure.

Arguments
=========
- `RH`: relative humidity in percent
- `T`: air temperature in degrees Celsius
- `P`: atmospheric pressure in hPa (mbar)
"""
function relative_humidity_to_specific_humidity(RH, T, P)
    Mₐ = 28.9634 # Molar mass of dry air in g/mol
    Mᵥ = 18.01528 # Molar mass of water vapor in g/mol
    ε = Mᵥ / Mₐ  # Ratio of molar masses

    eₛ = 6.1094 * exp(17.625 * T / (T + 243.04))  # Saturation vapor pressure in hPa
    eₚ = (RH / 100) * eₛ  # Actual vapor pressure in hPa
    q = ε * eₚ / (P - (1 - ε) * eₚ)  # Specific humidity in kg/kg

    return q
end

"""
    fill_gaps!(data; max_gap=6)

Fill NaN gaps in a 1D time series using linear interpolation.
Gaps longer than `max_gap` points are left as NaN with a warning.
"""
function fill_gaps!(data; max_gap=6)
    N = length(data)
    i = 1
    while i <= N
        if isnan(data[i])
            gap_start = i
            while i <= N && isnan(data[i])
                i += 1
            end
            gap_end = i - 1
            gap_length = gap_end - gap_start + 1

            if gap_start == 1 || gap_end == N
                # Edge gap: fill with nearest valid value
                if gap_start == 1 && gap_end < N
                    data[gap_start:gap_end] .= data[gap_end + 1]
                elseif gap_end == N && gap_start > 1
                    data[gap_start:gap_end] .= data[gap_start - 1]
                end
            elseif gap_length > max_gap
                @warn "Large gap of $gap_length hours at indices $gap_start:$gap_end left unfilled"
            else
                # Linear interpolation
                v0 = data[gap_start - 1]
                v1 = data[gap_end + 1]
                for j in gap_start:gap_end
                    α = (j - gap_start + 1) / (gap_length + 1)
                    data[j] = v0 + α * (v1 - v0)
                end
            end
        else
            i += 1
        end
    end
    return data
end

"""
    read_ospapa_variable(ds, varname, time_indices)

Read a variable from the OS Papa NetCDF dataset, squeezing spatial dimensions
to a 1D time series. Replaces `missing` values with `NaN`.
"""
function read_ospapa_variable(ds, varname, time_indices)
    raw = ds[varname][1, 1, 1, time_indices]
    data = Float64.(replace(raw, missing => NaN))
    return data
end

"""
    OSPapaPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                start_date = DateTime(2007, 6, 8),
                                end_date = DateTime(2023, 6, 1),
                                dir = download_OSPapa_cache,
                                surface_layer_height = 2.5,
                                max_gap_hours = 72)

Construct a `PrescribedAtmosphere` from Ocean Station Papa buoy observations.

Data is automatically downloaded from the NOAA/PMEL AWS S3 bucket if not
already cached locally.

The buoy provides hourly measurements of wind, air temperature, relative humidity,
barometric pressure, shortwave and longwave radiation, and precipitation.
Relative humidity is converted to specific humidity; units are converted to
SI (Kelvin, Pa, kg/m²/s).

!!! note "Radiation and albedo"
    The buoy `SW` and `LW` variables are **downwelling** fluxes. When this
    atmosphere is used with `OceanOnlyModel`, ClimaOcean applies its own
    ocean albedo (default α = 0.05) to compute net absorbed shortwave, and
    computes upwelling longwave from the model SST via Stefan-Boltzmann. This
    means the resulting net heat flux will differ from the COARE-computed
    `QNET` available via [`OSPapaPrescribedFluxes`](@ref). If you need the
    exact observed net fluxes, use [`OSPapaPrescribedFluxBoundaryConditions`](@ref)
    instead.

Keyword Arguments
=================
- `start_date`: start of the time range (default: `first_date(OSPapaHourly(), :air_temperature)`)
- `end_date`: end of the time range (default: `last_date(OSPapaHourly(), :air_temperature)`)
- `dir`: directory for cached data files
- `surface_layer_height`: measurement height in meters (default: 2.5, matching
  the buoy's temperature/humidity instruments)
- `max_gap_hours`: maximum gap size (in hours) to fill by linear interpolation
  (default: 72)
"""
function OSPapaPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                    start_date = first_date(OSPapaHourly(), :air_temperature),
                                    end_date = last_date(OSPapaHourly(), :air_temperature),
                                    dir = download_OSPapa_cache,
                                    surface_layer_height = 2.5,
                                    max_gap_hours = 72)

    on_arch = arr -> Oceananigans.on_architecture(architecture, arr)

    filepath = download_ospapa_file(dir)

    ds = NCDataset(filepath)

    all_times = ds["TIME"][:]
    time_indices = findall(t -> start_date <= t <= end_date, all_times)

    if isempty(time_indices)
        close(ds)
        error("No data found between $start_date and $end_date")
    end

    times_datetime = all_times[time_indices]

    # Read atmospheric variables
    uwnd = read_ospapa_variable(ds, "UWND", time_indices)
    vwnd = read_ospapa_variable(ds, "VWND", time_indices)
    airt = read_ospapa_variable(ds, "AIRT", time_indices)
    relh = read_ospapa_variable(ds, "RELH", time_indices)
    atms = read_ospapa_variable(ds, "ATMS", time_indices)
    sw   = read_ospapa_variable(ds, "SW",   time_indices)
    lw   = read_ospapa_variable(ds, "LW",   time_indices)
    rain = read_ospapa_variable(ds, "RAIN", time_indices)

    close(ds)

    # Fill gaps via linear interpolation
    for data in (uwnd, vwnd, airt, relh, atms, sw, lw, rain)
        fill_gaps!(data; max_gap=max_gap_hours)
    end

    # Unit conversions
    T_K  = airt .+ 273.15                # °C → K
    P_Pa = atms .* 100                   # mbar → Pa
    q    = relative_humidity_to_specific_humidity.(relh, airt, atms)  # % → kg/kg
    rain_kgm2s = rain ./ 3600            # mm/hr → kg/m²/s

    # Times in seconds relative to start
    t0 = times_datetime[1]
    times_seconds = Float64[Dates.value(t - t0) / 1000 for t in times_datetime]

    # Build atmosphere on a zero-dimensional grid
    atmosphere_grid = RectilinearGrid(architecture, FT;
                                      size = (),
                                      topology = (Flat, Flat, Flat))

    atmosphere = PrescribedAtmosphere(atmosphere_grid, times_seconds;
                                      surface_layer_height = convert(FT, surface_layer_height))

    Nt = length(times_seconds)
    parent(atmosphere.velocities.u)                    .= on_arch(reshape(FT.(uwnd),       1, 1, 1, Nt))
    parent(atmosphere.velocities.v)                    .= on_arch(reshape(FT.(vwnd),       1, 1, 1, Nt))
    parent(atmosphere.tracers.T)                       .= on_arch(reshape(FT.(T_K),        1, 1, 1, Nt))
    parent(atmosphere.tracers.q)                       .= on_arch(reshape(FT.(q),          1, 1, 1, Nt))
    parent(atmosphere.pressure)                        .= on_arch(reshape(FT.(P_Pa),       1, 1, 1, Nt))
    parent(atmosphere.downwelling_radiation.shortwave) .= on_arch(reshape(FT.(sw),         1, 1, 1, Nt))
    parent(atmosphere.downwelling_radiation.longwave)  .= on_arch(reshape(FT.(lw),         1, 1, 1, Nt))
    parent(atmosphere.freshwater_flux.rain)            .= on_arch(reshape(FT.(rain_kgm2s), 1, 1, 1, Nt))

    return atmosphere
end
