using ...Atmospheres: Atmospheres, PrescribedPrecipitationFlux, AtmosphereThermodynamicsParameters,
                      R_d, R_v
using ...EarthSystemModels.InterfaceComputations: saturation_specific_humidity
using KernelAbstractions: @kernel, @index
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: CenterField, interior
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Utils: launch!
using Thermodynamics: Liquid

const ERA5PrescribedAtmosphere = Atmospheres.PrescribedAtmosphere{<:ERA5Dataset}

ERA5PrescribedAtmosphere(arch::Distributed; kw...) = ERA5PrescribedAtmosphere(child_architecture(arch); kw...)

# ERA5 carries the 2 m dewpoint temperature, not specific humidity. The air is
# saturated at its dewpoint, so qᵛ = qᵛ⁺(Tᵈ, pˢ).
@inline function specific_humidity_from_dewpoint(i, j, k, grid, dewpoint, pressure, ℂ, phase)
    @inbounds Tᵈ = dewpoint[i, j, k]
    @inbounds pˢ = pressure[i, j, k]
    return saturation_specific_humidity(ℂ, Tᵈ, pˢ, phase)
end

function specific_humidity_field_time_series(dewpoint, pressure, thermodynamics_parameters)
    grid = dewpoint.grid
    times = dewpoint.times
    phase = Liquid()
    qᵛ = FieldTimeSeries{Center, Center, Nothing}(grid, times)

    for n in eachindex(times)
        q = KernelFunctionOperation{Center, Center, Nothing}(specific_humidity_from_dewpoint,
                                                             grid, dewpoint[n], pressure[n],
                                                             thermodynamics_parameters, phase)
        set!(qᵛ[n], q)
    end

    return qᵛ
end

"""
    ERA5PrescribedAtmosphere([architecture = CPU()];
                             dataset = ERA5HourlySingleLevel(),
                             start_date = first_date(dataset, :temperature),
                             end_date = last_date(dataset, :temperature),
                             dir = download_ERA5_cache,
                             time_indices_in_memory = 24,
                             time_indexing = Cyclical(),
                             surface_layer_height = 10,    # meters
                             boundary_layer_height = 512,  # meters
                             thermodynamics_parameters = nothing,
                             region = nothing,
                             other_kw...)

Return a [`PrescribedAtmosphere`](@ref) representing ERA5 single-level reanalysis, suitable for regional hindcast forcing.
Eastward/northward 10 m winds, 2 m temperature, and surface pressure are loaded directly; specific humidity is derived from
the 2 m dewpoint and surface pressure (`qᵛ = qᵛ⁺(Tᵈ, pˢ)`); total precipitation is converted from hourly-accumulated depth (m)
to a mass flux (kg m⁻² s⁻¹) at load time and wrapped in a `PrescribedPrecipitationFlux`.

`region` (a `BoundingBox`) restricts the download and the native grid to a sub-domain; the coupled model interpolates the
native-resolution atmosphere onto the exchange grid. Pass `thermodynamics_parameters` to share a specific thermodynamics with
the rest of the model (defaults to `AtmosphereThermodynamicsParameters` at the data's float type).
"""
function ERA5PrescribedAtmosphere(architecture = CPU();
                                  dataset = ERA5HourlySingleLevel(),
                                  start_date = first_date(dataset, :temperature),
                                  end_date = last_date(dataset, :temperature),
                                  dir = download_ERA5_cache,
                                  time_indices_in_memory = 24,
                                  time_indexing = Cyclical(),
                                  surface_layer_height = 10,
                                  boundary_layer_height = 512,
                                  thermodynamics_parameters = nothing,
                                  region = nothing,
                                  other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    # Download every variable up front in one bundle: backends batch a `MetadataSet`
    # across variables (one CDS request, or one era5cli invocation with concurrent
    # per-variable requests), so the `FieldTimeSeries` below find their files cached.
    mset = MetadataSet(:eastward_velocity, :northward_velocity, :temperature,
                       :dewpoint_temperature, :surface_pressure, :total_precipitation;
                       dataset, start_date, end_date, dir, region)
    Downloads.download(mset)

    era5_fts(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), architecture; kw...)

    u    = era5_fts(:eastward_velocity)
    v    = era5_fts(:northward_velocity)
    T    = era5_fts(:temperature)
    Tᵈ   = era5_fts(:dewpoint_temperature)
    p    = era5_fts(:surface_pressure)
    rain = era5_fts(:total_precipitation)

    grid  = u.grid
    times = u.times
    FT    = eltype(u)

    ℂ = isnothing(thermodynamics_parameters) ? AtmosphereThermodynamicsParameters(FT) :
                                                thermodynamics_parameters
    qᵛ = specific_humidity_field_time_series(Tᵈ, p, ℂ)

    precipitation_flux = PrescribedPrecipitationFlux(; rain)

    return Atmospheres.PrescribedAtmosphere(grid, times;
                                            source = dataset,
                                            velocities = (; u, v),
                                            temperature = T,
                                            specific_humidity = qᵛ,
                                            pressure = p,
                                            precipitation_flux,
                                            thermodynamics_parameters = ℂ,
                                            surface_layer_height  = convert(FT, surface_layer_height),
                                            boundary_layer_height = convert(FT, boundary_layer_height))
end

# Pressure on a `PressureLevelGrid` is the level coordinate (Pa), constant in space and time
function pressure_level_field(grid, dataset, architecture)
    FT = eltype(grid)
    pˡᵉᵛᵉˡ = on_architecture(architecture, FT.(dataset.pressure_levels))
    pressure = CenterField(grid)
    Nz = length(dataset.pressure_levels)
    interior(pressure) .= reshape(pˡᵉᵛᵉˡ, 1, 1, Nz)
    fill_halo_regions!(pressure)
    return pressure
end

@kernel function _reconstruct_near_surface_snapshot!(geopotential, temperature,
                                                      eastward_velocity, northward_velocity,
                                                      specific_humidity, cloud_liquid, rain,
                                                      cloud_ice, snow, pressure,
                                                      source_geopotential, source_temperature,
                                                      source_eastward_velocity,
                                                      source_northward_velocity,
                                                      source_specific_humidity,
                                                      source_cloud_liquid, source_rain,
                                                      source_cloud_ice, source_snow,
                                                      surface_geopotential, surface_temperature,
                                                      surface_eastward_velocity,
                                                      surface_northward_velocity,
                                                      surface_specific_humidity, surface_pressure,
                                                      pressure_levels, reference_height,
                                                      dry_air_gas_constant,
                                                      vapor_gas_constant,
                                                      gravitational_acceleration)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        qᵛ = clamp(surface_specific_humidity[i, j, 1], 0, 1)
        T = surface_temperature[i, j, 1]
        Rᵐ = dry_air_gas_constant * (1 - qᵛ) + vapor_gas_constant * qᵛ
        pˢ = surface_pressure[i, j, 1]
        pʳ = pˢ * exp(-gravitational_acceleration * reference_height / (Rᵐ * T))

        # Insert the surface anchor at k=1 and shift the useful pressure-level column up by one slot,
        # dropping only its topmost (1 hPa by default) level. Over elevated terrain, every shifted
        # pressure level below the anchor is replaced as well, preserving a monotone column without
        # exposing ERA5's extrapolated sub-surface values.
        source_k = max(1, k - 1)
        source_pressure = pressure_levels[source_k]
        reconstruct = (k == 1) | (source_pressure >= pʳ)
        Φʳ = surface_geopotential[i, j, 1] + gravitational_acceleration * reference_height

        geopotential[i, j, k] = ifelse(reconstruct, Φʳ, source_geopotential[i, j, source_k])
        temperature[i, j, k] = ifelse(reconstruct, T, source_temperature[i, j, source_k])
        eastward_velocity[i, j, k] = ifelse(reconstruct,
                                             surface_eastward_velocity[i, j, 1],
                                             source_eastward_velocity[i, j, source_k])
        northward_velocity[i, j, k] = ifelse(reconstruct,
                                              surface_northward_velocity[i, j, 1],
                                              source_northward_velocity[i, j, source_k])
        specific_humidity[i, j, k] = ifelse(reconstruct, qᵛ,
                                             source_specific_humidity[i, j, source_k])
        cloud_liquid[i, j, k] = ifelse(reconstruct, 0,
                                        source_cloud_liquid[i, j, source_k])
        rain[i, j, k] = ifelse(reconstruct, 0, source_rain[i, j, source_k])
        cloud_ice[i, j, k] = ifelse(reconstruct, 0,
                                     source_cloud_ice[i, j, source_k])
        snow[i, j, k] = ifelse(reconstruct, 0, source_snow[i, j, source_k])
        pressure[i, j, k] = ifelse(reconstruct, pʳ, source_pressure)
    end
end

function reconstruct_near_surface_snapshot!(grid, geopotential, state, pressure,
                                            source_geopotential, source_state,
                                            surface_geopotential, surface_state,
                                            pressure_levels, thermodynamics_parameters,
                                            reference_height)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _reconstruct_near_surface_snapshot!,
            geopotential, state.temperature, state.eastward_velocity,
            state.northward_velocity, state.specific_humidity,
            state.cloud_liquid, state.rain, state.cloud_ice, state.snow, pressure,
            source_geopotential, source_state.temperature,
            source_state.eastward_velocity, source_state.northward_velocity,
            source_state.specific_humidity, source_state.cloud_liquid,
            source_state.rain, source_state.cloud_ice, source_state.snow,
            surface_geopotential, surface_state.temperature,
            surface_state.eastward_velocity, surface_state.northward_velocity,
            surface_state.specific_humidity, surface_state.pressure,
            pressure_levels, reference_height,
            R_d(thermodynamics_parameters), R_v(thermodynamics_parameters),
            ERA5_gravitational_acceleration)
    return nothing
end

function copy_volume_field_time_series(field_time_series, grid, times)
    copied = FieldTimeSeries{Center, Center, Center}(grid, times)
    for n in eachindex(times)
        set!(copied[n], field_time_series[n])
    end
    return copied
end

function era5_near_surface_state(architecture, dataset, dates, region, dir,
                                 thermodynamics_parameters)
    surface_dataset = DataWrangling.matching_single_level_dataset(dataset)
    metadata = MetadataSet(:temperature, :dewpoint_temperature,
                           :eastward_velocity, :northward_velocity, :surface_pressure;
                           dataset = surface_dataset, dates, region, dir)
    Downloads.download(metadata)

    surface_fts(name) = FieldTimeSeries(Metadata(name; dataset = surface_dataset, dates,
                                                  region, dir), architecture;
                                         time_indices_in_memory = length(dates))
    temperature = surface_fts(:temperature)
    dewpoint = surface_fts(:dewpoint_temperature)
    eastward_velocity = surface_fts(:eastward_velocity)
    northward_velocity = surface_fts(:northward_velocity)
    pressure = surface_fts(:surface_pressure)
    specific_humidity = specific_humidity_field_time_series(dewpoint, pressure,
                                                              thermodynamics_parameters)

    return (; temperature, eastward_velocity, northward_velocity,
            specific_humidity, pressure)
end

function reconstruct_near_surface_pressure_levels!(grid, times, state, surface_state,
                                                   pressure_levels,
                                                   thermodynamics_parameters,
                                                   reference_height)
    reference_height >= 0 || throw(ArgumentError("near-surface reference height must be nonnegative"))
    surface_state.temperature.times == times ||
        throw(ArgumentError("near-surface and pressure-level time axes must match"))

    arch = architecture(grid)
    FT = eltype(grid)
    pressure_levels = on_architecture(arch, FT.(pressure_levels))
    pressure = FieldTimeSeries{Center, Center, Center}(grid, times)
    geopotential = grid.z.geopotential.time_series
    surface_geopotential = grid.z.surface_geopotential
    source_geopotential = copy_volume_field_time_series(geopotential, grid, times)
    source_state = map(field_time_series -> copy_volume_field_time_series(field_time_series,
                                                                          grid, times), state)

    for n in eachindex(times)
        snapshot = map(field_time_series -> field_time_series[n], state)
        source_snapshot = map(field_time_series -> field_time_series[n], source_state)
        surface_snapshot = map(field_time_series -> field_time_series[n], surface_state)
        reconstruct_near_surface_snapshot!(grid, geopotential[n], snapshot, pressure[n],
                                           source_geopotential[n], source_snapshot,
                                           surface_geopotential, surface_snapshot,
                                           pressure_levels, thermodynamics_parameters,
                                           FT(reference_height))
    end

    fill_halo_regions!(geopotential)
    fill_halo_regions!(state)
    fill_halo_regions!(pressure)
    return pressure
end

"""
    ERA5PrescribedAtmosphere(bounding_box::BoundingBox, dates;
                             architecture = CPU(),
                             dataset = ERA5HourlyPressureLevels(),
                             dir = download_ERA5_cache,
                             time_indices_in_memory = nothing,
                             thermodynamics_parameters = nothing,
                             reconstruct_near_surface = false,
                             near_surface_reference_height = 10,
                             other_kw...)

Return a 3-D [`PrescribedAtmosphere`](@ref) built from ERA5 **pressure-level** reanalysis over `bounding_box` at the
requested `dates` — a range or vector of dates, or a `(start_date, end_date)` tuple that expands to the dataset's native
(hourly or monthly) cadence — on ERA5's **native grid**: a `PressureLevelGrid` at the reanalysis' native horizontal resolution
with a **time-varying** geopotential-height vertical (each pressure level's height follows the reanalysis as the atmosphere's
clock advances). Each variable loads natively and raw (per pressure level, no vertical remap); a downstream model (e.g. a
`NestedSimulation` child) interpolates the parent onto its own grid on the fly, at the current heights.
`time_indices_in_memory` defaults to all dates.

With `reconstruct_near_surface=true`, a synthetic near-surface anchor is inserted into the lowest
slot and the pressure-level column is shifted up one slot, preserving the lowest pressure level and
dropping the topmost level (1 hPa for the default dataset). The anchor combines ERA5 2 m temperature
and dewpoint, 10 m winds, surface pressure, and surface geopotential at
`near_surface_reference_height` (10 m by default). Surface
pressure is hydrostatically adjusted to that height; hydrometeors are set to zero at the anchor.
Over terrain, all nominal pressure levels below the anchor are replaced as well. This avoids holding
the 1000 hPa state constant from its geopotential height down to the surface. The initial
implementation requires all requested dates to remain resident in memory.

The atmosphere holds eastward/northward `velocities`, `temperature`, `specific_humidity`, `microphysical_variables = (; qᶜˡ, qʳ, qᶜⁱ, qˢ)`
(cloud liquid/ice + rain/snow water content), and `pressure` (the level coordinate, via [`pressure_level_field`](@ref),
or a time-varying reconstructed field when near-surface reconstruction is enabled).
Use as the parent of a [`NestedSimulation`](@ref).
"""
function ERA5PrescribedAtmosphere(bounding_box::BoundingBox, dates;
                                  architecture = CPU(),
                                  dataset = ERA5HourlyPressureLevels(),
                                  dir = download_ERA5_cache,
                                  time_indices_in_memory = nothing,
                                  thermodynamics_parameters = nothing,
                                  reconstruct_near_surface = false,
                                  near_surface_reference_height = 10,
                                  other_kw...)

    region = bounding_box
    dates = DataWrangling.expand_dates(dataset, :temperature, dates)
    time_indices_in_memory = something(time_indices_in_memory, length(dates))
    reconstruct_near_surface && time_indices_in_memory < length(dates) &&
        throw(ArgumentError("near-surface reconstruction currently requires all dates in memory"))

    # One up-front bundle download (including the geopotential the vertical discretization
    # needs) so the per-variable loads below find their files cached.
    mset = MetadataSet(:temperature, :eastward_velocity, :northward_velocity,
                       :specific_humidity, :specific_cloud_liquid_water_content,
                       :specific_rain_water_content, :specific_cloud_ice_water_content,
                       :specific_snow_water_content, :geopotential;
                       dataset, dates, region, dir)
    Downloads.download(mset)

    # One clock drives both the atmosphere's own time and the pressure levels' time-varying
    # geopotential heights. `time_step!(atmosphere, Δt)` advances it, so the grid geometry a child
    # interpolates over follows the reanalysis in time.
    temperature_metadata = Metadata(:temperature; dataset, dates, region, dir)
    FT = eltype(temperature_metadata)
    clock = Clock{FT}(time = zero(FT))

    # Build the native PressureLevelGrid once, with a geopotential `TimeSeriesInterpolation` bound to
    # `clock`. `_with_z` carries the fully built vertical only into this transient grid construction —
    # the atmosphere's `source` stays the plain dataset (product identity only).
    plvd = per_column_geopotential_discretization(temperature_metadata; clock)
    grid = native_grid(Metadata(:temperature; dataset = _with_z(dataset, plvd), dates, region, dir), architecture)

    era5_fts(name) = era5_native_pressure_fts(Metadata(name; dataset, dates, region, dir), grid; time_indices_in_memory, other_kw...)

    u   = era5_fts(:eastward_velocity)
    v   = era5_fts(:northward_velocity)
    T   = era5_fts(:temperature)
    qᵛ  = era5_fts(:specific_humidity)
    qᶜˡ = era5_fts(:specific_cloud_liquid_water_content)
    qʳ  = era5_fts(:specific_rain_water_content)
    qᶜⁱ = era5_fts(:specific_cloud_ice_water_content)
    qˢ  = era5_fts(:specific_snow_water_content)

    times = T.times
    FT = eltype(T)
    ℂ = isnothing(thermodynamics_parameters) ? AtmosphereThermodynamicsParameters(FT) :
                                                thermodynamics_parameters
    state = (; temperature = T, eastward_velocity = u, northward_velocity = v,
             specific_humidity = qᵛ, cloud_liquid = qᶜˡ, rain = qʳ,
             cloud_ice = qᶜⁱ, snow = qˢ)

    if reconstruct_near_surface
        surface_state = era5_near_surface_state(architecture, dataset, dates, region, dir, ℂ)
        pressure = reconstruct_near_surface_pressure_levels!(grid, times, state, surface_state,
                                                             dataset.pressure_levels, ℂ,
                                                             near_surface_reference_height)
    else
        pressure = pressure_level_field(grid, dataset, architecture)
    end

    return Atmospheres.PrescribedAtmosphere(grid, times;
                                            clock,
                                            source = dataset,
                                            velocities = (; u, v),
                                            temperature = T,
                                            specific_humidity = qᵛ,
                                            microphysical_variables = (; qᶜˡ, qʳ, qᶜⁱ, qˢ),
                                            pressure,
                                            thermodynamics_parameters = ℂ)
end

"""
    PrescribedAtmosphere(bounding_box, dates, dataset::ERA5PressureLevelsDataset; kw...)

Dataset-dispatched constructor: build an [`ERA5PrescribedAtmosphere`](@ref) over `bounding_box`
at `dates` on `dataset`'s native grid. Keyword arguments flow to `ERA5PrescribedAtmosphere`.
"""
Atmospheres.PrescribedAtmosphere(bounding_box::BoundingBox, dates, dataset::ERA5PressureLevelsDataset; kw...) =
    ERA5PrescribedAtmosphere(bounding_box, dates; dataset, kw...)
