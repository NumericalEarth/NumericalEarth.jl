using Oceananigans.Operators: intrinsic_vector
using Oceananigans.Fields: FractionalIndices, interpolate
using Oceananigans.OutputReaders: cpu_interpolating_time_indices, FTS0

using Oceananigans.TimeSteppers: next_time

using ..EarthSystemModels: component_model
using ..Oceans: forcing_barotropic_potential

"""Interpolate the atmospheric state onto the ocean / sea-ice grid."""
function EarthSystemModels.interpolate_state!(exchanger, grid, atmosphere::PrescribedAtmosphere, coupled_model)
    atmosphere_grid = atmosphere.grid

    # Basic model properties
    arch = architecture(grid)
    clock = coupled_model.clock

    #####
    ##### First interpolate atmosphere time series
    ##### in time and to the ocean grid.
    #####

    # We use .data here to save parameter space (unlike Field, adapt_structure for
    # fts = FieldTimeSeries does not return fts.data)
    atmosphere_velocities = (u = atmosphere.velocities.u.data,
                             v = atmosphere.velocities.v.data)

    atmosphere_tracers = merge((T = atmosphere.temperature.data,
                                q = atmosphere.specific_humidity.data),
                               atmosphere.tracers)

    rainfall_flux = surface_rainfall_flux(atmosphere)
    snowfall_flux = surface_snowfall_flux(atmosphere)
    atmosphere_pressure = atmosphere.pressure.data

    atmosphere_fields = exchanger.state
    regridder = exchanger.regridder
    space_fractional_indices = (i = regridder.i, j = regridder.j)

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_data = unwrap_fields(atmosphere_fields)

    kernel_parameters = interface_kernel_parameters(grid)

    # Each field is interpolated in time on its own time axis
    t = clock.time
    precipitation_flux = something(atmosphere.precipitation_flux, PrescribedPrecipitationFlux())

    velocity_time_arguments = map(u -> time_arguments(arch, u, t), atmosphere.velocities)
    pressure_time_arguments = time_arguments(arch, atmosphere.pressure, t)
    rainfall_time_arguments = time_arguments(arch, precipitation_flux.rain, t)
    snowfall_time_arguments = time_arguments(arch, precipitation_flux.snow, t)

    # Tracers other than T and q carry their own grid and location
    tracer_fractional_indices = merge((T = space_fractional_indices, q = space_fractional_indices), regridder.tracers)
    tracer_time_arguments = map(tracer -> time_arguments(arch, tracer, t),
                                merge((T = atmosphere.temperature, q = atmosphere.specific_humidity), atmosphere.tracers))

    launch!(arch, grid, kernel_parameters,
            _interpolate_primary_atmospheric_state!,
            atmosphere_data,
            space_fractional_indices,
            grid,
            atmosphere_velocities,
            velocity_time_arguments,
            atmosphere_tracers,
            tracer_fractional_indices,
            tracer_time_arguments,
            atmosphere_pressure,
            pressure_time_arguments,
            rainfall_flux,
            rainfall_time_arguments,
            snowfall_flux,
            snowfall_time_arguments)

    return nothing
end

"""Fill the ocean's barotropic potential `p / ρᵒᶜ` at the start and end of the coupled time step."""
function EarthSystemModels.update_barotropic_potential!(atmosphere::PrescribedAtmosphere, coupled_model, Δt)
    Φ = forcing_barotropic_potential(coupled_model.ocean)
    isnothing(Φ) && return nothing

    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    regridder = exchanger.atmosphere.regridder
    space_fractional_indices = (i = regridder.i, j = regridder.j)
    ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
    p = atmosphere.pressure
    clock = coupled_model.clock

    for (potential, t) in ((Φ.previous, clock.time), (Φ.next, next_time(clock, Δt)))
        update_field_time_series!(p, Time(t))
        time_interpolator = cpu_interpolating_time_indices(arch, p.times, p.time_indexing, t)
        launch!(arch, grid, interface_kernel_parameters(grid),
                _interpolate_barotropic_potential!,
                potential, space_fractional_indices, time_interpolator,
                p.data, p.backend, p.time_indexing, ρᵒᶜ)
    end

    ocean_clock = component_model(coupled_model.ocean).clock
    KT = eltype(Φ.times)
    copyto!(Φ.times, KT[ocean_clock.time, next_time(ocean_clock, Δt)])

    return nothing
end

@kernel function _interpolate_barotropic_potential!(Φ, space_fractional_indices, time_interpolator,
                                                     pressure, backend, time_indexing, ρᵒᶜ)
    i, j = @index(Global, NTuple)

    fi = get_fractional_index(i, j, space_fractional_indices.i)
    fj = get_fractional_index(i, j, space_fractional_indices.j)
    X = FractionalIndices(fi, fj, nothing)
    pᵃᵗ = interp_atmos_time_series(pressure, X, time_interpolator, backend, time_indexing)

    @inbounds Φ[i, j, 1] = pᵃᵗ / ρᵒᶜ
end

@inline @generated function unwrap_fields(fields::NamedTuple{names}) where names
    unwrapped = Tuple(:(underlying_data(fields.$name)) for name in names)
    return :(NamedTuple{names}(($(unwrapped...),)))
end

time_arguments(arch, fts, t) = (cpu_interpolating_time_indices(arch, fts.times, fts.time_indexing, t), fts.backend, fts.time_indexing)
time_arguments(arch, ::ConstantField, t) = nothing
time_arguments(arch, ::Nothing, t) = (nothing, nothing, nothing)

@inline get_fractional_index(i, j, ::Nothing) = nothing
@inline get_fractional_index(i, j, frac) = @inbounds frac[i, j, 1]

@inline underlying_data(f) = f.data
@inline underlying_data(c::ConstantField) = c

@kernel function _interpolate_primary_atmospheric_state!(surface_atmos_state,
                                                         space_fractional_indices,
                                                         exchange_grid,
                                                         atmos_velocities,
                                                         velocity_time_arguments,
                                                         atmos_tracers,
                                                         tracer_fractional_indices,
                                                         tracer_time_arguments,
                                                         atmos_pressure,
                                                         pressure_time_arguments,
                                                         rainfall_flux,
                                                         rainfall_time_arguments,
                                                         snowfall_flux,
                                                         snowfall_time_arguments)

    i, j = @index(Global, NTuple)

    ii = space_fractional_indices.i
    jj = space_fractional_indices.j
    fi = get_fractional_index(i, j, ii)
    fj = get_fractional_index(i, j, jj)

    x_itp = FractionalIndices(fi, fj, nothing)

    uᵃᵗ = interp_atmos_time_series(atmos_velocities.u, x_itp, velocity_time_arguments.u...)
    vᵃᵗ = interp_atmos_time_series(atmos_velocities.v, x_itp, velocity_time_arguments.v...)
    pᵃᵗ = interp_atmos_time_series(atmos_pressure,     x_itp, pressure_time_arguments...)

    Mr = interp_atmos_time_series(rainfall_flux, x_itp, rainfall_time_arguments...)
    Ms = interp_atmos_time_series(snowfall_flux, x_itp, snowfall_time_arguments...)

    # Convert atmosphere velocities (usually defined on a latitude-longitude grid) to
    # the frame of reference of the native grid
    kᴺ = size(exchange_grid, 3) # index of the top ocean cell
    uᵃᵗ, vᵃᵗ = intrinsic_vector(i, j, kᴺ, exchange_grid, uᵃᵗ, vᵃᵗ)

    @inbounds begin
        surface_atmos_state.u[i, j, 1] = uᵃᵗ
        surface_atmos_state.v[i, j, 1] = vᵃᵗ
        surface_atmos_state.p[i, j, 1] = pᵃᵗ
        surface_atmos_state.Jʳⁿ[i, j, 1] = Mr
        surface_atmos_state.Jˢⁿ[i, j, 1] = Ms
    end

    update_tracer_states!(i, j, surface_atmos_state, atmos_tracers, tracer_fractional_indices, tracer_time_arguments)
end

@inline @generated function update_tracer_states!(i, j, surface_atmos_state, atmos_tracers::NamedTuple{names},
                                                  fractional_indices, time_arguments) where names
    calls = [:(update_tracer_state!(i, j, surface_atmos_state.$name, atmos_tracers.$name,
                                    fractional_indices.$name, time_arguments.$name)) for name in names]
    return quote
        $(calls...)
        return nothing
    end
end

@inline function update_tracer_state!(i, j, state, tracer, fractional_indices, (time_interpolator, backend, time_indexing))
    fi = get_fractional_index(i, j, fractional_indices.i)
    fj = get_fractional_index(i, j, fractional_indices.j)
    X = FractionalIndices(fi, fj, nothing)
    @inbounds state[i, j, 1] = interpolate(X, time_interpolator, tracer, backend, time_indexing)
    return nothing
end

@inline function update_tracer_state!(i, j, state, tracer::FTS0, fractional_indices, (time_interpolator, backend, time_indexing))
    X = FractionalIndices(nothing, nothing, nothing)
    @inbounds state[1, 1, 1] = interpolate(X, time_interpolator, tracer, backend, time_indexing)
    return nothing
end

@inline update_tracer_state!(i, j, state, ::ConstantField, fractional_indices, time_arguments) = nothing

#####
##### Utility for interpolating tuples of fields
#####

@inline interp_atmos_time_series(::Nothing, X, time, grid, args...) = 0

# Note: assumes loc = (c, c, nothing) (and the third location should not matter.)
@inline interp_atmos_time_series(J::AbstractArray, X::FractionalIndices, time, args...) =
    interpolate(X, time, J, args...)

@inline interp_atmos_time_series(J::AbstractArray, X, time, grid, args...) =
    interpolate(X, time, J, (Center(), Center(), nothing), grid, args...)

@inline interp_atmos_time_series(ΣJ::NamedTuple, args...) =
    interp_atmos_time_series(values(ΣJ), args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...) +
    interp_atmos_time_series(ΣJ[3], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any, <:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...) +
    interp_atmos_time_series(ΣJ[3], args...) +
    interp_atmos_time_series(ΣJ[4], args...)

@inline interp_atmos_time_series(ΣJ::Tuple{<:Any, <:Any, <:Any, <:Any, <:Any}, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2], args...) +
    interp_atmos_time_series(ΣJ[3], args...) +
    interp_atmos_time_series(ΣJ[4], args...) +
    interp_atmos_time_series(ΣJ[5], args...)

@inline interp_atmos_time_series(ΣJ::Tuple, args...) =
    interp_atmos_time_series(ΣJ[1], args...) +
    interp_atmos_time_series(ΣJ[2:end], args...)
