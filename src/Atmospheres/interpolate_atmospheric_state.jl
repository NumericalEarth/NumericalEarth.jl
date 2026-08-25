using Oceananigans.Operators: intrinsic_vector
using Oceananigans.Grids: _node
using Oceananigans.Fields: FractionalIndices, interpolate
using Oceananigans.OutputReaders: cpu_interpolating_time_indices, FTS0

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

    # Extract info for time-interpolation
    u = atmosphere.velocities.u # for example
    atmosphere_times = u.times
    atmosphere_backend = u.backend
    atmosphere_time_indexing = u.time_indexing

    atmosphere_fields = exchanger.state
    space_fractional_indices = exchanger.regridder

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_data = NamedTuple(k=>underlying_data(v) for (k, v) in pairs(atmosphere_fields))

    kernel_parameters = interface_kernel_parameters(grid)

    # Assumption, should be generalized
    ua = atmosphere.velocities.u

    times = ua.times
    time_indexing = ua.time_indexing
    t = clock.time
    time_interpolator = cpu_interpolating_time_indices(arch, times, time_indexing, t)

    launch!(arch, grid, kernel_parameters,
            _interpolate_primary_atmospheric_state!,
            atmosphere_data,
            space_fractional_indices,
            time_interpolator,
            grid,
            atmosphere_velocities,
            atmosphere_tracers,
            atmosphere_pressure,
            rainfall_flux,
            snowfall_flux,
            atmosphere_backend,
            atmosphere_time_indexing)

    # Set ocean barotropic pressure forcing
    #
    # TODO: find a better design for this that doesn't have redundant
    # arrays for the barotropic potential
    potential = forcing_barotropic_potential(coupled_model.ocean)
    ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density

    if !isnothing(potential)
        parent(potential) .= parent(atmosphere_data.p) ./ ρᵒᶜ
    end
end

@inline get_fractional_index(i, j, ::Nothing) = nothing
@inline get_fractional_index(i, j, frac) = @inbounds frac[i, j, 1]

@inline underlying_data(f) = f.data
@inline underlying_data(::ConstantField) = nothing

@kernel function _interpolate_primary_atmospheric_state!(surface_atmos_state,
                                                         space_fractional_indices,
                                                         time_interpolator,
                                                         exchange_grid,
                                                         atmos_velocities,
                                                         atmos_tracers,
                                                         atmos_pressure,
                                                         rainfall_flux,
                                                         snowfall_flux,
                                                         atmos_backend,
                                                         atmos_time_indexing)

    i, j = @index(Global, NTuple)

    ii = space_fractional_indices.i
    jj = space_fractional_indices.j
    fi = get_fractional_index(i, j, ii)
    fj = get_fractional_index(i, j, jj)

    x_itp = FractionalIndices(fi, fj, nothing)
    t_itp = time_interpolator
    atmos_args = (x_itp, t_itp, atmos_backend, atmos_time_indexing)

    uᵃᵗ = interp_atmos_time_series(atmos_velocities.u, atmos_args...)
    vᵃᵗ = interp_atmos_time_series(atmos_velocities.v, atmos_args...)
    pᵃᵗ = interp_atmos_time_series(atmos_pressure,     atmos_args...)

    Mr = interp_atmos_time_series(rainfall_flux, atmos_args...)
    Ms = interp_atmos_time_series(snowfall_flux, atmos_args...)

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

    for (tn, tv) in pairs(atmos_tracers)
        update_tracer_state!(i, j, surface_atmos_state[tn], tv, atmos_args, t_itp)
    end
end

@inline function update_tracer_state!(i, j, state, tracer, atmos_args, t_itp)
    @inbounds state[i, j, 1] = interp_atmos_time_series(tracer, atmos_args...)
    
    return nothing
end

@inline update_tracer_state!(i, j, state, ::ConstantField, atmos_args, t_itp) = nothing
@inline function update_tracer_state!(i, j, state, zero_D_fts::FTS0, atmos_args, t_itp)
    @inbounds state[1, 1, 1] = interpolate(FractionalIndices(1, 1, 1), t_itp, zero_D_fts, atmos_args[3:4]...)

    return nothing
end

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
