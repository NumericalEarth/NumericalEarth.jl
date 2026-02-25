using Oceananigans.Operators: intrinsic_vector
using Oceananigans.Grids: _node
using Oceananigans.Fields: FractionalIndices, interpolate
using Oceananigans.OutputReaders: TimeInterpolator

using NumericalEarth.Oceans: forcing_barotropic_potential

"""Interpolate the atmospheric state onto the ocean / sea-ice grid."""
function interpolate_state!(exchanger, grid, atmosphere::PrescribedAtmosphere, coupled_model)
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

    atmosphere_tracers = (T = atmosphere.tracers.T.data,
                          q = atmosphere.tracers.q.data)

    ℐꜜˢʷ = atmosphere.downwelling_radiation.shortwave
    ℐꜜˡʷ = atmosphere.downwelling_radiation.longwave
    downwelling_radiation = (shortwave=ℐꜜˢʷ.data, longwave=ℐꜜˡʷ.data)
    freshwater_flux = map(ϕ -> ϕ.data, atmosphere.freshwater_flux)
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
    atmosphere_data = (u    = atmosphere_fields.u.data,
                       v    = atmosphere_fields.v.data,
                       T    = atmosphere_fields.T.data,
                       p    = atmosphere_fields.p.data,
                       q    = atmosphere_fields.q.data,
                       ℐꜜˢʷ = atmosphere_fields.ℐꜜˢʷ.data,
                       ℐꜜˡʷ = atmosphere_fields.ℐꜜˡʷ.data,
                       Jᶜ   = atmosphere_fields.Jᶜ.data)

    kernel_parameters = interface_kernel_parameters(grid)

    # Assumption, should be generalized
    ua = atmosphere.velocities.u

    times = ua.times
    time_indexing = ua.time_indexing
    t = clock.time
    time_interpolator = TimeInterpolator(ua.time_indexing, times, clock.time)

    launch!(arch, grid, kernel_parameters,
            _interpolate_primary_atmospheric_state!,
            atmosphere_data,
            space_fractional_indices,
            time_interpolator,
            grid,
            atmosphere_velocities,
            atmosphere_tracers,
            atmosphere_pressure,
            downwelling_radiation,
            freshwater_flux,
            atmosphere_backend,
            atmosphere_time_indexing)

    # Separately interpolate the auxiliary freshwater fluxes, which may
    # live on a different grid than the primary fluxes and atmospheric state.
    auxiliary_freshwater_flux = atmosphere.auxiliary_freshwater_flux
    interpolated_prescribed_freshwater_flux = atmosphere_data.Jᶜ

    if !isnothing(auxiliary_freshwater_flux)
        # TODO: do not assume that `auxiliary_freshater_flux` is a tuple
        auxiliary_data = map(ϕ -> ϕ.data, auxiliary_freshwater_flux)

        first_auxiliary_flux    = first(auxiliary_freshwater_flux)
        auxiliary_grid          = first_auxiliary_flux.grid
        auxiliary_times         = first_auxiliary_flux.times
        auxiliary_backend       = first_auxiliary_flux.backend
        auxiliary_time_indexing = first_auxiliary_flux.time_indexing

        launch!(arch, grid, kernel_parameters,
                _interpolate_auxiliary_freshwater_flux!,
                interpolated_prescribed_freshwater_flux,
                grid,
                clock,
                auxiliary_data,
                auxiliary_grid,
                auxiliary_times,
                auxiliary_backend,
                auxiliary_time_indexing)
    end

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

@kernel function _interpolate_primary_atmospheric_state!(surface_atmos_state,
                                                         space_fractional_indices,
                                                         time_interpolator,
                                                         exchange_grid,
                                                         atmos_velocities,
                                                         atmos_tracers,
                                                         atmos_pressure,
                                                         downwelling_radiation,
                                                         prescribed_freshwater_flux,
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
    Tᵃᵗ = interp_atmos_time_series(atmos_tracers.T,    atmos_args...)
    qᵃᵗ = interp_atmos_time_series(atmos_tracers.q,    atmos_args...)
    pᵃᵗ = interp_atmos_time_series(atmos_pressure,     atmos_args...)

    ℐꜜˢʷ = interp_atmos_time_series(downwelling_radiation.shortwave, atmos_args...)
    ℐꜜˡʷ = interp_atmos_time_series(downwelling_radiation.longwave,  atmos_args...)

    # Usually precipitation
    Mh = interp_atmos_time_series(prescribed_freshwater_flux, atmos_args...)

    # Convert atmosphere velocities (usually defined on a latitude-longitude grid) to
    # the frame of reference of the native grid
    kᴺ = size(exchange_grid, 3) # index of the top ocean cell
    uᵃᵗ, vᵃᵗ = intrinsic_vector(i, j, kᴺ, exchange_grid, uᵃᵗ, vᵃᵗ)

    @inbounds begin
        surface_atmos_state.u[i, j, 1] = uᵃᵗ
        surface_atmos_state.v[i, j, 1] = vᵃᵗ
        surface_atmos_state.T[i, j, 1] = Tᵃᵗ
        surface_atmos_state.p[i, j, 1] = pᵃᵗ
        surface_atmos_state.q[i, j, 1] = qᵃᵗ
        surface_atmos_state.ℐꜜˢʷ[i, j, 1] = ℐꜜˢʷ
        surface_atmos_state.ℐꜜˡʷ[i, j, 1] = ℐꜜˡʷ
        surface_atmos_state.Jᶜ[i, j, 1] = Mh
    end
end

@kernel function _interpolate_auxiliary_freshwater_flux!(freshwater_flux,
                                                         interface_grid,
                                                         clock,
                                                         auxiliary_freshwater_flux,
                                                         auxiliary_grid,
                                                         auxiliary_times,
                                                         auxiliary_backend,
                                                         auxiliary_time_indexing)

    i, j = @index(Global, NTuple)
    kᴺ = size(interface_grid, 3) # index of the top ocean cell

    @inbounds begin
        X = _node(i, j, kᴺ + 1, interface_grid, Center(), Center(), Face())
        time = Time(clock.time)
        Mr = interp_atmos_time_series(auxiliary_freshwater_flux, X, time,
                                      auxiliary_grid,
                                      auxiliary_times,
                                      auxiliary_backend,
                                      auxiliary_time_indexing)

        freshwater_flux[i, j, 1] += Mr
    end
end

#####
##### Utility for interpolating tuples of fields
#####

# Note: assumes loc = (c, c, nothing) (and the third location should not matter.)
@inline interp_atmos_time_series(J::AbstractArray, x_itp::FractionalIndices, t_itp, args...) =
    interpolate(x_itp, t_itp, J, args...)

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
