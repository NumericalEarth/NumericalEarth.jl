using NumericalEarth.Atmospheres: interp_atmos_time_series

"""Interpolate the prescribed downwelling radiation onto the exchange grid."""
function interpolate_state!(exchanger, grid, radiation::PrescribedRadiation, coupled_model)
    arch = architecture(grid)
    clock = coupled_model.clock

    ℐꜜˢʷ = radiation.downwelling_shortwave
    ℐꜜˡʷ = radiation.downwelling_longwave
    downwelling = (shortwave = ℐꜜˢʷ.data, longwave = ℐꜜˡʷ.data)

    radiation_times = ℐꜜˢʷ.times
    radiation_backend = ℐꜜˢʷ.backend
    radiation_time_indexing = ℐꜜˢʷ.time_indexing

    space_fractional_indices = exchanger.regridder
    state = exchanger.state
    state_data = (ℐꜜˢʷ = state.ℐꜜˢʷ.data,
                  ℐꜜˡʷ = state.ℐꜜˡʷ.data)

    t = clock.time
    time_interpolator = cpu_interpolating_time_indices(arch, radiation_times,
                                                       radiation_time_indexing, t)

    kernel_parameters = interface_kernel_parameters(grid)

    launch!(arch, grid, kernel_parameters,
            _interpolate_radiation_state!,
            state_data,
            space_fractional_indices,
            time_interpolator,
            grid,
            downwelling,
            radiation_backend,
            radiation_time_indexing)

    return nothing
end

@inline get_fractional_index(i, j, ::Nothing) = nothing
@inline get_fractional_index(i, j, frac) = @inbounds frac[i, j, 1]

@kernel function _interpolate_radiation_state!(state,
                                               space_fractional_indices,
                                               time_interpolator,
                                               exchange_grid,
                                               downwelling,
                                               rad_backend,
                                               rad_time_indexing)

    i, j = @index(Global, NTuple)

    ii = space_fractional_indices.i
    jj = space_fractional_indices.j
    fi = get_fractional_index(i, j, ii)
    fj = get_fractional_index(i, j, jj)

    x_itp = FractionalIndices(fi, fj, nothing)
    t_itp = time_interpolator
    args = (x_itp, t_itp, rad_backend, rad_time_indexing)

    ℐꜜˢʷ = interp_atmos_time_series(downwelling.shortwave, args...)
    ℐꜜˡʷ = interp_atmos_time_series(downwelling.longwave,  args...)

    @inbounds begin
        state.ℐꜜˢʷ[i, j, 1] = ℐꜜˢʷ
        state.ℐꜜˡʷ[i, j, 1] = ℐꜜˡʷ
    end
end
