using Oceananigans.Grids: _node

using ..Atmospheres: interp_atmos_time_series

"""Interpolate the land state (freshwater fluxes) onto the exchange grid."""
function EarthSystemModels.interpolate_state!(exchanger, grid, land::PrescribedLand, coupled_model)
    arch = architecture(grid)
    clock = coupled_model.clock
    land_freshwater_flux = exchanger.state.freshwater_flux

    # Zero the land freshwater flux before accumulating
    fill!(land_freshwater_flux, 0)

    freshwater_flux = land.freshwater_flux
    freshwater_data = map(ϕ -> ϕ.data, freshwater_flux)

    first_flux = first(freshwater_flux)
    land_grid = first_flux.grid
    land_times = first_flux.times
    land_backend = first_flux.backend
    land_time_indexing = first_flux.time_indexing

    kernel_parameters = interface_kernel_parameters(grid)

    launch!(arch, grid, kernel_parameters,
            _interpolate_land_freshwater_flux!,
            land_freshwater_flux.data,
            grid,
            clock,
            freshwater_data,
            land_grid,
            land_times,
            land_backend,
            land_time_indexing)

    iceberg_freshwater_flux = exchanger.state.iceberg_freshwater_flux
    fill!(iceberg_freshwater_flux, 0)

    if haskey(freshwater_data, :icebergs)
        launch!(arch, grid, kernel_parameters,
                _interpolate_land_freshwater_flux!,
                iceberg_freshwater_flux.data,
                grid,
                clock,
                (freshwater_data.icebergs,),
                land_grid,
                land_times,
                land_backend,
                land_time_indexing)
    end

    return nothing
end

@kernel function _interpolate_land_freshwater_flux!(land_freshwater_flux,
                                                    interface_grid,
                                                    clock,
                                                    freshwater_data,
                                                    land_grid,
                                                    land_times,
                                                    land_backend,
                                                    land_time_indexing)

    i, j = @index(Global, NTuple)
    kᴺ = size(interface_grid, 3)

    @inbounds begin
        X = _node(i, j, kᴺ + 1, interface_grid, Center(), Center(), Face())
        time = Time(clock.time)
        Mr = interp_atmos_time_series(freshwater_data, X, time,
                                      land_grid,
                                      land_times,
                                      land_backend,
                                      land_time_indexing)

        land_freshwater_flux[i, j, 1] = Mr
    end
end
