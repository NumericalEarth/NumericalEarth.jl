using ClimaSeaIce
using Oceananigans
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, rk_substep!, update_state!, cache_current_fields!

RKSI = SeaIceModel{<:Any, <:Any, <:Any, <:SplitRungeKuttaTimeStepper}
RKHM = HydrostaticFreeSurfaceModel{<:SplitRungeKuttaTimeStepper}
TCM  = EarthSystemModel{<:RKSI, <:Any, <:RKHM}

function time_step_ocean_sea_ice_components!(coupled_model::TCM, ocean::RKHM, sea_ice::RKSI, Δt)

    ocean_callbacks = ocean.callbacks
    sea_ice_callbacks = sea_ice.callbacks

    if coupled_model.clock.iteration == 0
        update_state!(ocean.model, ocean_callbacks)
        update_state!(sea_ice.model, sea_ice_callbacks)
    end

    cache_current_fields!(ocean.model)
    cache_current_fields!(sea_ice.model)
    grid = ocean.model.grid

    ####
    #### Loop over the stages
    ####

    for (stage, β) in enumerate(ocean.model.timestepper.β)
        # Update the clock stage
        ocean.model.clock.stage = stage

        # Perform the substep
        Δτ = Δt / β
        rk_substep!(ocean.model, Δτ, ocean_callbacks)
        rk_substep!(sea_ice.model, Δτ, sea_ice_callbacks)

        # Update the state
        update_state!(ocean.model, ocean_callbacks)
        update_state!(sea_ice.model, sea_ice_callbacks)

        compute_sea_ice_ocean_fluxes!(coupled_model)
        update_net_fluxes!(coupled_model, ocean)
        update_net_fluxes!(coupled_model, sea_ice)
    end

    # TODO: 
    # add here all the callbacks infrastructure from the ocean and
    # sea ice components which now lives in the `time_step!(simulation)`
    # framework

    # Finalize step
    tick!(coupled_model.clock, Δt)
    tick!(ocean.model.clock, Δt)
    tick!(sea_ice.model.clock, Δt)
end
