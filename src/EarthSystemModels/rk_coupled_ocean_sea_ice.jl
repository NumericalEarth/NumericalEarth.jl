using ClimaSeaIce
using Oceananigans
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, rk_substep!, update_state!, cache_current_fields!
using Oceananigans.Simulations: TimeStepCallsite, TendencyCallsite, UpdateStateCallsite, run_diagnostic!, write_output!

const RKSI = SeaIceModel{<:Any, <:Any, <:Any, <:SplitRungeKuttaTimeStepper}
const RKHM = HydrostaticFreeSurfaceModel{<:SplitRungeKuttaTimeStepper}
const RKCM = EarthSystemModel{<:RKSI, <:Any, <:RKHM}

const ModelCallsite = Union{TendencyCallsite, UpdateStateCallsite}

function time_step_ocean_sea_ice_components!(coupled_model::RKCM, ocean::RKHM, sea_ice::RKSI, Δt)

    ocean_callbacks = Tuple(cb for cb in values(ocean.callbacks) if cb.callsite isa ModelCallsite)
    sea_ice_callbacks = Tuple(cb for cb in values(sea_ice.callbacks) if cb.callsite isa ModelCallsite)

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

    # Finalize step
    tick!(coupled_model.clock, Δt)
    tick!(ocean.model.clock, Δt)
    tick!(sea_ice.model.clock, Δt)

    # Ocean Callbacks and callback-like things
    for diag in values(ocean.diagnostics)
        diag.schedule(ocean.model) && run_diagnostic!(diag, ocean.model)
    end

    for callback in values(ocean.callbacks)
        callback.callsite isa TimeStepCallsite && callback.schedule(ocean.model) && callback(ocean)
    end

    for writer in values(ocean.output_writers)
        writer.schedule(ocean.model) && write_output!(writer, ocean)
    end

    # Sea Ice Callbacks and callback-like things
    for diag in values(sea_ice.diagnostics)
        diag.schedule(ocean.model) && run_diagnostic!(diag, sea_ice.model)
    end

    for callback in values(sea_ice.callbacks)
        callback.callsite isa TimeStepCallsite && callback.schedule(sea_ice.model) && callback(sea_ice)
    end

    for writer in values(sea_ice.output_writers)
        writer.schedule(ocean.model) && write_output!(writer, sea_ice)
    end

    return nothing
end
