import ..EarthSystemModels: EarthSystemModel, checkpoint_auxiliary_state, restore_auxiliary_state!,
                            reference_density, heat_capacity

mutable struct MeridionalHeatTransportState
    cumulative_ohc_tendency::Any
    cumulative_heat_flux::Any
    last_time::Float64
    last_iteration::Int
end

const meridional_heat_transport_states = IdDict{Any, MeridionalHeatTransportState}()

"""
    meridional_heat_transport(esm::EarthSystemModel; method=:ohc_tendency, reference_temperature=0.0)

Return the meridional heat transport for the coupled `esm` using either of two methods:

Keyword Arguments
=================

* `method`: Determines the method the computation is done. Options are:
  1. `method = :ohc_tendency` (default):

     ```math
     ∫_{y_S}^y dy [(∫dt ∫dx ∫dz ρᵒᶜ cᵒᶜ ∂ₜT) - (∫dt ∫dx Q)]
     ```

     where `Q` is the net heat flux into the ocean.

  2. `method = :vt_instantaneous`:

     ```math
     ρᵒᶜ cᵒᶜ ∫dx ∫dz v (T - T_{\\rm reference})
     ```

  Above, ``ρᵒᶜ`` and ``cᵒᶜ`` are the ocean reference density and heat capacity respectively
  and they are inferred from the ocean component, `esm.ocean`.

* `reference_temperature`: The reference temperature used for `:vt_instantaneous` method.
"""
function meridional_heat_transport(esm::EarthSystemModel; method=:ohc_tendency, reference_temperature=0.0)
    if method === :ohc_tendency
        return ohc_tendency_mht(esm)
    elseif method === :vt_instantaneous
        return instantaneous_vt_mht(esm, reference_temperature)
    else
        throw(ArgumentError("Unknown method=$(repr(method)). Supported methods are :ohc_tendency and :vt_instantaneous."))
    end
end

allocate_storage_like(field) = Field(instantiated_location(field), field.grid; indices=indices(field))

"""
    reset_meridional_heat_transport_state!(esm)

Clear cached for the meridional heat transport state
(cumulative OHC tendency, cumulative heat flux, and clock metadata)
for a coupled `esm`. This method should be called before
restarting from a checkpoint or when reusing a model object.
"""
function reset_meridional_heat_transport_state!(esm)
    pop!(meridional_heat_transport_states, esm, nothing)
    return nothing
end

function initialize_mht_state!(esm, heat_flux_field, ohc_tendency_field, time, iteration)
    cumulative_ohc_tendency = allocate_storage_like(ohc_tendency_field)
    set!(cumulative_ohc_tendency, 0)

    cumulative_heat_flux = allocate_storage_like(heat_flux_field)
    set!(cumulative_heat_flux, 0)

    state = MeridionalHeatTransportState(cumulative_ohc_tendency, cumulative_heat_flux, time, iteration)
    meridional_heat_transport_states[esm] = state
    return state
end

function current_heat_flux_field(esm)
    flux_outputs = InterfaceFluxOutputs(esm;
                                        isolate_sea_ice=false,
                                        units=:physical,
                                        reference_salinity=35)

    heat_flux_raw = haskey(flux_outputs, "heat_flux") ? flux_outputs["heat_flux"] : flux_outputs[:heat_flux]
    heat_flux_field = Field(heat_flux_raw)
    compute!(heat_flux_field)
    return heat_flux_field
end

function ohc_tendency_mht(esm)
    ρᵒᶜ = reference_density(esm.ocean)
    cᵒᶜ = heat_capacity(esm.ocean)
    heat_flux = current_heat_flux_field(esm)
    T_tendency = esm.ocean.model.timestepper.Gⁿ.T

    ohc_tendency = Field(ρᵒᶜ * cᵒᶜ * Integral(T_tendency, dims=(1, 3)))
    compute!(ohc_tendency)

    model_time = esm.ocean.model.clock.time
    model_iteration = esm.ocean.model.clock.iteration
    state = get(meridional_heat_transport_states, esm, nothing)
    state === nothing &&
        (state = initialize_mht_state!(esm, heat_flux, ohc_tendency, model_time, model_iteration))

    if model_iteration != state.last_iteration
        Δt = max(0.0, model_time - state.last_time)
        if Δt == 0.0 && model_iteration > 0
            Δt = ocean.model.clock.Δt
        end

        set!(state.cumulative_ohc_tendency, state.cumulative_ohc_tendency + Δt * ohc_tendency_field)
        set!(state.cumulative_heat_flux, state.cumulative_heat_flux + Δt * heat_flux_field)
        state.last_time = model_time
        state.last_iteration = model_iteration
    end

    ∫heat_flux = Integral(state.cumulative_heat_flux, dims=1)

    return CumulativeIntegral(state.cumulative_ohc_tendency - ∫heat_flux, dims=(2))
end

function instantaneous_vt_mht(esm, reference_temperature)
    ρᵒᶜ = reference_density(esm.ocean)
    cᵒᶜ = heat_capacity(esm.ocean)
    T = esm.ocean.model.tracers.T
    v = esm.ocean.model.velocities.v
    return ρᵒᶜ * cᵒᶜ * Integral(v * (T - reference_temperature), dims=(1, 3))
end

function checkpoint_auxiliary_state(esm::EarthSystemModel)
    state = get(meridional_heat_transport_states, esm, nothing)
    state === nothing && return nothing

    return (
        meridional_heat_transport = (
            cumulative_ohc_tendency = Array(interior(state.cumulative_ohc_tendency)),
            cumulative_heat_flux = Array(interior(state.cumulative_heat_flux)),
            last_time = state.last_time,
            last_iteration = state.last_iteration
        ),
    )
end

function restore_auxiliary_state!(esm::EarthSystemModel, auxiliary_state)
    auxiliary_state === nothing && return nothing
    hasproperty(auxiliary_state, :meridional_heat_transport) || return nothing

    mht_state = auxiliary_state.meridional_heat_transport
    mht_state === nothing && return nothing

    heat_flux_field = current_heat_flux_field(esm)
    ohc_tendency_template = Field(Integral(esm.ocean.model.timestepper.Gⁿ.T, dims=(1, 3)))
    compute!(ohc_tendency_template)

    reset_meridional_heat_transport_state!(esm)
    state = initialize_mht_state!(esm,
                                  heat_flux_field,
                                  ohc_tendency_template,
                                  Float64(mht_state.last_time),
                                  Int(mht_state.last_iteration))
    set!(state.cumulative_ohc_tendency, mht_state.cumulative_ohc_tendency)
    set!(state.cumulative_heat_flux, mht_state.cumulative_heat_flux)
    state.last_time = Float64(mht_state.last_time)
    state.last_iteration = Int(mht_state.last_iteration)
    return nothing
end
