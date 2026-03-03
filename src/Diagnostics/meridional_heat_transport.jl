using ..EarthSystemModels: reference_density, heat_capacity

import ..EarthSystemModels: EarthSystemModel, checkpoint_auxiliary_state, restore_auxiliary_state!

struct OceanHeatContentTendencyMethod end
struct MeridionalHeatFluxMethod end

mutable struct MeridionalHeatTransportState{FT, FL}
    cumulative_ohc_tendency :: FL
    cumulative_heat_flux :: FL
    last_time :: FT
    last_iteration :: Int
end

const meridional_heat_transport_states = IdDict{Any, MeridionalHeatTransportState}()

"""
    meridional_heat_transport(esm::EarthSystemModel, OceanHeatContentTendencyMethod();
                              reference_temperature=0.0)

Return the meridional heat transport for the coupled `esm` using either of two methods:

Arguments
=========

* `esm`: An EarthSystemModel.

* `OceanHeatContentTendencyMethod()` or `MeridionalHeatFluxMethod()` denoting the method
  that the meridional heat transport is computed.

  1. For `OceanHeatContentTendencyMethod()` (default), the meridional heat transport
     is computed via:

     ```math
     ∫_{y_S}^y dy [(∫dt ∫dx ∫dz ρᵒᶜ cᵒᶜ ∂ₜT) - (∫dt ∫dx 𝒬)]
     ```

     where ``y_S`` is the Southern-most latitude of the domain and ``𝒬`` is the
     net heat flux into the ocean.

  2. For `MeridionalHeatFluxMethod()`, the meridional heat transport is computed via:

     ```math
     ρᵒᶜ cᵒᶜ ∫dx ∫dz v (T - T_{\\rm ref})
     ```

  Above, ``ρᵒᶜ`` and ``cᵒᶜ`` are the ocean reference density and heat capacity respectively
  and they are inferred from the ocean component, `esm.ocean`, and ``T_{\\rm ref}`` is a
  reference temperature.


Keyword Arguments
=================

* `reference_temperature`: The reference temperature used for `:vt_instantaneous` method.
"""
function meridional_heat_transport(esm::EarthSystemModel, ::OceanHeatContentTendencyMethod;
                                   reference_temperature=0.0)
    return meridional_heat_transport_via_ocean_heat_content_tendency(esm)
end

function meridional_heat_transport(esm::EarthSystemModel, ::MeridionalHeatFluxMethod;
                                   reference_temperature=0.0)
    return meridional_heat_transport_via_meridional_heat_flux(esm; reference_temperature)
end

meridional_heat_transport(esm::EarthSystemModel) = meridional_heat_transport(esm, OceanHeatContentTendencyMethod())

meridional_heat_transport(esm::EarthSystemModel, method; reference_temperature=0.0) =
    throw(ArgumentError("Unknown method $(method); choose one of OceanHeatContentTendencyMethod() or MeridionalHeatFluxMethod()."))

"""
    reset_meridional_heat_transport_state!(esm)

Clear cached for the meridional heat transport state (cumulative ocean heat
content tendency, cumulative heat flux, and clock metadata) for a coupled
`esm`. This method should be called before restarting from a checkpoint or
when reusing a model object.
"""
function reset_meridional_heat_transport_state!(esm)
    pop!(meridional_heat_transport_states, esm, nothing)
    return nothing
end

function initialize_mht_state!(esm, heat_flux, ohc_tendency, time, iteration)
    cumulative_ohc_tendency = deepcopy(ohc_tendency)
    set!(cumulative_ohc_tendency, 0)

    cumulative_heat_flux = deepcopy(heat_flux)
    set!(cumulative_heat_flux, 0)

    FT = eltype(esm)
    FL = typeof(cumulative_ohc_tendency)
    state = MeridionalHeatTransportState{FT, FL}(cumulative_ohc_tendency,
                                                 cumulative_heat_flux,
                                                 time,
                                                 iteration)
    meridional_heat_transport_states[esm] = state
    return state
end

function meridional_heat_transport_via_ocean_heat_content_tendency(esm)
    ρᵒᶜ = reference_density(esm.ocean)
    cᵒᶜ = heat_capacity(esm.ocean)
    heat_flux = net_ocean_heat_flux(esm)
    ∂T_∂t = esm.ocean.model.timestepper.Gⁿ.T

    ohc_tendency = Field(Integral(ρᵒᶜ * cᵒᶜ * ∂T_∂t, dims=(1, 3)))
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

function meridional_heat_transport_via_meridional_heat_flux(esm; reference_temperature)
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

    heat_flux = net_ocean_heat_flux(esm)
    ∂T_∂t = esm.ocean.model.timestepper.Gⁿ.T
    ohc_tendency = Field(Integral(∂T_∂t, dims=(1, 3)))
    compute!(ohc_tendency)

    reset_meridional_heat_transport_state!(esm)
    state = initialize_mht_state!(esm,
                                  heat_flux,
                                  ohc_tendency,
                                  mht_state.last_time,
                                  mht_state.last_iteration)
    set!(state.cumulative_ohc_tendency, mht_state.cumulative_ohc_tendency)
    set!(state.cumulative_heat_flux, mht_state.cumulative_heat_flux)
    state.last_time = mht_state.last_time
    state.last_iteration = mht_state.last_iteration
    return nothing
end
