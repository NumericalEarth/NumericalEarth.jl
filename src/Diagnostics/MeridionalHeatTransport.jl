import ..EarthSystemModels: EarthSystemModel, checkpoint_auxiliary_state, restore_auxiliary_state!

"""
    Meridional_Heat_Transport(coupled_model)

Return meridional heat transport diagnosed from the OHC anomaly budget:

`MHT = CumulativeIntegral((OHC - OHC₀) - ∫ₓ(∫Q dt), dims=(2))`,

where `OHC = ρₒ cₚ ∫ₓ∫z T`, `OHC₀` is the first sampled OHC profile, and
`Q` is `"heat_flux"` from `InterfaceFluxOutputs`.
"""
mutable struct MeridionalHeatTransportState
    initial_ohc::Any
    cumulative_heat_flux::Any
    last_time::Float64
end

const meridional_heat_transport_states = IdDict{Any, MeridionalHeatTransportState}()

allocate_storage_like(field) = Field(instantiated_location(field), field.grid; indices=indices(field))

"""
    reset_meridional_heat_transport_state!(coupled_model)

Clear cached MHT state (`OHC₀`, cumulative heat flux, last time) for `coupled_model`.
Call this before restarting from a checkpoint or when reusing a model object.
"""
function reset_meridional_heat_transport_state!(coupled_model)
    pop!(meridional_heat_transport_states, coupled_model, nothing)
    return nothing
end

function initialize_mht_state!(coupled_model, heat_flux_field, ohc_field, time)
    initial_ohc = allocate_storage_like(ohc_field)
    set!(initial_ohc, ohc_field)

    cumulative_heat_flux = allocate_storage_like(heat_flux_field)
    set!(cumulative_heat_flux, 0)

    state = MeridionalHeatTransportState(initial_ohc, cumulative_heat_flux, time)
    meridional_heat_transport_states[coupled_model] = state
    return state
end

function current_heat_flux_field(coupled_model)
    flux_outputs = InterfaceFluxOutputs(coupled_model;
                                        isolate_sea_ice=false,
                                        units=:physical,
                                        reference_salinity=35)

    heat_flux_raw = haskey(flux_outputs, "heat_flux") ? flux_outputs["heat_flux"] : flux_outputs[:heat_flux]
    heat_flux_field = Field(heat_flux_raw)
    compute!(heat_flux_field)
    return heat_flux_field
end

function Meridional_Heat_Transport(coupled_model; ρₒ=1035.0, cₚ=3991.86795711963)
    ocean = coupled_model.ocean
    heat_flux_field = current_heat_flux_field(coupled_model)
    ohc_field = Field(ρₒ * cₚ * Integral(ocean.model.tracers.T, dims=(1, 3)))
    compute!(ohc_field)

    model_time = Float64(ocean.model.clock.time)
    state = get(meridional_heat_transport_states, coupled_model, nothing)
    state === nothing && (state = initialize_mht_state!(coupled_model, heat_flux_field, ohc_field, model_time))

    Δt = max(0.0, model_time - state.last_time)
    if Δt == 0.0 && ocean.model.clock.iteration > 0
        Δt = Float64(ocean.model.clock.Δt)
    end
    state.last_time = model_time

    set!(state.cumulative_heat_flux, state.cumulative_heat_flux + Δt * heat_flux_field)

    Δohc = ohc_field - state.initial_ohc
    flux_int = Integral(state.cumulative_heat_flux, dims=(1))

    return CumulativeIntegral(Δohc - flux_int, dims=(2))
end

function checkpoint_auxiliary_state(coupled_model::EarthSystemModel)
    state = get(meridional_heat_transport_states, coupled_model, nothing)
    state === nothing && return nothing

    return (
        meridional_heat_transport = (
            initial_ohc = Array(interior(state.initial_ohc)),
            cumulative_heat_flux = Array(interior(state.cumulative_heat_flux)),
            last_time = state.last_time
        ),
    )
end

function restore_auxiliary_state!(coupled_model::EarthSystemModel, auxiliary_state)
    auxiliary_state === nothing && return nothing
    hasproperty(auxiliary_state, :meridional_heat_transport) || return nothing

    mht_state = auxiliary_state.meridional_heat_transport
    mht_state === nothing && return nothing

    heat_flux_field = current_heat_flux_field(coupled_model)
    ohc_template = Field(Integral(coupled_model.ocean.model.tracers.T, dims=(1, 3)))
    compute!(ohc_template)

    reset_meridional_heat_transport_state!(coupled_model)
    state = initialize_mht_state!(coupled_model, heat_flux_field, ohc_template, Float64(mht_state.last_time))
    set!(state.initial_ohc, mht_state.initial_ohc)
    set!(state.cumulative_heat_flux, mht_state.cumulative_heat_flux)
    state.last_time = Float64(mht_state.last_time)
    return nothing
end
