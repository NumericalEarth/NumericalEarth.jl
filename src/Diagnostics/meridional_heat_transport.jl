import ..EarthSystemModels: EarthSystemModel, checkpoint_auxiliary_state, restore_auxiliary_state!,
                             reference_density, heat_capacity

"""
    Meridional_Heat_Transport(coupled_model; method=:ohc_tendency, T_ref=0.0)

Compute meridional heat transport with one of two methods:

1. `method = :ohc_tendency` (default):

   `MHT = CumulativeIntegral((∫ ρₒ cₚ ∫ₓ∫z ∂ₜT dt) - ∫ₓ(∫Q dt), dims=(2))`,
   where `Q` is `"heat_flux"` from `InterfaceFluxOutputs`.

2. `method = :vt_instantaneous`:

   `MHT = ρₒ cₚ ∫ₓ∫z v * (T - T_ref)`

`ρₒ` and `cₚ` are inferred from `coupled_model.ocean` via
`reference_density` and `heat_capacity`.
"""
mutable struct MeridionalHeatTransportState
    cumulative_ohc_tendency::Any
    cumulative_heat_flux::Any
    last_time::Float64
    last_iteration::Int
end

const meridional_heat_transport_states = IdDict{Any, MeridionalHeatTransportState}()

allocate_storage_like(field) = Field(instantiated_location(field), field.grid; indices=indices(field))

"""
    reset_meridional_heat_transport_state!(coupled_model)

Clear cached MHT state (cumulative OHC tendency, cumulative heat flux, and clock metadata)
for `coupled_model`.
Call this before restarting from a checkpoint or when reusing a model object.
"""
function reset_meridional_heat_transport_state!(coupled_model)
    pop!(meridional_heat_transport_states, coupled_model, nothing)
    return nothing
end

function initialize_mht_state!(coupled_model, heat_flux_field, ohc_tendency_field, time, iteration)
    cumulative_ohc_tendency = allocate_storage_like(ohc_tendency_field)
    set!(cumulative_ohc_tendency, 0)

    cumulative_heat_flux = allocate_storage_like(heat_flux_field)
    set!(cumulative_heat_flux, 0)

    state = MeridionalHeatTransportState(cumulative_ohc_tendency, cumulative_heat_flux, time, iteration)
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

@inline mht_constants(coupled_model) = reference_density(coupled_model.ocean), heat_capacity(coupled_model.ocean)

function ohc_tendency_mht(coupled_model, ρₒ, cₚ)
    ocean = coupled_model.ocean
    heat_flux_field = current_heat_flux_field(coupled_model)
    ohc_tendency_field = Field(ρₒ * cₚ * Integral(ocean.model.timestepper.Gⁿ.T, dims=(1, 3)))
    compute!(ohc_tendency_field)

    model_time = Float64(ocean.model.clock.time)
    model_iteration = Int(ocean.model.clock.iteration)
    state = get(meridional_heat_transport_states, coupled_model, nothing)
    state === nothing && (state = initialize_mht_state!(coupled_model, heat_flux_field, ohc_tendency_field, model_time, model_iteration))

    if model_iteration != state.last_iteration
        Δt = max(0.0, model_time - state.last_time)
        if Δt == 0.0 && model_iteration > 0
            Δt = Float64(ocean.model.clock.Δt)
        end

        set!(state.cumulative_ohc_tendency, state.cumulative_ohc_tendency + Δt * ohc_tendency_field)
        set!(state.cumulative_heat_flux, state.cumulative_heat_flux + Δt * heat_flux_field)
        state.last_time = model_time
        state.last_iteration = model_iteration
    end

    flux_int = Integral(state.cumulative_heat_flux, dims=(1))

    return CumulativeIntegral(state.cumulative_ohc_tendency - flux_int, dims=(2))
end

function instantaneous_vt_mht(coupled_model, ρₒ, cₚ, T_ref)
    ocean_model = coupled_model.ocean.model
    return ρₒ * cₚ * Integral(ocean_model.velocities.v * (ocean_model.tracers.T - T_ref), dims=(1, 3))
end

function Meridional_Heat_Transport(coupled_model; method=:ohc_tendency, T_ref=0.0)
    ρₒ, cₚ = mht_constants(coupled_model)

    if method === :ohc_tendency
        return ohc_tendency_mht(coupled_model, ρₒ, cₚ)
    elseif method === :vt_instantaneous
        return instantaneous_vt_mht(coupled_model, ρₒ, cₚ, T_ref)
    else
        throw(ArgumentError("Unknown MHT method=$(repr(method)). Supported methods are :ohc_tendency and :vt_instantaneous."))
    end
end

function checkpoint_auxiliary_state(coupled_model::EarthSystemModel)
    state = get(meridional_heat_transport_states, coupled_model, nothing)
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

function restore_auxiliary_state!(coupled_model::EarthSystemModel, auxiliary_state)
    auxiliary_state === nothing && return nothing
    hasproperty(auxiliary_state, :meridional_heat_transport) || return nothing

    mht_state = auxiliary_state.meridional_heat_transport
    mht_state === nothing && return nothing

    heat_flux_field = current_heat_flux_field(coupled_model)
    ohc_tendency_template = Field(Integral(coupled_model.ocean.model.timestepper.Gⁿ.T, dims=(1, 3)))
    compute!(ohc_tendency_template)

    reset_meridional_heat_transport_state!(coupled_model)
    state = initialize_mht_state!(coupled_model,
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
