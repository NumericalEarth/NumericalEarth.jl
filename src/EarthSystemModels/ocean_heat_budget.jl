"""
    OceanHeatBudget

Coupled ocean heat-budget fields on the native ocean grid.

`residual` contains the completed column budget

```math
Q_{surface} + \\partial_t H - Q_{radiative} - Q_{freshwater},
```

for the most recently completed coupling interval. Here `freshwater_heat_flux`
is the heat carried into the ocean by freshwater at the ocean surface
temperature. All fields have units of W m⁻².

The fields are ordinary Oceananigans fields, so output schedules apply to them
in the usual way. For example, a conservatively remapped daily mean can be
written with

```julia
budget = esm.interfaces.budgets.ocean_heat
heat_budget = RegriddedOperation(budget.residual, destination_grid)
outputs = (; heat_budget)

writer = JLD2Writer(esm, outputs;
                    schedule = AveragedTimeInterval(1day),
                    filename = "ocean_heat_budget")
```
"""
struct OceanHeatBudget{H, P, T, S, R, A, F, B}
    heat_content :: H
    previous_heat_content :: P
    tendency :: T
    surface_flux :: S
    radiative_heat_flux :: R
    applied_radiative_heat_flux :: A
    freshwater_heat_flux :: F
    residual :: B
end

ocean_heat_budget(ocean) = nothing

# Implemented alongside the other public interface-flux diagnostics in
# `Diagnostics/interface_fluxes.jl`.
function net_ocean_heat_flux end
function ocean_freshwater_heat_flux end
function frazil_heat_flux end

prepare_ocean_heat_budget!(::Nothing, esm) = nothing
complete_ocean_heat_budget!(::Nothing, esm, Δt) = nothing

function prepare_ocean_heat_budget!(budget::OceanHeatBudget, esm)
    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.previous_heat_content, budget.heat_content)

    # Frazil heat stored on the interfaces has already been applied directly
    # to the ocean by the preceding `update_state!`. Cache only the fluxes
    # that will be applied during the component steps below.
    applied_surface_flux = net_ocean_heat_flux(esm) - frazil_heat_flux(esm)
    Oceananigans.set!(budget.surface_flux, applied_surface_flux)

    prepare_radiative_heat_flux!(budget.applied_radiative_heat_flux,
                                 budget.radiative_heat_flux)

    Oceananigans.set!(budget.freshwater_heat_flux, ocean_freshwater_heat_flux(esm))

    return nothing
end

function prepare_radiative_heat_flux!(applied_flux, ::Nothing)
    Oceananigans.set!(applied_flux, 0)
    return nothing
end

function prepare_radiative_heat_flux!(applied_flux, radiative_heat_flux)
    Oceananigans.compute!(radiative_heat_flux)
    Oceananigans.set!(applied_flux, radiative_heat_flux)
    return nothing
end

function complete_ocean_heat_budget!(budget::OceanHeatBudget, esm, Δt)
    # `update_state!` has now applied the newly diagnosed frazil correction
    # directly to ocean temperature. Include that new correction in this
    # interval's surface flux before forming the residual.
    completed_surface_flux = budget.surface_flux + frazil_heat_flux(esm)
    Oceananigans.set!(budget.surface_flux, completed_surface_flux)

    Oceananigans.compute!(budget.heat_content)
    heat_content_tendency = (budget.heat_content - budget.previous_heat_content) / Δt
    Oceananigans.set!(budget.tendency, heat_content_tendency)

    residual = budget.surface_flux + budget.tendency -
               budget.applied_radiative_heat_flux - budget.freshwater_heat_flux
    Oceananigans.set!(budget.residual, residual)

    return nothing
end

function Oceananigans.prognostic_state(budget::OceanHeatBudget)
    return (previous_heat_content = prognostic_state(budget.previous_heat_content),
            tendency = prognostic_state(budget.tendency),
            surface_flux = prognostic_state(budget.surface_flux),
            applied_radiative_heat_flux = prognostic_state(budget.applied_radiative_heat_flux),
            freshwater_heat_flux = prognostic_state(budget.freshwater_heat_flux),
            residual = prognostic_state(budget.residual))
end

function Oceananigans.restore_prognostic_state!(budget::OceanHeatBudget, state)
    restore_prognostic_state!(budget.previous_heat_content, state.previous_heat_content)
    restore_prognostic_state!(budget.tendency, state.tendency)
    restore_prognostic_state!(budget.surface_flux, state.surface_flux)
    restore_prognostic_state!(budget.applied_radiative_heat_flux, state.applied_radiative_heat_flux)
    if hasproperty(state, :freshwater_heat_flux)
        restore_prognostic_state!(budget.freshwater_heat_flux, state.freshwater_heat_flux)
    else
        Oceananigans.set!(budget.freshwater_heat_flux, 0)
    end
    restore_prognostic_state!(budget.residual, state.residual)
    return budget
end
