"""
    OceanHeatBudget

Coupled ocean heat-budget fields on the native ocean grid.

`residual` contains the completed column budget

```math
Q_{surface} + \\partial_t H - Q_{radiative},
```

for the most recently completed coupling interval. All fields have units of
W m⁻².

The fields are ordinary Oceananigans fields, so output schedules apply to them
in the usual way. For example, a conservatively remapped daily mean can be
written with

```julia
budget = esm.interfaces.budgets.ocean_heat
heat_budget = RegriddedField(budget.residual, destination_grid)
outputs = (; heat_budget)

writer = JLD2Writer(esm, outputs;
                    schedule = AveragedTimeInterval(1day),
                    filename = "ocean_heat_budget")
```
"""
struct OceanHeatBudget{H, P, T, S, R, A, B}
    heat_content :: H
    previous_heat_content :: P
    tendency :: T
    surface_flux :: S
    radiative_heat_flux :: R
    applied_radiative_heat_flux :: A
    residual :: B
end

ocean_heat_budget(ocean) = nothing

# Implemented alongside the other public interface-flux diagnostics in
# `Diagnostics/interface_fluxes.jl`.
function net_ocean_heat_flux end

prepare_ocean_heat_budget!(::Nothing, esm) = nothing
complete_ocean_heat_budget!(::Nothing, esm, Δt) = nothing

function prepare_ocean_heat_budget!(budget::OceanHeatBudget, esm)
    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.previous_heat_content, budget.heat_content)

    Oceananigans.set!(budget.surface_flux, net_ocean_heat_flux(esm))

    prepare_radiative_heat_flux!(budget.applied_radiative_heat_flux,
                                 budget.radiative_heat_flux)

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
    Oceananigans.compute!(budget.heat_content)
    heat_content_tendency = (budget.heat_content - budget.previous_heat_content) / Δt
    Oceananigans.set!(budget.tendency, heat_content_tendency)

    residual = budget.surface_flux + budget.tendency - budget.applied_radiative_heat_flux
    Oceananigans.set!(budget.residual, residual)

    return nothing
end

function Oceananigans.prognostic_state(budget::OceanHeatBudget)
    return (previous_heat_content = prognostic_state(budget.previous_heat_content),
            tendency = prognostic_state(budget.tendency),
            surface_flux = prognostic_state(budget.surface_flux),
            applied_radiative_heat_flux = prognostic_state(budget.applied_radiative_heat_flux),
            residual = prognostic_state(budget.residual))
end

function Oceananigans.restore_prognostic_state!(budget::OceanHeatBudget, state)
    restore_prognostic_state!(budget.previous_heat_content, state.previous_heat_content)
    restore_prognostic_state!(budget.tendency, state.tendency)
    restore_prognostic_state!(budget.surface_flux, state.surface_flux)
    restore_prognostic_state!(budget.applied_radiative_heat_flux, state.applied_radiative_heat_flux)
    restore_prognostic_state!(budget.residual, state.residual)
    return budget
end
