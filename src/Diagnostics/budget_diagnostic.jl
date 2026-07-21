"""
    BudgetComputation(:temperature, model)

Track the ocean temperature budget for a coupled model.

```julia
budget = BudgetComputation(:temperature, model)
add_callback!(simulation, budget)
```

The callback runs once after every timestep. It saves the fields needed to
finish the next budget, so restarting from a checkpoint preserves the budget.
The residual field contains the budget from the most recently completed timestep.
"""
struct BudgetComputation{H, P, T, S, R, A, F, B}
    tracer_name :: Symbol
    heat_content :: H
    previous_heat_content :: P
    tendency :: T
    surface_flux :: S
    radiative_heat_flux :: R
    applied_radiative_heat_flux :: A
    freshwater_heat_flux :: F
    residual :: B
end

function BudgetComputation(tracer_name::Symbol, esm::EarthSystemModel)
    tracer_name === :temperature ||
        throw(ArgumentError("BudgetComputation currently supports only :temperature."))

    ocean = esm.ocean
    hasproperty(ocean, :model) ||
        throw(ArgumentError("BudgetComputation(:temperature, model) requires a prognostic Oceananigans ocean."))

    model = ocean.model
    grid = model.grid
    ρᵒᶜ = reference_density(ocean)
    cᵒᶜ = heat_capacity(ocean)

    H = Field(Integral(ρᵒᶜ * cᵒᶜ * model.tracers.T, dims=3))
    ColumnField = Field{Center, Center, Nothing}
    H⁻ = ColumnField(grid)
    ∂t_H = ColumnField(grid)
    Qˢ = ColumnField(grid)
    Qʳ = ColumnField(grid)
    Qᶠ = ColumnField(grid)
    B = ColumnField(grid)

    forcing = get_radiative_forcing(ocean)
    R = radiative_heat_flux(forcing, model, ρᵒᶜ, cᵒᶜ)

    return BudgetComputation(tracer_name, H, H⁻, ∂t_H, Qˢ, R, Qʳ, Qᶠ, B)
end

radiative_heat_flux(::Nothing, model, ρᵒᶜ, cᵒᶜ) = nothing

function radiative_heat_flux(forcing, model, ρᵒᶜ, cᵒᶜ)
    operation = KernelFunctionOperation{Center, Center, Center}(forcing,
                                                                 model.grid,
                                                                 model.clock,
                                                                 Oceananigans.fields(model))
    return Field(Integral(ρᵒᶜ * cᵒᶜ * operation, dims=3))
end

function cache_budget!(budget::BudgetComputation, esm)
    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.previous_heat_content, budget.heat_content)

    # The previous frazil flux has already changed ocean temperature. Store
    # only the fluxes that will be applied during the next timestep.
    surface_flux = net_ocean_heat_flux(esm) - frazil_heat_flux(esm)
    Oceananigans.set!(budget.surface_flux, surface_flux)

    cache_radiative_heat_flux!(budget.applied_radiative_heat_flux, budget.radiative_heat_flux)
    Oceananigans.set!(budget.freshwater_heat_flux, ocean_freshwater_heat_flux(esm))
    return nothing
end

function cache_radiative_heat_flux!(applied_flux, ::Nothing)
    Oceananigans.set!(applied_flux, 0)
    return nothing
end

function cache_radiative_heat_flux!(applied_flux, radiative_heat_flux)
    Oceananigans.compute!(radiative_heat_flux)
    Oceananigans.set!(applied_flux, radiative_heat_flux)
    return nothing
end

function complete_budget!(budget::BudgetComputation, esm, Δt)
    # The new frazil flux has now changed ocean temperature. Include it in
    # the surface flux for this completed timestep.
    Oceananigans.set!(budget.surface_flux, budget.surface_flux + frazil_heat_flux(esm))

    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.tendency,
                      (budget.heat_content - budget.previous_heat_content) / Δt)

    residual = budget.surface_flux + budget.tendency -
               budget.applied_radiative_heat_flux - budget.freshwater_heat_flux
    Oceananigans.set!(budget.residual, residual)
    return nothing
end

function Oceananigans.initialize!(budget::BudgetComputation, simulation)
    cache_budget!(budget, simulation.model)
    return nothing
end

function (budget::BudgetComputation)(simulation)
    simulation.model.clock.iteration == 0 && return nothing
    complete_budget!(budget, simulation.model, simulation.model.clock.last_Δt)
    cache_budget!(budget, simulation.model)
    return nothing
end

function Oceananigans.Simulations.Callback(budget::BudgetComputation, schedule=IterationInterval(1);
                  parameters=nothing, callsite=TimeStepCallsite())
    callsite isa TimeStepCallsite ||
        throw(ArgumentError("BudgetComputation must run after each timestep."))
    schedule = validate_schedule(budget, schedule)
    return Callback(budget, schedule, callsite, parameters)
end

function Oceananigans.Simulations.validate_schedule(::BudgetComputation, schedule::IterationInterval)
    schedule == IterationInterval(1) ||
        throw(ArgumentError("BudgetComputation must run every iteration."))
    return schedule
end

Oceananigans.Simulations.validate_schedule(::BudgetComputation, schedule) =
    throw(ArgumentError("BudgetComputation must use IterationInterval(1)."))

function Oceananigans.prognostic_state(budget::BudgetComputation)
    return (; previous_heat_content = Oceananigans.prognostic_state(budget.previous_heat_content),
            tendency = Oceananigans.prognostic_state(budget.tendency),
            surface_flux = Oceananigans.prognostic_state(budget.surface_flux),
            applied_radiative_heat_flux = Oceananigans.prognostic_state(budget.applied_radiative_heat_flux),
            freshwater_heat_flux = Oceananigans.prognostic_state(budget.freshwater_heat_flux),
            residual = Oceananigans.prognostic_state(budget.residual))
end

function Oceananigans.restore_prognostic_state!(budget::BudgetComputation, state)
    Oceananigans.restore_prognostic_state!(budget.previous_heat_content, state.previous_heat_content)
    Oceananigans.restore_prognostic_state!(budget.tendency, state.tendency)
    Oceananigans.restore_prognostic_state!(budget.surface_flux, state.surface_flux)
    Oceananigans.restore_prognostic_state!(budget.applied_radiative_heat_flux, state.applied_radiative_heat_flux)
    Oceananigans.restore_prognostic_state!(budget.freshwater_heat_flux, state.freshwater_heat_flux)
    Oceananigans.restore_prognostic_state!(budget.residual, state.residual)
    return budget
end

function Oceananigans.prognostic_state(callback::Callback{P, <:BudgetComputation}) where P
    return (; func = Oceananigans.prognostic_state(callback.func),
            schedule = Oceananigans.prognostic_state(callback.schedule))
end

function Oceananigans.restore_prognostic_state!(callback::Callback{P, <:BudgetComputation}, state) where P
    Oceananigans.restore_prognostic_state!(callback.func, state.func)
    Oceananigans.restore_prognostic_state!(callback.schedule, state.schedule)
    return callback
end

function Oceananigans.Simulations.add_callback!(simulation::Simulation, budget::BudgetComputation;
                                                 name=:temperature_budget)
    callback = Callback(budget)
    return Oceananigans.Simulations.add_callback!(simulation, callback; name)
end
