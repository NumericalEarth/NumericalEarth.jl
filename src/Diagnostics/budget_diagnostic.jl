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
Attach the budget before the first timestep. For a checkpoint restart, attach it
before calling `run!(simulation; pickup=true)` so its saved history is restored.
"""
struct BudgetComputation{H, P, T, S, R, A, C, I, Q, F, G, U, B}
    tracer_name :: Symbol
    heat_content :: H
    previous_heat_content :: P
    tendency :: T
    surface_flux :: S
    radiative_heat_flux :: R
    applied_radiative_heat_flux :: A
    cached_radiative_heat_flux :: C
    instantaneous_surface_flux :: I
    previous_surface_flux :: Q
    previous_radiative_heat_flux :: F
    stage_temperature :: G
    net_heat_flux :: U
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
    Qʳ_cache = ColumnField(grid)
    Qⁱ = ColumnField(grid)
    Q⁻ = ColumnField(grid)
    Qʳ⁻ = ColumnField(grid)
    B = ColumnField(grid)

    T_stage = Field{Center, Center, Center}(grid)
    # Frazil heat is applied after the ocean timestep, during the coupled-model
    # update. It is added separately when the completed budget is assembled.
    Q = net_ocean_heat_flux(esm, T_stage) - frazil_heat_flux(esm)

    forcing = get_radiative_forcing(ocean)
    R = radiative_heat_flux(forcing, model, ρᵒᶜ, cᵒᶜ)

    return BudgetComputation(tracer_name, H, H⁻, ∂t_H, Qˢ, R, Qʳ,
                             Qʳ_cache, Qⁱ, Q⁻, Qʳ⁻, T_stage, Q, B)
end

radiative_heat_flux(::Nothing, model, ρᵒᶜ, cᵒᶜ) = nothing

function radiative_heat_flux(forcing, model, ρᵒᶜ, cᵒᶜ)
    operation = KernelFunctionOperation{Center, Center, Center}(forcing,
                                                                 model.grid,
                                                                 model.clock,
                                                                 Oceananigans.fields(model))
    return Field(Integral(ρᵒᶜ * cᵒᶜ * operation, dims=3))
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

function initialize_budget_cache!(budget::BudgetComputation, esm)
    model = esm.ocean.model
    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.previous_heat_content, budget.heat_content)
    Oceananigans.set!(budget.stage_temperature, model.tracers.T)
    cache_radiative_heat_flux!(budget.cached_radiative_heat_flux, budget.radiative_heat_flux)
    Oceananigans.set!(budget.previous_radiative_heat_flux, budget.cached_radiative_heat_flux)
    Oceananigans.set!(budget.surface_flux, 0)
    Oceananigans.set!(budget.applied_radiative_heat_flux, 0)
    Oceananigans.set!(budget.instantaneous_surface_flux, 0)
    Oceananigans.set!(budget.previous_surface_flux, 0)
    return nothing
end

function cache_heat_content!(budget::BudgetComputation)
    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.previous_heat_content, budget.heat_content)
    return nothing
end

function cache_applied_budget_flux!(model, budget::BudgetComputation)
    cache_applied_budget_flux!(model, budget, model.timestepper)
    return nothing
end

function cache_applied_budget_flux!(model, budget,
                                    timestepper::Oceananigans.TimeSteppers.SplitRungeKuttaTimeStepper)
    stage = model.clock.stage
    final_stage = timestepper.Nstages

    if stage == final_stage - 1
        Oceananigans.set!(budget.stage_temperature, model.tracers.T)
        cache_radiative_heat_flux!(budget.cached_radiative_heat_flux, budget.radiative_heat_flux)
    elseif stage == final_stage
        Oceananigans.set!(budget.instantaneous_surface_flux, budget.net_heat_flux)
        Oceananigans.set!(budget.surface_flux, budget.instantaneous_surface_flux)
        Oceananigans.set!(budget.applied_radiative_heat_flux, budget.cached_radiative_heat_flux)
    end

    return nothing
end

function cache_applied_budget_flux!(model, budget,
                                    timestepper::Oceananigans.TimeSteppers.QuasiAdamsBashforth2TimeStepper)
    # The initialization update_state! occurs before the first AB2 tendency.
    model.clock.iteration == 0 && return nothing

    Oceananigans.set!(budget.instantaneous_surface_flux, budget.net_heat_flux)

    χ = timestepper.χ
    α = 3 / 2 + χ
    β = 1 / 2 + χ
    Oceananigans.set!(budget.surface_flux,
                      α * budget.instantaneous_surface_flux - β * budget.previous_surface_flux)
    Oceananigans.set!(budget.applied_radiative_heat_flux,
                      α * budget.cached_radiative_heat_flux - β * budget.previous_radiative_heat_flux)

    Oceananigans.set!(budget.previous_surface_flux, budget.instantaneous_surface_flux)
    Oceananigans.set!(budget.previous_radiative_heat_flux, budget.cached_radiative_heat_flux)
    Oceananigans.set!(budget.stage_temperature, model.tracers.T)

    # update_state! has advanced time-dependent forcing to the next AB2 state.
    cache_radiative_heat_flux!(budget.cached_radiative_heat_flux, budget.radiative_heat_flux)
    return nothing
end

function cache_applied_budget_flux!(model, budget, timestepper)
    throw(ArgumentError("BudgetComputation does not support timestepper " * string(typeof(timestepper)) * "."))
end

function complete_budget!(budget::BudgetComputation, esm, Δt)
    # The coupled-model update has now applied the new frazil correction to
    # ocean temperature. Unlike an ocean tendency, this correction is applied
    # once and must not receive Runge--Kutta or Adams--Bashforth weights.
    Oceananigans.set!(budget.surface_flux,
                      budget.surface_flux + frazil_heat_flux(esm))

    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.tendency,
                      (budget.heat_content - budget.previous_heat_content) / Δt)

    residual = budget.surface_flux + budget.tendency - budget.applied_radiative_heat_flux
    Oceananigans.set!(budget.residual, residual)
    return nothing
end

function Oceananigans.initialize!(budget::BudgetComputation, simulation)
    initialize_budget_cache!(budget, simulation.model)
    return nothing
end

function (budget::BudgetComputation)(simulation)
    simulation.model.clock.iteration == 0 && return nothing
    complete_budget!(budget, simulation.model, simulation.model.clock.last_Δt)
    cache_heat_content!(budget)
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
            cached_radiative_heat_flux = Oceananigans.prognostic_state(budget.cached_radiative_heat_flux),
            instantaneous_surface_flux = Oceananigans.prognostic_state(budget.instantaneous_surface_flux),
            previous_surface_flux = Oceananigans.prognostic_state(budget.previous_surface_flux),
            previous_radiative_heat_flux = Oceananigans.prognostic_state(budget.previous_radiative_heat_flux),
            stage_temperature = Oceananigans.prognostic_state(budget.stage_temperature),
            residual = Oceananigans.prognostic_state(budget.residual))
end

function Oceananigans.restore_prognostic_state!(budget::BudgetComputation, state)
    Oceananigans.restore_prognostic_state!(budget.previous_heat_content, state.previous_heat_content)
    Oceananigans.restore_prognostic_state!(budget.tendency, state.tendency)
    Oceananigans.restore_prognostic_state!(budget.surface_flux, state.surface_flux)
    Oceananigans.restore_prognostic_state!(budget.applied_radiative_heat_flux, state.applied_radiative_heat_flux)
    Oceananigans.restore_prognostic_state!(budget.cached_radiative_heat_flux, state.cached_radiative_heat_flux)
    Oceananigans.restore_prognostic_state!(budget.instantaneous_surface_flux, state.instantaneous_surface_flux)
    Oceananigans.restore_prognostic_state!(budget.previous_surface_flux, state.previous_surface_flux)
    Oceananigans.restore_prognostic_state!(budget.previous_radiative_heat_flux, state.previous_radiative_heat_flux)
    Oceananigans.restore_prognostic_state!(budget.stage_temperature, state.stage_temperature)
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
    iteration = simulation.model.clock.iteration

    if iteration > 0
        message = """
        BudgetComputation cannot be attached after the simulation has already started.
        Every budget needs the heat content from the start of the timestep. The AB2
        method also needs surface and radiation fluxes from the previous timestep.
        A newly created budget does not have that history, and using zero instead
        would give a wrong heat budget.

        For a new run, attach BudgetComputation before the first timestep. For a
        checkpoint restart, create and attach it before calling
        run!(simulation; pickup=true). Its saved history will then be restored from
        the checkpoint. The checkpoint must have been written with this budget attached.
        """
        @warn message current_iteration=iteration
        throw(ArgumentError(message))
    end

    flux_cache = Callback(cache_applied_budget_flux!, IterationInterval(1);
                          parameters=budget,
                          callsite=Oceananigans.UpdateStateCallsite())
    Oceananigans.Simulations.add_callback!(simulation.model.ocean, flux_cache;
                                           name=:temperature_budget_flux_cache)

    callback = Callback(budget)
    return Oceananigans.Simulations.add_callback!(simulation, callback; name)
end
