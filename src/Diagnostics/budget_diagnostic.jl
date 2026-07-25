"""
    BudgetComputation(name, model)

Track an ocean column budget for a coupled model.

```julia
budget = BudgetComputation(:temperature, model)
add_callback!(simulation, budget)
```

Supported budgets are `:temperature`, `:salinity`, and `:mass`.

The callback runs once after every timestep. It saves the fields needed to
finish the next budget, so restarting from a checkpoint preserves the budget.
The residual field contains the budget from the most recently completed timestep.
If the budget is attached after the simulation has started, it starts from the
current model state. Older budgets are available only if they were saved in a
checkpoint and restored during pickup.
"""
struct BudgetComputation{H, P, T, S, R, A, B}
    tracer_name :: Symbol
    heat_content :: H
    previous_heat_content :: P
    tendency :: T
    surface_flux :: S
    radiative_heat_flux :: R
    applied_radiative_heat_flux :: A
    residual :: B
end

Base.summary(budget::BudgetComputation) =
    string("BudgetComputation(:", budget.tracer_name, ") on ", summary(budget.residual.grid))

Base.show(io::IO, budget::BudgetComputation) = print(io, summary(budget))

BudgetComputation(tracer_name::Symbol, esm::EarthSystemModel) =
    BudgetComputation(Val(tracer_name), tracer_name, esm)

function BudgetComputation(kind::Val, tracer_name::Symbol, esm::EarthSystemModel)
    ocean = esm.ocean
    hasproperty(ocean, :model) ||
        throw(ArgumentError("BudgetComputation(name, model) requires a prognostic Oceananigans ocean."))

    model = ocean.model
    grid = model.grid

    H = budget_inventory(kind, esm)
    ColumnField = Field{Center, Center, Nothing}
    H⁻ = ColumnField(grid)
    ∂t_H = ColumnField(grid)
    Qˢ = ColumnField(grid)
    Qʳ = ColumnField(grid)
    B = ColumnField(grid)

    R = budget_radiative_flux(kind, esm)

    return BudgetComputation(tracer_name, H, H⁻, ∂t_H, Qˢ, R, Qʳ, B)
end

function budget_inventory(::Val{:temperature}, esm)
    ocean = esm.ocean
    model = ocean.model
    ρᵒᶜ = reference_density(ocean)
    cᵒᶜ = heat_capacity(ocean)
    return Field(Integral(ρᵒᶜ * cᵒᶜ * model.tracers.T, dims=3))
end

budget_inventory(::Val{:salinity}, esm) =
    Field(Integral(esm.ocean.model.tracers.S, dims=3))

function budget_inventory(::Val{:mass}, esm)
    ocean = esm.ocean
    model = ocean.model
    ρᵒᶜ = reference_density(ocean)
    return Field(Integral(0 * model.tracers.T + ρᵒᶜ, dims=3))
end

function budget_inventory(::Val{name}, esm) where name
    throw(ArgumentError("BudgetComputation supports :temperature, :salinity, and :mass, not :" * string(name) * "."))
end

function budget_radiative_flux(::Val{:temperature}, esm)
    ocean = esm.ocean
    model = ocean.model
    ρᵒᶜ = reference_density(ocean)
    cᵒᶜ = heat_capacity(ocean)
    forcing = get_radiative_forcing(ocean)
    return radiative_heat_flux(forcing, model, ρᵒᶜ, cᵒᶜ)
end

budget_radiative_flux(::Val{:salinity}, esm) = nothing
budget_radiative_flux(::Val{:mass}, esm) = nothing

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

function cache_budget!(budget::BudgetComputation, esm)
    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.previous_heat_content, budget.heat_content)
    Oceananigans.set!(budget.surface_flux, budget_surface_flux(Val(budget.tracer_name), esm))
    cache_radiative_heat_flux!(budget.applied_radiative_heat_flux, budget.radiative_heat_flux)
    return nothing
end

function budget_surface_flux(::Val{:temperature}, esm)
    # The previous frazil flux has already changed ocean temperature. Store
    # only the fluxes that will be applied during the next timestep.
    return net_ocean_heat_flux(esm) - frazil_heat_flux(esm)
end

budget_surface_flux(::Val{:salinity}, esm) = net_ocean_salinity_flux(esm)

# `net_ocean_freshwater_flux` is positive into the ocean. The budget residual
# uses outward-positive surface fluxes, so mass gets the opposite sign.
budget_surface_flux(::Val{:mass}, esm) = - net_ocean_freshwater_flux(esm)

function complete_budget!(budget::BudgetComputation, esm, Δt)
    Oceananigans.set!(budget.surface_flux,
                      completed_surface_flux(Val(budget.tracer_name), budget.surface_flux, esm))

    Oceananigans.compute!(budget.heat_content)
    Oceananigans.set!(budget.tendency,
                      (budget.heat_content - budget.previous_heat_content) / Δt)

    residual = budget.surface_flux + budget.tendency - budget.applied_radiative_heat_flux
    Oceananigans.set!(budget.residual, residual)
    return nothing
end

function completed_surface_flux(::Val{:temperature}, surface_flux, esm)
    # The new frazil flux has now changed ocean temperature. Include it in
    # the surface flux for this completed timestep.
    return surface_flux + frazil_heat_flux(esm)
end

completed_surface_flux(::Val{:salinity}, surface_flux, esm) = surface_flux
completed_surface_flux(::Val{:mass}, surface_flux, esm) = surface_flux

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
            residual = Oceananigans.prognostic_state(budget.residual))
end

function Oceananigans.restore_prognostic_state!(budget::BudgetComputation, state)
    Oceananigans.restore_prognostic_state!(budget.previous_heat_content, state.previous_heat_content)
    Oceananigans.restore_prognostic_state!(budget.tendency, state.tendency)
    Oceananigans.restore_prognostic_state!(budget.surface_flux, state.surface_flux)
    Oceananigans.restore_prognostic_state!(budget.applied_radiative_heat_flux, state.applied_radiative_heat_flux)
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
                                                 name=nothing)
    name = isnothing(name) ? Symbol(budget.tracer_name, :_budget) : name
    iteration = simulation.model.clock.iteration
    callback = Callback(budget)

    if iteration > 0
        message = """
        BudgetComputation is being attached after the simulation has already started.
        The diagnostic will start from the current model state, so the first completed
        budget will measure changes after this attachment point. Budgets before this
        iteration are not available unless they were saved in an older checkpoint.
        """

        @warn message current_iteration=iteration
        Oceananigans.initialize!(callback, simulation)
    end

    return Oceananigans.Simulations.add_callback!(simulation, callback; name)
end
