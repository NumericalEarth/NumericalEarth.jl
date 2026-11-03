#####
##### NestedSimulation: type alias + convenience constructor
#####

"""
    NestedSimulation

Type alias for `Simulation{<:NestedModel}` — an Oceananigans `Simulation`
whose model is a [`NestedModel`](@ref). Useful for `::NestedSimulation`
dispatch.
"""
const NestedSimulation = Simulation{<:NestedModel}

"""
    NestedSimulation(parent, child_model::AbstractModel; simulation_kwargs...)

Convenience for `Simulation(NestedModel(parent, child_model); simulation_kwargs...)`.

`simulation_kwargs` are forwarded to the `Simulation` constructor (`Δt`,
`stop_time`, `stop_iteration`, `verbose`, etc.). Parent-sync is handled by
`NestedModel`'s `time_step!`, so no callback is installed — the result is a
plain Oceananigans `Simulation` whose model happens to be a `NestedModel`.
"""
NestedSimulation(parent, child_model::AbstractModel; simulation_kwargs...) =
    Simulation(NestedModel(parent, child_model); simulation_kwargs...)
