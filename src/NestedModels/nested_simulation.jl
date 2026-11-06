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
    NestedSimulation(parent, child; simulation_kwargs...)

Convenience for `Simulation(NestedModel(parent, child); simulation_kwargs...)`.

`simulation_kwargs` are forwarded to the `Simulation` constructor (`Δt`,
`stop_time`, `stop_iteration`, `verbose`, etc.). Its `Δt` is the exchange interval:
each step brings the parent, then the child, to the next exchange time.
"""
NestedSimulation(parent, child; simulation_kwargs...) =
    Simulation(NestedModel(parent, child); simulation_kwargs...)
