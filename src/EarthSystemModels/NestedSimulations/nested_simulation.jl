#####
##### NestedSimulation: coordinates a child Simulation against a parent
#####
#
# v1 is one-way only — the parent drives the child via boundary conditions and
# (later) interior relaxation; the child does not feed back into the parent.

"""
    NestedSimulation(parent, child::Simulation;
                     parent_update_schedule = IterationInterval(1))

Couple a `child` Oceananigans `Simulation` to a `parent` state. The `parent`
must expose `parent.clock` (an Oceananigans `Clock`) and support
`Oceananigans.TimeSteppers.time_step!(parent, Δt)` — which `PrescribedAtmosphere`
already does, and which any Oceananigans `Simulation` satisfies through its
`model.clock`.

A callback is installed on the child that, after each iteration matching
`parent_update_schedule`, advances the parent so its clock matches the child's.
That ticks the parent's `FieldTimeSeries` backing forward so boundary-condition
queries see the right time window.

`run!(::NestedSimulation)` and `time_step!(::NestedSimulation, ...)` forward to
the child; the parent is kept in sync via the installed callback.
"""
mutable struct NestedSimulation{P, C}
    parent :: P
    child  :: C
end

function NestedSimulation(parent, child::Simulation;
                          parent_update_schedule = IterationInterval(1))

    sync = ParentSyncCallback(parent)
    child.callbacks[:nested_simulation_parent_sync] = Callback(sync, parent_update_schedule)

    # Explicit parametric form to dodge recursion into this outer method.
    return NestedSimulation{typeof(parent), typeof(child)}(parent, child)
end

# A struct (not a closure) so it adapts cleanly.
struct ParentSyncCallback{P}
    parent :: P
end

function (sync::ParentSyncCallback)(sim)
    Δt = sim.model.clock.time - sync.parent.clock.time
    Δt > 0 && time_step!(sync.parent, Δt)
    return nothing
end

# ---------------------------------------------------------------------------
# Simulation-like surface.

time_step!(ns::NestedSimulation, Δt) = time_step!(ns.child, Δt)
time_step!(ns::NestedSimulation)     = time_step!(ns.child)
run!(ns::NestedSimulation; kwargs...) = run!(ns.child; kwargs...)

Base.summary(ns::NestedSimulation) =
    string("NestedSimulation of ", summary(ns.child.model),
           " driven by ", summary(ns.parent))

function Base.show(io::IO, ns::NestedSimulation)
    print(io, summary(ns), '\n',
              "├── parent: ", summary(ns.parent), '\n',
              "└── child:  ", summary(ns.child.model))
end
