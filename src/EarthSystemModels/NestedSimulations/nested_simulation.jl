#####
##### NestedSimulation: coordinates a child Simulation against a parent
#####
#
# Currently one-way only: the parent drives the child via boundary conditions
# and interior relaxation; the child does not feed back into the parent. Two-way
# coupling will get its own design when we know what it should look like.

"""
    NestedSimulation(parent, child::Simulation;
                     parent_update_schedule = IterationInterval(1))

Couple a `child` Oceananigans `Simulation` to a `parent` state. The `parent` is
any object satisfying the duck-typed parent interface (see `parent_clock`,
`parent_field`, `parent_time_step!`, `parent_update_state!`).

A callback is installed on the child that, after every iteration matching
`parent_update_schedule`, syncs the parent's clock to the child's and calls
`parent_update_state!`. The parent's `FieldTimeSeries` backing stays aligned
with the child's wall-time so boundary-condition and forcing queries see the
right time window.

`run!(::NestedSimulation)` and `time_step!(::NestedSimulation, ...)` forward to
the child; the parent is kept in sync via the installed callback.
"""
mutable struct NestedSimulation{P, C}
    parent :: P
    child  :: C
end

function NestedSimulation(parent, child::Simulation;
                          parent_update_schedule = IterationInterval(1))

    install_parent_sync_callback!(child, parent, parent_update_schedule)

    # Use the parametric inner constructor explicitly to avoid recursion into
    # this outer method (the outer method is more specific on child::Simulation).
    return NestedSimulation{typeof(parent), typeof(child)}(parent, child)
end

function install_parent_sync_callback!(child, parent, schedule)
    sync = ParentSyncCallback(parent)
    child.callbacks[:nested_simulation_parent_sync] = Callback(sync, schedule)
    return nothing
end

# A struct (not a closure) so it survives Adapt/serialization cleanly.
struct ParentSyncCallback{P}
    parent :: P
end

function (sync::ParentSyncCallback)(sim)
    parent_clock(sync.parent).time = sim.model.clock.time
    parent_update_state!(sync.parent)
    return nothing
end

# ---------------------------------------------------------------------------
# Simulation-like surface: time_step!, run!, plus a clock accessor for tooling.

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
