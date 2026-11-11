#####
##### NestedModel: an Oceananigans `AbstractModel` that pairs a child simulation
##### with a parent and exchanges state between them during `time_step!`.
#####
#
# The child carries the actual prognostic state and the NormalFlow BCs / interior
# `Relaxation` forcings that reference the parent's state, so a plain
# `Simulation(NestedModel(parent, child))` just works.
#
# The Oceananigans model-protocol calls (`fields`, `prognostic_fields`, `architecture`, `update_state!`,
# `cell_advection_timescale`, ...) are forwarded to the child's model with explicit methods, not via
# `Base.getproperty`. The child is a `Simulation`, so it keeps its own Δt, callbacks, and output.

import Oceananigans
import Oceananigans: AbstractModel, fields, prognostic_state, restore_prognostic_state!, prognostic_fields, set!
import Oceananigans.Architectures: architecture
import Oceananigans.TimeSteppers: time_step!, update_state!
import Oceananigans.Simulations: iteration

using Oceananigans.Simulations: ModelCallsite
using Oceananigans.TimeSteppers: tick!

"""
    NestedModel(parent, child, exchanger=nothing)

Pair a `child` `Simulation` with a `parent`, which is either a `Simulation` or a prescribed component
(e.g. a 3D `PrescribedAtmosphere`). A `child` model is wrapped in a `Simulation` whose own Δt is
unbounded, so it takes one step per `NestedModel` step.

`time_step!(nested, Δt)` exchanges the parent's state with the child every `Δt`: the parent is brought
to the next exchange time, then the child steps until it reaches the same time. Each `Simulation` steps
with its own Δt and lands exactly on the exchange time, so the child never passes the parent.
"""
mutable struct NestedModel{P, S, X, C, TS, A} <: AbstractModel{TS, A}
    parent    :: P
    child     :: S
    exchanger :: X    # mediates parent state → child variables (see `exchange_state!`); `nothing` if unused
    clock     :: C    # the exchange clock: one iteration per exchange
end

function NestedModel(parent, child::Simulation{<:AbstractModel{TS, A}}, exchanger=nothing) where {TS, A}
    remove_default_stop_callbacks!(child)
    clock = deepcopy(component_model(child).clock)
    return NestedModel{typeof(parent), typeof(child), typeof(exchanger), typeof(clock), TS, A}(parent, child, exchanger, clock)
end

NestedModel(parent, child::AbstractModel, exchanger=nothing) =
    NestedModel(parent, Simulation(child; Δt = Inf, verbose = false), exchanger)

# Refresh the parent-derived child state held by the exchanger to the current `time`. The exchanger
# (a NumericalEarth object; see the Breeze extension's `StateExchanger`) recomputes the child
# prognostics on the parent grid as needed. Default no-op so a bare `NestedModel(parent, child)` — or
# one whose parent needs no state transform — just forwards.
exchange_state!(exchanger, time) = nothing
exchange_state!(exchanger, start_time, end_time) = exchange_state!(exchanger, end_time)
exchange_state!(exchangers::NamedTuple, time) = foreach(exchanger -> exchange_state!(exchanger, time), exchangers)
exchange_state!(exchangers::NamedTuple, start_time, end_time) =
    foreach(exchanger -> exchange_state!(exchanger, start_time, end_time), exchangers)

# Model-protocol dispatches, forwarded explicitly to the child's model.
fields(nm::NestedModel)            = fields(component_model(nm.child))
prognostic_fields(nm::NestedModel) = prognostic_fields(component_model(nm.child))
architecture(nm::NestedModel)      = architecture(component_model(nm.child))
iteration(nm::NestedModel)         = nm.clock.iteration

Oceananigans.Diagnostics.default_nan_checker(nm::NestedModel) =
    Oceananigans.Diagnostics.default_nan_checker(component_model(nm.child))

Oceananigans.Advection.cell_advection_timescale(nm::NestedModel) =
    Oceananigans.Advection.cell_advection_timescale(component_model(nm.child))

model_callbacks(sim::Simulation) = Tuple(cb for cb in values(sim.callbacks) if cb.callsite isa ModelCallsite)

# Refresh the parent-derived child state first so the child's boundary conditions / forcings see current
# values. The child's own model callbacks run alongside any passed in.
function update_state!(nm::NestedModel, callbacks=[]; kwargs...)
    exchange_state!(nm.exchanger, nm.clock.time)
    return update_state!(component_model(nm.child), (callbacks..., model_callbacks(nm.child)...); kwargs...)
end

# Setting the initial state targets the child's prognostics.
set!(nm::NestedModel, args...; kwargs...) = set!(component_model(nm.child), args...; kwargs...)

# Refresh the exchanger over the exchange interval, so the parent data the child interpolates is resident
# before the child steps.
function time_step!(nm::NestedModel, Δt; callbacks=())
    isempty(callbacks) || throw(ArgumentError("NestedModel does not run model callbacks; add them to the child Simulation instead."))
    start_time = nm.clock.time
    tick!(nm.clock, Δt)
    advance_parent!(nm.parent, nm.clock)
    exchange_state!(nm.exchanger, start_time, nm.clock.time)
    step_until!(nm.child, nm.clock.time)
    return nothing
end

# TODO: move `step_until!` to Oceananigans.Simulations, aligning to `time` without resetting `stop_time`.
"""
$(TYPEDSIGNATURES)

Step `simulation` with its own Δt until its clock reaches `time`, shortening the last step to land on it.
"""
function step_until!(simulation::Simulation, time)
    simulation.stop_time = time
    while component_model(simulation).clock.time < time
        time_step!(simulation)
    end
    return nothing
end

# A prescribed parent is a function of time: it takes the exchange clock and refreshes its state there.
# No branch on the gap between the clocks, which is a traced number inside a compiled step.
function advance_parent!(parent::AbstractPrescribedComponent, clock)
    copy_clock!(parent.clock, clock)
    update_state!(parent)
    return nothing
end

function advance_parent!(parent::BoundaryPrescribedComponent, clock)
    copy_clock!(parent.clock, clock)
    foreach(strip -> advance_parent!(strip, clock), boundary_strips(parent))
    return nothing
end

advance_parent!(parent::Simulation, clock) = step_until!(parent, clock.time)

# Rebinding the fields is for host clocks; a traced clock (NumericalEarthReactantExt) writes through its
# own numbers instead, so the two clocks stay distinct program results.
function copy_clock!(clock, source)
    clock.time = source.time
    clock.iteration = source.iteration
    clock.last_Δt = source.last_Δt
    clock.last_stage_Δt = source.last_stage_Δt
    return nothing
end

prognostic_state(nm::NestedModel) = (clock  = prognostic_state(nm.clock),
                                     parent = prognostic_state(nm.parent),
                                     child  = prognostic_state(nm.child))

function restore_prognostic_state!(nm::NestedModel, state)
    restore_prognostic_state!(nm.clock, state.clock)
    restore_prognostic_state!(nm.parent, state.parent)
    restore_prognostic_state!(nm.child, state.child)
    exchange_state!(nm.exchanger, nm.clock.time)
    return nm
end

restore_prognostic_state!(nm::NestedModel, ::Nothing) = nm
Base.summary(nm::NestedModel) = string("NestedModel(", summary(nm.parent), " → ", summary(component_model(nm.child)), ")")

function Base.show(io::IO, nm::NestedModel)
    print(io, summary(nm), '\n',
              "├── parent: ", summary(nm.parent), '\n',
              "└── child:  ", summary(component_model(nm.child)))
end
