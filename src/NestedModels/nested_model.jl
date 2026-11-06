#####
##### NestedModel: an Oceananigans `AbstractModel` that pairs a child model
##### with a parent state and ticks the parent's clock during `time_step!`.
#####
#
# The child carries the actual prognostic state and the NormalFlow BCs / interior
# `Relaxation` forcings that reference the parent's `FieldTimeSeries`. Wrapping
# the child in `NestedModel` lifts the parent-sync into the integrator itself
# (where it belongs — it's not output behavior), so a plain
# `Simulation(NestedModel(parent, child))` just works.
#
# The Oceananigans model-protocol calls (`fields`, `prognostic_fields`, `architecture`, `iteration`,
# `update_state!`, `cell_advection_timescale`, ...) are forwarded to the child with explicit methods —
# NOT via `Base.getproperty`, which would silently forward typos and is discouraged (see AGENTS.md).
# `Simulation`/`run!` reads `model.clock` as a field, so `NestedModel` holds the child's clock directly
# (a shared reference). `time_step!` advances the parent, refreshes the exchanger, then steps the child
# over a step clipped so the child cannot pass the parent's time. Child-specific state (`velocities`,
# `grid`, `dynamics`, ...) is reached explicitly through `nested.child`.

import Oceananigans
import Oceananigans: AbstractModel, fields, prognostic_state, restore_prognostic_state!, prognostic_fields, set!
import Oceananigans.Architectures: architecture
import Oceananigans.TimeSteppers: time_step!, update_state!
import Oceananigans.Simulations: iteration
using Oceananigans.Simulations: new_time_step

"""
    NestedModel(parent, child::AbstractModel, exchanger=nothing, child_callbacks=(); parent_Δt=nothing)

Pair a `child` Oceananigans model with a `parent` state (e.g. a 3D
`PrescribedAtmosphere`). The result is itself an `AbstractModel` whose
property access and integrator-protocol methods forward to the child, except
for `time_step!`, which advances the parent ahead of the child and clips the
child's step so the child cannot pass the parent's state. The clipped step
lands the child exactly on the parent's time, so the two clocks coincide at
every rendezvous.

`parent_Δt` is the step taken by a parent that integrates its own dynamics: `nothing` keeps the
parent in lockstep with the child, a number is a fixed parent `Δt`, a function of the child's `Δt` such as
`Δt -> 3Δt` holds a refinement ratio while the child's `Δt` adapts, and a `TimeStepWizard` sizes each
parent step from the parent's own CFL. A fixed `Δt` cannot hold a ratio against an adapting child: once
the child's `Δt` outgrows it, the child is clipped down to the parent's rate. Each level's value is absolute, not a ratio to the child's `Δt`,
so a telescope whose `fine` child runs at `Δt`,
`NestedModel(NestedModel(coarse, medium; parent_Δt=4Δt), fine; parent_Δt=2Δt)`, steps
`medium` at `2Δt` and `coarse` at `4Δt`. A parent `Δt` below the child's own CFL limit drags the child
down to the parent's rate, since the child never steps past the parent; [`time_step_ratio`](@ref) gives
a parent `Δt` commensurate with the child's. A prescribed parent is a function of time rather than an
integration, so it constrains nothing and is sampled at the end of every child step.

Construct via [`child_model`](@ref) plus this wrapper, or use the
[`NestedSimulation`](@ref) convenience that bundles model construction and
`Simulation` wrapping.
"""
mutable struct NestedModel{P, M, X, CB, D, C, TS, A} <: AbstractModel{TS, A}
    parent          :: P
    child           :: M
    exchanger       :: X    # mediates parent state → child variables (see `exchange_state!`); `nothing` if unused
    child_callbacks :: CB   # callbacks added to the child's at every time step (e.g. boundary relaxation)
    parent_Δt       :: D    # the parent's own Δt; `nothing` keeps the parent in lockstep with the child
    clock           :: C    # shared reference to the child's clock (Simulation/run! read `model.clock`)
end

"""
$(TYPEDSIGNATURES)

Integer time-refinement factor between two nest levels: the ratio of the parent's smallest horizontal
grid spacing to the child's, rounded and floored at 1. Scaling the child's `Δt` by it gives a
`parent_Δt` commensurate with the child's step; an incommensurate parent `Δt` costs the child one
short step per parent interval, since the child lands on the parent's time.

```jldoctest
using NumericalEarth
using Oceananigans

nest_grid(extent, size) =
    LatitudeLongitudeGrid(; size, longitude = (-98 - extent/2, -98 + extent/2),
                          latitude = (36 - extent/2, 36 + extent/2), z = (0, 1e4),
                          topology = (Bounded, Bounded, Bounded))

time_step_ratio(nest_grid(8, (32, 32, 4)), nest_grid(2, (32, 32, 4)))

# output
4
```
"""
function time_step_ratio(parent_grid, child_grid)
    parent_spacing = min(minimum_xspacing(parent_grid), minimum_yspacing(parent_grid))
    child_spacing  = min(minimum_xspacing(child_grid),  minimum_yspacing(child_grid))
    return max(1, round(Int, parent_spacing / child_spacing))
end

NestedModel(parent, child::AbstractModel{TS, A}, exchanger=nothing, child_callbacks=();
            parent_Δt=nothing) where {TS, A} =
    NestedModel{typeof(parent), typeof(child), typeof(exchanger), typeof(child_callbacks),
                typeof(parent_Δt), typeof(child.clock), TS, A}(
        parent, child, exchanger, child_callbacks, parent_Δt, child.clock)

# Refresh the parent-derived child state held by the exchanger to the current `time`. The exchanger
# (a NumericalEarth object; see the Breeze extension's `StateExchanger`) recomputes the child
# prognostics on the parent grid as needed. Default no-op so a bare `NestedModel(parent, child)` — or
# one whose parent needs no state transform — just forwards.
exchange_state!(exchanger, time) = nothing
exchange_state!(exchangers::NamedTuple, time) = foreach(exchanger -> exchange_state!(exchanger, time), exchangers)

# Model-protocol dispatches, forwarded explicitly to the child.
fields(nm::NestedModel)            = fields(nm.child)
prognostic_fields(nm::NestedModel) = prognostic_fields(nm.child)
architecture(nm::NestedModel)      = architecture(nm.child)
iteration(nm::NestedModel)         = iteration(nm.child)

# Adaptive Δt: the wizard's `AdvectiveCFL` queries the simulation model, which is the
# `NestedModel` — forward to the child (the prognostic model whose grid/velocities set the CFL).
Oceananigans.Advection.cell_advection_timescale(nm::NestedModel) =
    Oceananigans.Advection.cell_advection_timescale(nm.child)

# Refresh the parent-derived child state first so the child's boundary conditions / forcings see current
# values. The child's own callbacks run alongside any passed in.
function update_state!(nm::NestedModel, callbacks=[]; kwargs...)
    exchange_state!(nm.exchanger, nm.clock.time)
    return update_state!(nm.child, (callbacks..., nm.child_callbacks...); kwargs...)
end

# Setting the initial state targets the child's prognostics.
set!(nm::NestedModel, args...; kwargs...) = set!(nm.child, args...; kwargs...)

# Ordering: advance the parent FIRST, so the child's step is bracketed by parent states rather than
# reaching past the parent's latest one. Then refresh the exchanger — it advances the parent's own FTS
# windows and fills the derived 2-level window — so the parent data the child interpolates over its step
# is resident *before* the child steps. Bracket the step END, not its start: the exchanger positions a
# 2-level window around the passed time, so refreshing at the start leaves the whole step extrapolated
# whenever that time lands on a parent time level (every hourly ERA5 crossing). Refreshing at the end
# keeps the step's late stages in-window; at most a ≤Δt start-side extrapolation remains on the crossing
# step (a 2-level window cannot span a parent interval). `Simulation` calls
# `time_step!(model, Δt; callbacks=...)`.
function time_step!(nm::NestedModel, Δt; callbacks=())
    parent_time = advance_parent!(nm.parent, nm.child.clock, Δt, nm.parent_Δt)
    Δt_child = aligned_child_time_step(Δt, nm.clock.time, parent_time)
    exchange_state!(nm.exchanger, nm.clock.time + Δt_child)
    time_step!(nm.child, Δt_child; callbacks = (callbacks..., nm.child_callbacks...))
    return nothing
end

# A prescribed parent constrains nothing, so the child takes the Δt it was given. Dispatch, not a
# comparison: the child's clock is a traced number inside a compiled step, and branching on it is what
# issue #816 is about.
aligned_child_time_step(Δt, time, ::Nothing) = Δt

# The child may not land past the parent's state. With more than a step of room, or a step that lands
# exactly on the parent, the child takes the Δt it was given; otherwise it takes the whole runway and
# lands on the parent, which also absorbs a remainder too thin to be worth its own step.
function aligned_child_time_step(Δt, time, parent_time)
    runway = parent_time - time
    return runway > (1 + 1e-6) * Δt || time + Δt == parent_time ? Δt : runway
end

# A prescribed parent is a function of time: it refreshes its state at the end of the step the child is
# about to take, and constrains that step not at all.
function advance_parent!(parent::AbstractPrescribedComponent, clock, Δt, parent_Δt)
    copy_clock!(parent.clock, clock, Δt)
    update_state!(parent)
    return nothing
end

function advance_parent!(parent::BoundaryPrescribedComponent, clock, Δt, parent_Δt)
    copy_clock!(parent.clock, clock, Δt)
    foreach(strip -> advance_parent!(strip, clock, Δt, parent_Δt), boundary_strips(parent))
    return nothing
end

# A parent with its own dynamics steps whenever the child has caught up with it, and reports the time
# the child may not step past.
function advance_parent!(parent, clock, Δt, parent_Δt)
    parent.clock.time ≤ clock.time && time_step!(parent, next_parent_Δt(parent_Δt, parent, Δt))
    return parent.clock.time
end

# The parent's own step. `nothing` keeps it in lockstep with the child.
next_parent_Δt(::Nothing, parent, Δt) = Δt
next_parent_Δt(parent_Δt::Number, parent, Δt) = parent_Δt

# A rule relative to the child's step, which holds a refinement ratio while the child's Δt adapts.
next_parent_Δt(rule::Function, parent, Δt) = rule(Δt)

# A fresh clock's `last_Δt` is `Inf`, so seed the wizard with the child's step instead.
function next_parent_Δt(wizard, parent, Δt)
    last_Δt = parent.clock.last_Δt
    return new_time_step(isfinite(last_Δt) ? last_Δt : Δt, wizard, parent)
end

# Set `clock` to the end of the step `Δt` that `source`'s model is about to take.
# Rebinding the fields is for host clocks; a traced clock (NumericalEarthReactantExt) writes through its
# own numbers instead, so the two clocks stay distinct program results.
function copy_clock!(clock, source, Δt)
    clock.time = source.time + Δt
    clock.iteration = source.iteration + 1
    clock.last_Δt = Δt
    clock.last_stage_Δt = Δt
    return nothing
end

# Checkpointing: only the child has prognostic state.
prognostic_state(nm::NestedModel) = prognostic_state(nm.child)
restore_prognostic_state!(nm::NestedModel, state) = restore_prognostic_state!(nm.child, state)
Base.summary(nm::NestedModel) = string("NestedModel(", summary(nm.parent), " → ", summary(nm.child), ")")

function Base.show(io::IO, nm::NestedModel)
    print(io, summary(nm), '\n',
              "├── parent: ", summary(nm.parent), '\n',
              "└── child:  ", summary(nm.child))
end
