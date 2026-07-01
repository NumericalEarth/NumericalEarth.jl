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
# (a shared reference). `time_step!` carries the new behavior: refresh the exchanger, step the child,
# then advance the parent to match. Child-specific state (`velocities`, `grid`, `dynamics`, ...) is
# reached explicitly through `nested.child`.

import Oceananigans
import Oceananigans: AbstractModel, fields, prognostic_state, restore_prognostic_state!, prognostic_fields, set!
import Oceananigans.Architectures: architecture
import Oceananigans.TimeSteppers: time_step!, update_state!
import Oceananigans.Simulations: iteration

"""
    NestedModel(parent, child::AbstractModel)

Pair a `child` Oceananigans model with a `parent` state (e.g. a 3D
`PrescribedAtmosphere`). The result is itself an `AbstractModel` whose
property access and integrator-protocol methods forward to the child, except
for `time_step!`, which advances the child and then ticks the parent to
match the child's new clock time.

Construct via [`child_model`](@ref) plus this wrapper, or use the
[`NestedSimulation`](@ref) convenience that bundles model construction and
`Simulation` wrapping.
"""
mutable struct NestedModel{P, M, X, C, TS, A} <: AbstractModel{TS, A}
    parent    :: P
    child     :: M
    exchanger :: X   # mediates parent state → child variables (see `exchange_state!`); `nothing` if unused
    clock     :: C   # shared reference to the child's clock (Simulation/run! read `model.clock`)
end

NestedModel(parent, child::AbstractModel{TS, A}, exchanger=nothing) where {TS, A} =
    NestedModel{typeof(parent), typeof(child), typeof(exchanger), typeof(child.clock), TS, A}(
        parent, child, exchanger, child.clock)

# Refresh the parent-derived child state held by the exchanger to the current `time`. The exchanger
# (a NumericalEarth object; see the Breeze extension's `StateExchanger`) recomputes the child
# prognostics on the parent grid as needed. Default no-op so a bare `NestedModel(parent, child)` — or
# one whose parent needs no state transform — just forwards.
exchange_state!(exchanger, time) = nothing

# Model-protocol dispatches, forwarded explicitly to the child.
fields(nm::NestedModel)            = fields(nm.child)
prognostic_fields(nm::NestedModel) = prognostic_fields(nm.child)
architecture(nm::NestedModel)      = architecture(nm.child)
iteration(nm::NestedModel)         = iteration(nm.child)

# Adaptive Δt: the wizard's `AdvectiveCFL` queries the simulation model, which is the
# `NestedModel` — forward to the child (the prognostic model whose grid/velocities set the CFL).
Oceananigans.Advection.cell_advection_timescale(nm::NestedModel) =
    Oceananigans.Advection.cell_advection_timescale(nm.child)

# `update_state!` may receive callbacks; forward verbatim. Refresh the parent-derived child state first
# so the child's boundary conditions / forcings see current values.
function update_state!(nm::NestedModel, callbacks=[]; kwargs...)
    exchange_state!(nm.exchanger, nm.clock.time)
    return update_state!(nm.child, callbacks; kwargs...)
end

# Setting the initial state targets the child's prognostics.
set!(nm::NestedModel, args...; kwargs...) = set!(nm.child, args...; kwargs...)

# The whole point of NestedModel: step the child, then advance the parent
# clock to match. `Simulation` calls `time_step!(model, Δt; callbacks=...)`.
function time_step!(nm::NestedModel, Δt; kwargs...)
    exchange_state!(nm.exchanger, nm.clock.time)
    time_step!(nm.child, Δt; kwargs...)

    Δt_parent = nm.child.clock.time - nm.parent.clock.time
    Δt_parent > 0 && time_step!(nm.parent, Δt_parent)

    return nothing
end

# Checkpointing: only the child has prognostic state.
prognostic_state(nm::NestedModel) = prognostic_state(nm.child)

restore_prognostic_state!(nm::NestedModel, state) = restore_prognostic_state!(nm.child, state)

Base.summary(nm::NestedModel) =
    string("NestedModel(", summary(nm.parent), " → ", summary(nm.child), ")")

function Base.show(io::IO, nm::NestedModel)
    print(io, summary(nm), '\n',
              "├── parent: ", summary(nm.parent), '\n',
              "└── child:  ", summary(nm.child))
end
