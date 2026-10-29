# Temporary monkey patches for the Reactant tests, each standing in for an upstream fix that has not
# landed. Every patch names what retires it: when that lands, delete the patch and rerun the tests;
# delete this file when no patch is left.

using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.NestedModels: NestedModel, exchange_state!

# Guarded so the file can be included by more than one test of a session
if !(@isdefined ReactantNestedModel)
    const ReactantNestedModel = NestedModel{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ReactantState}
end

# PATCH 1. The nested step computes `Δt_parent = child.clock.time - parent.clock.time` and branches on it to
# tick the parent's clock: a host branch on a traced value once the child's clock is a device clock, or
# inside a traced loop over steps. The parent's clock only drives its geopotential interpolation and its
# series windows, both pinned when the exchanger's window holds every parent snapshot, so on
# ReactantState the step advances the child alone.
# Retire when: the parent follows the child's clock (NumericalEarth/NumericalEarth.jl#817).
function Oceananigans.TimeSteppers.time_step!(nm::ReactantNestedModel, Δt; callbacks = ())
    exchange_state!(nm.exchanger, nm.clock.time + Δt)
    time_step!(nm.child, Δt; callbacks = (callbacks..., nm.child_callbacks...))
    return nothing
end
