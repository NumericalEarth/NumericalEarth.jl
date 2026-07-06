module NumericalEarthReactantExt

using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed

using NumericalEarth: EarthSystemModel

import Oceananigans
import Oceananigans.TimeSteppers: reconcile_state!, maybe_prepare_first_time_step!, first_time_step!
using Oceananigans.TimeSteppers: update_state!, time_step!

const OceananigansReactantExt = Base.get_extension(
     Oceananigans, :OceananigansReactantExt
)

# Any EarthSystemModel (every component combination) carried on `ReactantState`.
const ReactantESM{R, A, L, I, O, F, C} = Union{
    EarthSystemModel{R, A, L, I, O, F, C, <:ReactantState},
    EarthSystemModel{R, A, L, I, O, F, C, <:Distributed{ReactantState}},
}

function reconcile_state!(model::ReactantESM)
    @jit Oceananigans.initialize!(model.interfaces.exchanger, model)
    @jit Oceananigans.TimeSteppers.update_state!(model)
    return nothing
end

# Mirror Oceananigans' Reactant paradigm. The generic `maybe_prepare_first_time_step!` runs
# on every step (called from the top of `time_step!`) and refreshes the auxiliary/flux state
# only on the first step, guarded by `if model.clock.iteration == 0`. On `ReactantState` the
# iteration is a traced scalar, so that guard cannot branch inside a compiled loop. Rather
# than trace a first-step-only conditional into every iteration, we make the hook a no-op and
# do the refresh explicitly and unconditionally in `first_time_step!` — matching how
# Oceananigans handles its own Reactant models, and keeping the compiled stepping loop lean.
maybe_prepare_first_time_step!(::ReactantESM, Δt, callbacks) = nothing

function first_time_step!(model::ReactantESM, Δt)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

import NumericalEarth.EarthSystemModels: same_time_type

same_time_type(::Reactant.ConcretePJRTNumber{FT}, ::TT) where {FT, TT} = FT == TT
same_time_type(::Reactant.ConcretePJRTNumber{FT}, ::Reactant.ConcretePJRTNumber{TT}) where {FT, TT} = FT == TT
same_time_type(::FT, ::Reactant.ConcretePJRTNumber{TT}) where {FT, TT} = FT == TT

end # module NumericalEarthReactantExt
