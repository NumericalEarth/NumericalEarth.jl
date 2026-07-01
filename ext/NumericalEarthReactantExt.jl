module NumericalEarthReactantExt

using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed

using NumericalEarth: EarthSystemModel

import Oceananigans
import Oceananigans.TimeSteppers: reconcile_state!, maybe_prepare_first_time_step!
using Oceananigans.TimeSteppers: update_state!

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

# The generic `maybe_prepare_first_time_step!` reconciles the auxiliary/flux state with the
# prognostic fields on the first step only, guarded by `if model.clock.iteration == 0`. On
# `ReactantState` the iteration is a traced scalar, so a plain `if` cannot branch on it
# inside a compiled loop and the guard silently never fires: prognostic state `set!` after
# construction is not reconciled, and the first compiled step uses stale construction-time
# fluxes. Use Reactant's traced conditional so the refresh still runs only on the first
# step (no cost on subsequent steps). `update_state!` is referenced unqualified because the
# `@trace` branch body is lifted into a closure where module-qualified names do not resolve.
function maybe_prepare_first_time_step!(model::ReactantESM, Δt, callbacks)
    Reactant.@trace if model.clock.iteration == 0
        update_state!(model, callbacks)
    end
    return nothing
end

import NumericalEarth.EarthSystemModels: same_time_type

same_time_type(::Reactant.ConcretePJRTNumber{FT}, ::TT) where {FT, TT} = FT == TT
same_time_type(::Reactant.ConcretePJRTNumber{FT}, ::Reactant.ConcretePJRTNumber{TT}) where {FT, TT} = FT == TT
same_time_type(::FT, ::Reactant.ConcretePJRTNumber{TT}) where {FT, TT} = FT == TT

end # module NumericalEarthReactantExt
