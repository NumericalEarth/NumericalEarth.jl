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

# On `ReactantState` the clock iteration is a traced scalar, so the generic
# `if iteration == 0` first-step guard can't branch inside a compiled loop. Make the
# hook a no-op and do the first-step refresh explicitly in `first_time_step!`, as
# Oceananigans does for its own Reactant models.
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
