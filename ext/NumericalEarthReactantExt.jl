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

# On `ReactantState` the traced clock iteration can't drive the `iteration == 0`
# first-step guard in a compiled loop, so refresh explicitly in `first_time_step!`.
maybe_prepare_first_time_step!(::ReactantESM, Δt, callbacks) = nothing

function first_time_step!(model::ReactantESM, Δt)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

import NumericalEarth.EarthSystemModels: same_time_type, default_clock
using Oceananigans.TimeSteppers: Clock

same_time_type(::Reactant.ConcreteRNumber{FT}, ::TT) where {FT,TT} = FT == TT
same_time_type(::Reactant.ConcreteRNumber{FT}, ::Reactant.ConcreteRNumber{TT}) where {FT,TT} = FT == TT
same_time_type(::FT, ::Reactant.ConcreteRNumber{TT}) where {FT,TT} = FT == TT

# The device clock `Clock(grid)` gives on ReactantState, for constructors that build their clock before the grid
default_clock(::ReactantState, FT) = Clock(; time = Reactant.ConcreteRNumber(0.0), iteration = Reactant.ConcreteRNumber(0),
                                           stage = 1, last_Δt = Reactant.ConcreteRNumber(Inf),
                                           last_stage_Δt = Reactant.ConcreteRNumber(Inf), kernel_time_type = FT)

end # module NumericalEarthReactantExt
