module NumericalEarthReactantExt

using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed

using NumericalEarth: EarthSystemModel
using NumericalEarth.NestedModels: NestedModels, NestedModel

import Oceananigans
import Oceananigans.TimeSteppers: reconcile_state!

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

const ReactantNestedModel = NestedModel{<:Any, <:Any, <:Any, <:Any, <:Any, <:ReactantState}

# Inside an enclosing `@compile` the call is already being traced, so it runs as part of that program.
function NestedModels.execute!(f!, nm::ReactantNestedModel)
    if Reactant.within_compile()
        f!(nm)
    else
        @jit f!(nm)
    end
    return nothing
end

import NumericalEarth.EarthSystemModels: same_time_type

same_time_type(::Reactant.ConcreteRNumber{FT}, ::TT) where {FT,TT} = FT == TT
same_time_type(::Reactant.ConcreteRNumber{FT}, ::Reactant.ConcreteRNumber{TT}) where {FT,TT} = FT == TT
same_time_type(::FT, ::Reactant.ConcreteRNumber{TT}) where {FT,TT} = FT == TT

end # module NumericalEarthReactantExt
