module NumericalEarthReactantExt

using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed

using NumericalEarth: EarthSystemModel

import Oceananigans
import Oceananigans.TimeSteppers: reconcile_state!

const OceananigansReactantExt = Base.get_extension(
     Oceananigans, :OceananigansReactantExt
)

const ReactantOSIM{I, A, L, O, F, C} = Union{
    EarthSystemModel{I, A, L, O, F, C, <:ReactantState},
    EarthSystemModel{I, A, L, O, F, C, <:Distributed{ReactantState}},
}

reconcile_state!(model::ReactantOSIM) = nothing

end # module NumericalEarthReactantExt
