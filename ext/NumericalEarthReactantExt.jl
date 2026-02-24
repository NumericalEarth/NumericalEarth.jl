module NumericalEarthReactantExt

using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed

using NumericalEarth: EarthSystemModel

import Oceananigans
import Oceananigans.Models: initialization_update_state!

const OceananigansReactantExt = Base.get_extension(
     Oceananigans, :OceananigansReactantExt
)

const ReactantOSIM{I, A, O, F, C} = Union{
    EarthSystemModel{I, A, O, F, C, <:ReactantState},
    EarthSystemModel{I, A, O, F, C, <:Distributed{ReactantState}},
}

initialization_update_state!(model::ReactantOSIM) = nothing

end # module NumericalEarthReactantExt
