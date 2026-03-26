module NumericalEarthReactantExt

using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed

using NumericalEarth: EarthSystemModel

import Oceananigans

const OceananigansReactantExt = Base.get_extension(
     Oceananigans, :OceananigansReactantExt
)

const ReactantOSIM{I, A, O, F, C} = Union{
    EarthSystemModel{I, A, O, F, C, <:ReactantState},
    EarthSystemModel{I, A, O, F, C, <:Distributed{ReactantState}},
}

end # module NumericalEarthReactantExt
