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

# Any EarthSystemModel (every component combination) carried on `ReactantState`.
const ReactantESM{R, A, L, I, O, F, C} = Union{
    EarthSystemModel{R, A, L, I, O, F, C, <:ReactantState},
    EarthSystemModel{R, A, L, I, O, F, C, <:Distributed{ReactantState}},
}

# Raise the coupled-model kernels to StableHLO. Without `raise`, the GPU backend
# lowers the kernel launch through EnzymeXLA's kernel-call ABI rewrite
# (`rewriteKernelCallABI` → `gpu.LaunchFuncOp`), which segfaults on the coupled
# `update_state!` kernel. Raising is also what the differentiable examples use for
# the gradient, so the forward and adjoint passes share the same lowering.
function reconcile_state!(model::ReactantESM)
    @jit raise=true Oceananigans.initialize!(model.interfaces.exchanger, model)
    @jit raise=true Oceananigans.TimeSteppers.update_state!(model)
    return nothing
end

import NumericalEarth.EarthSystemModels: same_time_type

same_time_type(::Reactant.ConcretePJRTNumber{FT}, ::TT) where {FT, TT} = FT == TT
same_time_type(::Reactant.ConcretePJRTNumber{FT}, ::Reactant.ConcretePJRTNumber{TT}) where {FT, TT} = FT == TT
same_time_type(::FT, ::Reactant.ConcretePJRTNumber{TT}) where {FT, TT} = FT == TT

end # module NumericalEarthReactantExt
