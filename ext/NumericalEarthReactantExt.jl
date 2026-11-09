module NumericalEarthReactantExt

using Oceananigans: Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.TimeSteppers: Clock, update_state!, time_step!

using NumericalEarth: NumericalEarth, EarthSystemModel
using NumericalEarth.EarthSystemModels: EarthSystemModels, compute_land_fluxes!, initialize_land_surface!
using NumericalEarth.NestedModels: NestedModels

using Reactant: Reactant, @jit, @trace

const OceananigansReactantExt = Base.get_extension(
     Oceananigans, :OceananigansReactantExt
)

# Any EarthSystemModel (every component combination) carried on `ReactantState`.
const ReactantESM{R, A, L, I, O, F, C} = Union{
    EarthSystemModel{R, A, L, I, O, F, C, <:ReactantState},
    EarthSystemModel{R, A, L, I, O, F, C, <:Distributed{ReactantState}},
}

function Oceananigans.TimeSteppers.reconcile_state!(model::ReactantESM)
    @jit Oceananigans.initialize!(model.interfaces.exchanger, model)
    @jit Oceananigans.TimeSteppers.update_state!(model)
    return nothing
end

# On `ReactantState` the traced clock iteration can't drive the `iteration == 0`
# first-step guard in a compiled loop, so refresh explicitly in `first_time_step!`.
Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(::ReactantESM, Δt, callbacks) = nothing

function Oceananigans.TimeSteppers.first_time_step!(model::ReactantESM, Δt)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

function EarthSystemModels.maybe_initialize_land_surface!(model::ReactantESM)
    @trace track_numbers=false if model.clock.iteration == 0
        compute_land_fluxes!(model, 0)
        initialize_land_surface!(model.land)
    end
    return nothing
end

EarthSystemModels.same_time_type(::Reactant.ConcreteRNumber{FT}, ::TT) where {FT,TT} = FT == TT
EarthSystemModels.same_time_type(::Reactant.ConcreteRNumber{FT}, ::Reactant.ConcreteRNumber{TT}) where {FT,TT} = FT == TT
EarthSystemModels.same_time_type(::FT, ::Reactant.ConcreteRNumber{TT}) where {FT,TT} = FT == TT

# The device clock `Clock(grid)` gives on ReactantState, for constructors that build their clock before the grid
EarthSystemModels.default_clock(::ReactantState, FT) =
    Clock(; time = Reactant.ConcreteRNumber(0.0), iteration = Reactant.ConcreteRNumber(0),
            stage = 1, last_Δt = Reactant.ConcreteRNumber(Inf),
            last_stage_Δt = Reactant.ConcreteRNumber(Inf), kernel_time_type = FT)

# A nested parent following the child's clock inside a compiled step: write the child's values into the
# parent's own traced numbers, as Oceananigans' traced `tick!` does, rather than rebinding the parent's
# fields to the child's. Rebinding leaves both clocks holding one result buffer after the call, which the
# next call refuses to donate twice.
const TracedClock = Clock{<:Reactant.TracedRNumber}

function NestedModels.copy_clock!(clock::TracedClock, source)
    assign!(clock.time, source.time)
    assign!(clock.iteration, source.iteration)
    assign!(clock.last_Δt, source.last_Δt)
    assign!(clock.last_stage_Δt, source.last_stage_Δt)
    return nothing
end

assign!(number::Reactant.TracedRNumber, value) = (number.mlir_data = Reactant.promote_to(typeof(number), value).mlir_data)

# The zonal loading operator's batched matrix product as one dot_general
NumericalEarth.Oceans.batched_contract(K::Reactant.TracedRArray, X::Reactant.TracedRArray) =
    Reactant.Ops.dot_general(K, X; contracting_dimensions = ([2], [1]), batching_dimensions = ([3], [3]))

# Constants of the loading operators: no derivative, even when the operator is inside `Duplicated` arguments
NumericalEarth.Oceans.no_derivative(x::Reactant.TracedRArray) = Reactant.Ops.ignore_derivatives(x)
NumericalEarth.Oceans.kernel_eltype(K::Reactant.AnyTracedRArray) = Reactant.unwrapped_eltype(K)
NumericalEarth.Oceans.convert_eltype(FT, X::Reactant.TracedRArray) =
    Reactant.unwrapped_eltype(X) == FT ? X : Reactant.Ops.convert(Reactant.TracedRArray{FT, ndims(X)}, X)

end # module NumericalEarthReactantExt
