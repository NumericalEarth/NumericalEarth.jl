module NumericalEarthReactantExt

using Oceananigans: Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.TimeSteppers: update_state!, time_step!

using NumericalEarth: EarthSystemModel
using NumericalEarth.EarthSystemModels: EarthSystemModels, compute_land_fluxes!, initialize_land_surface!

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

end # module NumericalEarthReactantExt
