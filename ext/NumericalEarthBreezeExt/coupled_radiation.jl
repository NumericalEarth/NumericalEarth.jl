#####
##### `CoupledRadiation` — a thin proxy that satisfies Breeze's radiation
##### contract (`update_radiation!`, `radiation_flux_divergence`) without
##### owning any allocations of its own. In the coupled configuration, the
##### real `Breeze.RadiativeTransferModel` lives at `EarthSystemModel.radiation`
##### and a `CoupledRadiation` lives at `atmosphere.radiation`, aliasing the
##### RTM's `flux_divergence` so Breeze's tendency machinery and the RTM's
##### update path share the same memory.
#####
##### The skeleton form `CoupledRadiation()` carries `Nothing` for both fields;
##### Breeze's tendency code reads `radiation_flux_divergence(::Nothing) = nothing`
##### and the inline `@inline radiation_flux_divergence(i,j,k,grid,::Nothing) =
##### zero(eltype(grid))` makes the radiative contribution literally zero. The
##### atmosphere runs radiatively-decoupled in this state and allocates nothing
##### extra. The materialized form holds a back-reference to the RTM and aliases
##### its `flux_divergence`; `update_radiation!` then delegates to the RTM,
##### preserving its schedule.
#####

using Accessors: @set
using NumericalEarth.NestedModels: NestedModel

"""
$(TYPEDSIGNATURES)

A thin proxy for the radiation interface that Breeze's `AtmosphereModel`
expects. Holds an aliased `flux_divergence` field plus a back-reference to
the source `RadiativeTransferModel` so that the atmosphere's
`update_radiation!` call can delegate without the atmosphere knowing about
the real RTM.

Defaults to the *skeleton* form (`flux_divergence = nothing`,
`radiative_transfer_model = nothing`), which makes the atmosphere
radiatively decoupled with zero extra allocation.
`materialize_earth_system_radiation!` replaces the skeleton in-place when an
`EarthSystemModel` is constructed with a non-`nothing` radiation.
"""
struct CoupledRadiation{F, R}
    flux_divergence          :: F
    radiative_transfer_model :: R
end

CoupledRadiation() = CoupledRadiation(nothing, nothing)
CoupledRadiation(radiative_transfer_model::Breeze.RadiativeTransferModel) =
    CoupledRadiation(radiative_transfer_model.flux_divergence, radiative_transfer_model)

Base.summary(::CoupledRadiation{Nothing, Nothing}) = "CoupledRadiation (skeleton)"
Base.summary(::CoupledRadiation) = "CoupledRadiation (materialized)"

# Breeze tendency hook: hand back the (possibly aliased) flux-divergence field.
# The (i,j,k,grid,::Nothing) inline in Breeze returns zero, so the skeleton
# form contributes nothing to the tendency without any branching here.
Breeze.AtmosphereModels.radiation_flux_divergence(r::CoupledRadiation) = r.flux_divergence

# Breeze update hook: skeleton form does nothing; materialized form delegates
# to the RTM's schedule-aware `update_radiation!`, which fires the RRTMGP
# solve when due and writes into `radiative_transfer_model.flux_divergence` —
# same memory as `r.flux_divergence`, so the atmosphere sees the update on
# the next tendency.
Breeze.AtmosphereModels.update_radiation!(::CoupledRadiation{Nothing, Nothing}, model) = nothing
Breeze.AtmosphereModels.update_radiation!(r::CoupledRadiation, model) =
    Breeze.AtmosphereModels.update_radiation!(r.radiative_transfer_model, model)

# Time step at the EarthSystemModel level is a no-op for the RTM: the
# atmosphere's own `update_state!` (which runs each step) drives the radiation
# update via the proxy. The ESM time stepper's existing
# `!isnothing(radiation) && time_step!(radiation, Δt)` line then does nothing.
Oceananigans.TimeSteppers.time_step!(::Breeze.RadiativeTransferModel, Δt) = nothing

#####
##### Materialize a CoupledRadiation skeleton inside an atmosphere Simulation
#####

# Default (no Breeze RTM): leave the skeleton in place.
NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:Breeze.AtmosphereModel}, ::Nothing) = atmosphere

# Real RTM: replace the AtmosphereModel's CoupledRadiation with one that
# aliases radiative_transfer_model.flux_divergence. The atmosphere is now coupled.
function NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(
        atmosphere               :: Simulation{<:Breeze.AtmosphereModel},
        radiative_transfer_model :: Breeze.RadiativeTransferModel)
    materialized = CoupledRadiation(radiative_transfer_model)
    return @set atmosphere.model.radiation = materialized
end

# Nested atmosphere: materialize the child, then rebuild the (concretely-typed) nest and
# Simulation around it; all child fields are shared by reference.
function NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(
        atmosphere               :: Simulation{<:NestedModel{<:Any, <:Breeze.AtmosphereModel}},
        radiative_transfer_model :: Breeze.RadiativeTransferModel)
    nest = atmosphere.model
    child = nest.child
    child = @set child.radiation = CoupledRadiation(radiative_transfer_model)
    nest = NestedModel(nest.parent, child, nest.exchanger)
    return Simulation(nest; Δt = atmosphere.Δt)
end
