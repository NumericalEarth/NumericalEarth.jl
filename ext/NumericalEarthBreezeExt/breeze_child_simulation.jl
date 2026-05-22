#####
##### child_simulation dispatch for Breeze.AtmosphereModel
#####
#
# Two extensions to NumericalEarth.NestedSimulations:
#
# 1. `parent_variables(::Type{<:AtmosphereModel}, ::PrescribedAtmosphere)` —
#    declares the (child, parent) pair mapping. Returns a NamedTuple naming
#    the parent FTSs that drive the child's momentum BCs.
#
#    Density convention (important): Breeze's prognostic is density-weighted
#    momentum (ρu, ρv, ρw). The BC value Oceananigans writes into ρu at the
#    boundary is exactly the value of the source we hand it; so the parent's
#    velocity FTS slots are interpreted as **momentum** values (ρ̄·u). The user
#    populating a `PrescribedAtmosphere` for use with Breeze is expected to
#    pre-multiply by ρ̄ when calling `set!`. See [[nested-simulations-design]]
#    and the deferred PrescribedAtmosphere refactor (issue #266) for the
#    slot-naming followup.
#
# 2. `_build_child_model(::Type{<:AtmosphereModel}, grid; …)` — routes
#    `child_simulation(AtmosphereModel, …)` through `atmosphere_simulation`
#    instead of constructing `AtmosphereModel` directly, so Breeze's default
#    advection / microphysics settings apply unless the user overrides them.

import NumericalEarth.EarthSystemModels.NestedSimulations: parent_variables,
                                                            _build_child_model

parent_variables(::Type{<:Breeze.AtmosphereModel}, parent::NumericalEarth.PrescribedAtmosphere) =
    (ρu = parent.velocities.u, ρv = parent.velocities.v)

_build_child_model(::Type{<:Breeze.AtmosphereModel}, grid; kwargs...) =
    NumericalEarth.Atmospheres.atmosphere_simulation(grid; kwargs...)
