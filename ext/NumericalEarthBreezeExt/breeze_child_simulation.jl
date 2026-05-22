#####
##### child_simulation dispatch for Breeze.AtmosphereModel
#####
#
# Breeze's prognostic is density-weighted momentum (ρu, ρv, ρw). We assume the
# parent's velocity FTS slots already hold momentum values — i.e. the user has
# `set!`-ed `parent.velocities.u = ρ̄·u`, etc. — so the BC values land on the
# prognostic momentum at the boundary face directly. PR #722 in Breeze
# propagates these momentum BCs to the derived velocities.
#
# We override `_build_child_model` so that `child_simulation(AtmosphereModel, …)`
# routes through `atmosphere_simulation(grid; …)` — matching the established
# `*_simulation` helper convention — instead of constructing the `AtmosphereModel`
# directly. That way the call picks up Breeze's default advection / microphysics
# settings when the user doesn't override them.

import NumericalEarth.EarthSystemModels.NestedSimulations: default_parent_variables,
                                                            _build_child_model

default_parent_variables(::Type{<:Breeze.AtmosphereModel}, parent) =
    (ρu = parent.velocities.u, ρv = parent.velocities.v)

_build_child_model(::Type{<:Breeze.AtmosphereModel}, grid; kwargs...) =
    NumericalEarth.Atmospheres.atmosphere_simulation(grid; kwargs...)
