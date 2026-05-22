#####
##### child_model dispatch for Breeze.AtmosphereModel
#####
#
# Breeze's prognostic is density-weighted momentum (ρu, ρv, ρw). We assume the
# parent's velocity FTS slots already hold momentum values — i.e. the user has
# `set!`-ed `parent.velocities.u = ρ̄·u`, etc. — so the BC values land on the
# prognostic momentum at the boundary face directly. PR #722 in Breeze
# propagates these momentum BCs to the derived velocities.

import NumericalEarth.EarthSystemModels.NestedSimulations: default_parent_variables

default_parent_variables(::Type{<:Breeze.AtmosphereModel}, parent) =
    (ρu = parent.velocities.u, ρv = parent.velocities.v)
