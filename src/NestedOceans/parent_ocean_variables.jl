#####
##### parent_ocean_variables: the parent series a nested ocean child is driven by
#####
#
# Dispatched on the parent type so a prognostic parent (a live `HydrostaticFreeSurfaceModel`) can join
# without touching the exchanger: `Interpolated` accepts a `Field` source as readily as a
# `FieldTimeSeries`, so only this mapping and `refresh_parent_state!` differ between the two.

"""
$(TYPEDSIGNATURES)

Return the parent state driving a nested ocean child as a `NamedTuple` of `u`, `v`, `T`, `S` and `η`.
"""
parent_ocean_variables(parent::PrescribedOcean) = (u = parent.velocities.u,
                                                   v = parent.velocities.v,
                                                   T = parent.temperature,
                                                   S = parent.salinity,
                                                   η = parent.free_surface)

# Reposition the parent's series ahead of the child's step. `NestedModel` steps the parent only after
# the child, so the exchanger — not the parent's own `time_step!` — brackets the upcoming step.
refresh_parent_state!(parent::PrescribedOcean, time) = update_prescribed_ocean_series!(parent, time)
