#####
##### child_simulation: one-call construction of a parent-driven child model
#####
#
# Despite the name (chosen for consistency with `ocean_simulation` /
# `atmosphere_simulation`), `child_simulation` returns an `AbstractModel` — the
# Breeze `atmosphere_simulation` helper returns a model, not a `Simulation`,
# and we follow that convention so the call sites compose cleanly:
#
#     child  = child_simulation(modeltype, grid, parent; …)
#     nested = NestedSimulation(parent, child; Δt, stop_time, …)
#
# Variable mapping (which parent FTS drives which child field) dispatches on
# `default_parent_variables(modeltype, parent)`; the underlying model
# constructor is selected by `_build_child_model(modeltype, grid; …)` — extend
# either via package extensions for new model types.

"""
    default_parent_variables(modeltype, parent) → NamedTuple

Return a `child_variable_name => FieldTimeSeries` mapping naming the parent
FTSs that drive each open boundary of a child model of type `modeltype`.
Extended by dispatch — add a method per child model type.
"""
function default_parent_variables end

# Oceananigans NonhydrostaticModel: velocity-form prognostic (u, v, w).
default_parent_variables(::Type{<:NonhydrostaticModel}, parent) =
    (u = parent.velocities.u, v = parent.velocities.v)

"""
    _build_child_model(modeltype, grid; kwargs...)

Construct the child model. Defaults to `modeltype(grid; kwargs...)`; the
Breeze package extension overrides this for `AtmosphereModel` to dispatch
through `atmosphere_simulation`, matching the established
`ocean_simulation` / `atmosphere_simulation` convention.
"""
_build_child_model(modeltype, grid; kwargs...) = modeltype(grid; kwargs...)

"""
    child_simulation(modeltype, grid, parent;
                     sides = (:west, :east, :south, :north),
                     schemes = NamedTuple(),
                     variables = default_parent_variables(modeltype, parent),
                     relaxation_rate = nothing,
                     relaxation_mask = 1,
                     model_kwargs...)

Construct a child model (of type `modeltype`) on `grid` with Open boundary
conditions driven by `parent`. Returns the constructed `AbstractModel`.

`parent_boundary_conditions` wires the BCs from `variables`; `model_kwargs`
are forwarded to the underlying constructor (see `_build_child_model`).

If `relaxation_rate` is supplied, [`parent_forcings`](@ref) builds an
Oceananigans `Relaxation` per variable that nudges the child interior toward
the parent's `FieldTimeSeries` at rate `relaxation_rate` and weighted by
`relaxation_mask` (default uniform). Per-variable values are supported via
`NamedTuple`s. The resulting forcings are merged with any `forcing` already
passed in `model_kwargs`.

Wrap with `NestedSimulation(parent, child; Δt, stop_time, …)` to integrate.
"""
function child_simulation(modeltype, grid, parent;
                          sides = (:west, :east, :south, :north),
                          schemes = NamedTuple(),
                          variables = default_parent_variables(modeltype, parent),
                          relaxation_rate = nothing,
                          relaxation_mask = 1,
                          model_kwargs...)

    bcs = parent_boundary_conditions(grid; variables, sides, schemes)

    # Pull any user-supplied `forcing` out so we can merge in the relaxation
    # forcings on the same field names without colliding.
    model_kwargs_nt = NamedTuple(model_kwargs)
    user_forcing   = get(model_kwargs_nt, :forcing, NamedTuple())
    rest_kwargs    = Base.structdiff(model_kwargs_nt, NamedTuple{(:forcing,)})

    forcing = if relaxation_rate === nothing
        user_forcing
    else
        relaxation = parent_forcings(; variables, rate = relaxation_rate, mask = relaxation_mask)
        merge(user_forcing, relaxation)
    end

    return _build_child_model(modeltype, grid;
                              boundary_conditions = bcs,
                              forcing,
                              rest_kwargs...)
end
