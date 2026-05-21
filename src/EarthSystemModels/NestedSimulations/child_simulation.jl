#####
##### child_simulation: one-call construction of a parent-driven child
#####
#
# Wraps `parent_boundary_conditions` + child-model construction + `Simulation` +
# `NestedSimulation` in a single call. The variable mapping (which parent FTS
# drives which child field) is chosen by dispatch on the child model type via
# `default_parent_variables` — extend that to add support for new model types
# (e.g. via package extensions for Breeze, etc.).

"""
    default_parent_variables(modeltype, parent) → NamedTuple

Return a `child_variable_name => FieldTimeSeries` mapping naming the parent
FTSs that drive each open boundary of a child model of type `modeltype`.
Extended by dispatch — add a method per child model type.
"""
function default_parent_variables end

# Oceananigans NonhydrostaticModel: velocity-form prognostic (u, v, w).
# Pulls horizontal velocity components directly from `parent.velocities`.
default_parent_variables(::Type{<:NonhydrostaticModel}, parent) =
    (u = parent.velocities.u, v = parent.velocities.v)

"""
    child_simulation(modeltype, grid, parent;
                     Δt,
                     stop_time = Inf,
                     verbose = false,
                     sides = (:west, :east, :south, :north),
                     schemes = NamedTuple(),
                     variables = default_parent_variables(modeltype, parent),
                     relaxation_rate = nothing,
                     relaxation_mask = 1,
                     parent_update_schedule = IterationInterval(1),
                     model_kwargs...)

Construct a `NestedSimulation` whose child model (of type `modeltype`) is
created on `grid` with Open boundary conditions driven by `parent`.

`parent_boundary_conditions` wires the BCs from `variables`; `model_kwargs`
are forwarded to the child model constructor; `Δt`, `stop_time`, `verbose` go
to the child `Simulation`.

If `relaxation_rate` is supplied, [`parent_forcings`](@ref) builds an
Oceananigans `Relaxation` per variable that nudges the child interior toward
the parent's `FieldTimeSeries` at rate `relaxation_rate` and weighted by
`relaxation_mask` (default uniform). Per-variable values are supported via
`NamedTuple`s. The resulting forcings are merged with any `forcing` already
passed in `model_kwargs`.

Returns the `NestedSimulation`.
"""
function child_simulation(modeltype, grid, parent;
                          Δt,
                          stop_time = Inf,
                          verbose = false,
                          sides = (:west, :east, :south, :north),
                          schemes = NamedTuple(),
                          variables = default_parent_variables(modeltype, parent),
                          relaxation_rate = nothing,
                          relaxation_mask = 1,
                          parent_update_schedule = IterationInterval(1),
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

    model = modeltype(grid; boundary_conditions = bcs, forcing, rest_kwargs...)
    simulation = Simulation(model; Δt, stop_time, verbose)
    return NestedSimulation(parent, simulation; parent_update_schedule)
end
