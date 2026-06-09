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
# Variable mapping (which parent FTS drives which child boundary) dispatches on
# `parent_variables(modeltype, parent)` — multi-dispatch on the (child, parent)
# pair. Methods live wherever both types are visible (often a package
# extension). Cross-realm pairs (e.g. ocean child × atmosphere parent) are
# intentionally undefined; the generic fallback throws an explanatory
# `ArgumentError` and the user must pass `variables=` explicitly to override.
#
# The underlying model constructor is selected by `build_child_model(modeltype, grid; …)`
# — also extended via package extensions for new model types.

"""
    parent_variables(child_modeltype, parent) → NamedTuple

Return the `child_variable_name => interpolatable_source` mapping naming
the parent fields that drive each child boundary, where the source may be
a `FieldTimeSeries` (prescribed parents) or an `AbstractField` (prognostic
parents).

Dispatch on the **(child_modeltype, parent_type)** pair — define a method
per pair, in whichever package can see both types. The generic fallback
errors so cross-realm pairs (e.g. an ocean child with an atmosphere parent)
fail loudly at construction time. Users can always sidestep by passing
`variables=` directly to [`child_simulation`](@ref).

The exact contents of the NamedTuple — what fields, whether they are
density-weighted, etc. — are determined by what the child consumes and what
the parent provides. For example, Breeze `AtmosphereModel` consumes
prognostic momentum (`ρu`, `ρv`, …), so a `PrescribedAtmosphere` parent
populated for that use case must store ρ·u in its velocity slots; the
method body documents the convention.
"""
function parent_variables end

# Generic fallback — errors with a descriptive message.
function parent_variables(child_modeltype, parent)
    throw(ArgumentError(string(
        "No `parent_variables` mapping is defined for ",
        "child=$(child_modeltype) with parent=$(typeof(parent)).\n",
        "\n",
        "Either:\n",
        "  • define `parent_variables(::Type{<:$(child_modeltype)}, ",
        "parent::$(typeof(parent)))` in the package where both types are visible, or\n",
        "  • pass `variables = (...)` explicitly to `child_simulation`.\n",
        "\n",
        "Note: cross-realm nesting (e.g. an ocean child driven by an atmosphere ",
        "parent) is intentionally unsupported as a default — the variable units ",
        "do not match. Define a custom mapping only if you own the semantics."
    )))
end

"""
    build_child_model(modeltype, grid; kwargs...)

Construct the child model. Defaults to `modeltype(grid; kwargs...)`; package
extensions override this for specific model types (e.g. the Breeze ext routes
`AtmosphereModel` through `atmosphere_simulation` so Breeze's default
advection / microphysics apply).
"""
build_child_model(modeltype, grid; kwargs...) = modeltype(grid; kwargs...)

"""
    child_simulation(modeltype, grid, parent;
                     sides = (:west, :east, :south, :north),
                     schemes = NamedTuple(),
                     variables = parent_variables(modeltype, parent),
                     relaxation_rate = nothing,
                     relaxation_mask = 1,
                     model_kwargs...)

Construct a child model (of type `modeltype`) on `grid` with Open boundary
conditions driven by `parent`. Returns the constructed `AbstractModel`.

`parent_boundary_conditions` wires the BCs from `variables`; `model_kwargs`
are forwarded to the underlying constructor (see `build_child_model`).

If `relaxation_rate` is supplied, [`parent_forcings`](@ref) builds an
Oceananigans `Relaxation` per variable that nudges the child interior toward
the parent at rate `relaxation_rate`, weighted by `relaxation_mask` (default
uniform). Per-variable values are supported via `NamedTuple`s. The resulting
forcings are merged with any `forcing` already passed in `model_kwargs`.

Wrap with `NestedSimulation(parent, child; Δt, stop_time, …)` to integrate.
"""
function child_simulation(modeltype, grid, parent;
                          sides = (:west, :east, :south, :north),
                          schemes = NamedTuple(),
                          variables = parent_variables(modeltype, parent),
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

    return build_child_model(modeltype, grid;
                              boundary_conditions = bcs,
                              forcing,
                              rest_kwargs...)
end
