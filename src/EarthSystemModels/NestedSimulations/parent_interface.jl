#####
##### Duck-typed parent interface
#####
#
# A "parent" in a NestedSimulation is anything that satisfies:
#
#   parent_clock(parent)               :: Clock
#   parent_field(parent, name)         :: Field, FieldTimeSeries, or callable
#   parent_time_step!(parent, Δt)
#   parent_update_state!(parent)
#   parent_interpolate(field, X, t)    :: scalar
#
# Generic methods for Oceananigans `Simulation` and for the `FieldTimeSeries` /
# `AbstractField` types returned by `parent_field` live here. Type-specific
# methods (e.g. for `PrescribedAtmosphere`) live next to the type they dispatch
# on, since those modules load after `NestedSimulations`.

function parent_clock end
function parent_field end
function parent_time_step! end
function parent_update_state! end

# ---------------------------------------------------------------------------
# Oceananigans Simulation: a Simulation can play the role of a prognostic
# parent. v1 doesn't ship a fully exercised prognostic-parent code path, but
# the interface is provided so the wiring is consistent when it does.

parent_clock(sim::Simulation) = sim.model.clock
parent_field(sim::Simulation, name::Symbol) = getproperty(fields(sim.model), name)
parent_time_step!(sim::Simulation, Δt) = time_step!(sim, Δt)
parent_update_state!(sim::Simulation) = update_state!(sim.model)

# ---------------------------------------------------------------------------
# Parent-state interpolation. Dispatches on what `parent_field` returns.

@inline function parent_interpolate(fts::FieldTimeSeries, X, t)
    return Oceananigans.Fields.interpolate(X, Time(t), fts,
                                           instantiated_location(fts),
                                           fts.grid)
end

@inline function parent_interpolate(field::AbstractField, X, t)
    return Oceananigans.Fields.interpolate(X, field,
                                           instantiated_location(field),
                                           field.grid)
end

# A bare callable (x, y, z, t) — useful for analytic / mocked parents in tests
# even though the v1 default path uses set!-populated FieldTimeSeries.
# Restricted to Function so `FieldTimeSeries` / `AbstractField` keep dispatching
# to the methods above (avoiding method ambiguity).
@inline parent_interpolate(f::Function, X::Tuple{Any,Any,Any}, t) = f(X[1], X[2], X[3], t)
@inline parent_interpolate(f::Function, X::Tuple{Any,Any},     t) = f(X[1], X[2],       t)
