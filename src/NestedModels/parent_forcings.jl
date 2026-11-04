#####
##### parent_forcings: interior relaxation toward a parent's FieldTimeSeries
#####
#
# Returns a `NamedTuple` of Oceananigans `Relaxation` forcings — one per child
# variable — whose `target(x, y, z, t)` interpolates the corresponding parent
# `FieldTimeSeries`. The `mask` parameter (a callable, Field, or any Oceananigans-
# compatible mask) decides where in the child domain the relaxation acts;
# typical use is a sponge layer near the open boundaries.

using Oceananigans.Fields: Field
using Adapt: Adapt, adapt

# A live parent `Field` as a relaxation target, sampled at `interpolation_node`.
struct InterpolatedParentTarget{F, L, G}
    field    :: F
    location :: L
    grid     :: G
end

InterpolatedParentTarget(field) = InterpolatedParentTarget(field, instantiated_location(field), field.grid)

Adapt.adapt_structure(to, t::InterpolatedParentTarget) =
    InterpolatedParentTarget(adapt(to, t.field), t.location, adapt(to, t.grid))

Oceananigans.OutputReaders.extract_field_time_series(::InterpolatedParentTarget) = ()

Base.summary(t::InterpolatedParentTarget) = "parent-sampled " * summary(t.field)

@inline Oceananigans.Forcings.evaluate_target(i, j, k, grid, t::InterpolatedParentTarget, ℓ, time) =
    interpolate(interpolation_node(i, j, k, grid, ℓ..., t.grid), t.field, t.location, t.grid)

relaxation_target(source) = source
relaxation_target(field::Field) = InterpolatedParentTarget(field)

"""
    parent_forcings(; variables, rate, mask = 1)

Build a `NamedTuple` of `Relaxation` forcings driving the child interior toward
the parent's values.

Arguments
=========

- `variables`: a `NamedTuple` mapping child field names to the parent
  `FieldTimeSeries` — or, for a live parent, the parent `Field` — that should drive them, e.g.
  `(u = parent.velocities.u, v = parent.velocities.v)` — same convention as
  [`parent_boundary_conditions`](@ref).

- `rate`: the relaxation rate (s⁻¹). May be a scalar (applied to all variables)
  or a `NamedTuple` keyed by child variable.

- `mask`: a callable `(x, y, z) -> [0, 1]`, a `Field`, a scalar, or an
  Oceananigans-supplied mask object (e.g. `GaussianMask`). May be a `NamedTuple`
  for per-variable masks. Default `1` relaxes uniformly across the domain.

Returns
=======

A `NamedTuple` of `Oceananigans.Forcings.Relaxation` ready to splat into the
`forcing` kwarg of an Oceananigans / Breeze model constructor.
"""
function parent_forcings(; variables, rate, mask=1)
    field_pairs = []
    for (child_name, source) in pairs(variables)
        r = rate isa NamedTuple ? getproperty(rate, child_name) : rate
        m = mask isa NamedTuple ? getproperty(mask, child_name) : mask
        # `Relaxation` accepts a `FieldTimeSeries` as `target` directly;
        # `materialize_forcing` wraps it in a `FieldTimeSeriesTarget` that
        # handles space/time interpolation and GPU adaptation.
        push!(field_pairs, child_name => Relaxation(rate = r, mask = m, target = relaxation_target(source)))
    end

    return (; field_pairs...)
end
