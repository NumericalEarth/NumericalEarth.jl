#####
##### parent_forcings: interior relaxation toward a parent's FieldTimeSeries
#####
#
# Returns a `NamedTuple` of Oceananigans `Relaxation` forcings — one per child
# variable — whose `target(x, y, z, t)` interpolates the corresponding parent
# `FieldTimeSeries`. The `mask` parameter (a callable, Field, or any Oceananigans-
# compatible mask) decides where in the child domain the relaxation acts;
# typical use is a sponge layer near the open boundaries.

using Oceananigans: instantiated_location
using Oceananigans.OutputReaders: FieldTimeSeries
using Adapt: Adapt, adapt

# A parent `FieldTimeSeries` as a relaxation target, sampled at `interpolation_node`.
struct ParentTarget{F, G}
    field_time_series :: F
    grid              :: G
end

Adapt.adapt_structure(to, t::ParentTarget) = ParentTarget(adapt(to, t.field_time_series), adapt(to, t.grid))

Base.summary(t::ParentTarget) = "parent-sampled " * summary(t.field_time_series)

@inline Oceananigans.Forcings.evaluate_target(i, j, k, grid, t::ParentTarget, ℓ, time) =
    interpolate(interpolation_node(i, j, k, grid, ℓ..., t.grid), Time(time), t.field_time_series,
                instantiated_location(t.field_time_series), t.grid)

relaxation_target(source) = source
relaxation_target(fts::FieldTimeSeries) = ParentTarget(fts, fts.grid)

"""
$(SIGNATURES)

Build a `NamedTuple` of `Relaxation` forcings driving the child interior toward
the parent's `FieldTimeSeries` values.

Arguments
=========

- `variables`: a `NamedTuple` mapping child field names to the parent
  `FieldTimeSeries` that should drive them, e.g.
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
    for (child_name, fts) in pairs(variables)
        r = rate isa NamedTuple ? getproperty(rate, child_name) : rate
        m = mask isa NamedTuple ? getproperty(mask, child_name) : mask
        push!(field_pairs, child_name => Relaxation(rate = r, mask = m, target = relaxation_target(fts)))
    end

    return (; field_pairs...)
end
