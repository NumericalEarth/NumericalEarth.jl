#####
##### Transform-aware interpolation: `interpolate(f, …)` applies `f` to each source value
##### *before* the weighted blend and `inverse_transform(f)` to the result. With `f = log`
##### this is log-space (geometric-mean-like) interpolation, which is accurate for fields that
##### vary exponentially (pressure, density) and — because the ideal gas law is linear in logs —
##### preserves EOS-consistency. `f = identity` reproduces Oceananigans' `interpolate` exactly.
#####
##### The lazy `log(field)` `AbstractField` trick only works for field→field interpolation; it
##### cannot reach the `FieldTimeSeries` space+time path (a `UnaryOperation` has no time axis and
##### is not a `FlavorOfFTS`). This routes the transform through the shared low-level blend
##### `_interpolate`, so it works for both Fields and FieldTimeSeries.
#####
##### LOCAL SHIM. This is destined to move upstream into Oceananigans' `interpolate` (a leading
##### `::Function` argument threaded into `_interpolate`); once that lands this file is deleted.
##### See `CLEANUP_PLAN_era5_breeze.md`.
#####

using Adapt: Adapt, adapt
using Oceananigans.Fields: FractionalIndices, flatten_node, interpolator, _interpolate
using Oceananigans.Grids: topology, Flat
using Oceananigans.OutputReaders: FlavorOfFTS, TimeInterpolator, memory_index

# Inverse-transform trait: the inverse applied to the (transform-space) blend. Defined only for
# transforms with an exact analytic inverse; an unlisted `f` is a deliberate "no inverse known".
@inline inverse_transform(::typeof(identity)) = identity
@inline inverse_transform(::typeof(log))      = exp
@inline inverse_transform(::typeof(log2))     = exp2
@inline inverse_transform(::typeof(log10))    = exp10

# Lazy elementwise transform of a source array: reads return `f(parent[I...])`, so the existing
# `_interpolate` blends `f`-transformed values without copying. GPU-safe when `f` is (log/exp are).
struct TransformedSource{T, N, F, A} <: AbstractArray{T, N}
    f    :: F
    data :: A
end

@inline TransformedSource(f::F, data::A) where {F, A} =
    TransformedSource{eltype(A), ndims(A), F, A}(f, data)

@inline Base.size(t::TransformedSource)  = size(t.data)
@inline Base.axes(t::TransformedSource)  = axes(t.data)
@inline Base.getindex(t::TransformedSource, I::Vararg{Int}) = t.f(@inbounds t.data[I...])

Adapt.adapt_structure(to, t::TransformedSource) = TransformedSource(t.f, adapt(to, t.data))

#####
##### interpolate(f, …) — field→node and FieldTimeSeries→node-at-time
#####

# Field source, interpolated in space at `at_node`.
@inline function Oceananigans.Fields.interpolate(f::Base.Callable, at_node, from_field, from_loc, from_grid)
    fidx = FractionalIndices(at_node, from_grid, from_loc...)
    ix = interpolator(fidx.i)
    iy = interpolator(fidx.j)
    iz = interpolator(fidx.k)
    blended = _interpolate(TransformedSource(f, from_field), ix, iy, iz)
    return inverse_transform(f)(blended)
end

# FieldTimeSeries source, interpolated in space + time. Mirrors Oceananigans'
# `interpolate(to_node, ::Time, ::FlavorOfFTS, …)` but with the transform: `f` is applied per
# source value (in both time slots), the slots are linearly blended in time (i.e. in transform
# space), and `inverse_transform(f)` is applied once to the result.
@inline function Oceananigans.Fields.interpolate(f::Base.Callable, to_node, to_time_index::Time,
                                                  from_fts::FlavorOfFTS, from_loc, from_grid)
    data          = from_fts.data
    times         = from_fts.times
    backend       = from_fts.backend
    time_indexing = from_fts.time_indexing

    interp = TimeInterpolator(time_indexing, times, to_time_index.time)
    node   = flatten_node(to_node...)

    fi = topology(from_grid) === (Flat, Flat, Flat) ?
         FractionalIndices(nothing, nothing, nothing) :
         FractionalIndices(node, from_grid, from_loc...)

    ix = interpolator(fi.i)
    iy = interpolator(fi.j)
    iz = interpolator(fi.k)

    ñ  = interp.fractional_index
    n₁ = convert(Int, interp.first_index)
    n₂ = convert(Int, interp.second_index)
    Nt = convert(Int, interp.length)
    m₁ = memory_index(backend, time_indexing, Nt, n₁)
    m₂ = memory_index(backend, time_indexing, Nt, n₂)

    tdata = TransformedSource(f, data)
    ψ₁ = _interpolate(tdata, ix, iy, iz, m₁)
    ψ₂ = _interpolate(tdata, ix, iy, iz, m₂)
    ψ̃  = ψ₂ * ñ + ψ₁ * (1 - ñ)

    return inverse_transform(f)(ifelse(n₁ == n₂, ψ₁, ψ̃))
end
