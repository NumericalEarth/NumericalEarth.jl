#####
##### interpolate(func, …): map each source value through `func` *before* the weighted blend, in the
##### spirit of `mean(func, itr)` — the result stays in func-space (NO inverse is applied). With
##### `func = log`, `exp(interpolate(log, …))` is a positivity-preserving, geometric-mean
##### interpolation, accurate for exponentially-varying fields (pressure, density). `func = identity`
##### reproduces the plain `interpolate` exactly. Works for `Field`s AND `FieldTimeSeries` (the lazy
##### `log(field)` `AbstractField` trick can't reach the FTS space+time path).
#####
##### LOCAL SHIM mirroring Oceananigans PR #5726 (CliMA/Oceananigans.jl#5726) — same `MappedData` /
##### `mapped_data` / func-space (caller-applies-inverse) design. Once #5726 lands and releases, delete
##### this file; the `interpolate(func, …)` calls match upstream verbatim. The FTS method here
##### reimplements the time path only because pre-#5726 Oceananigans' time-interpolation requires
##### `data::OffsetArray` (#5726 relaxes it to `AbstractArray` and reuses it).

using Adapt: Adapt, adapt
using Oceananigans.Fields: FractionalIndices, flatten_node, interpolator, _interpolate
using Oceananigans.Grids: topology, Flat
using Oceananigans.OutputReaders: FlavorOfFTS, TimeInterpolator, memory_index

"""
    MappedData(func, data)

Lazily apply `func` elementwise to `data`: a read returns `func(data[I...])`, so the existing
`_interpolate` blend operates on `func`-mapped values without copying. GPU-safe when `func` is
(`log`/`exp` are).
"""
struct MappedData{T, N, F, A} <: AbstractArray{T, N}
    func :: F
    data :: A
end

@inline MappedData(func::F, data::A) where {F, A} = MappedData{eltype(A), ndims(A), F, A}(func, data)
@inline Base.size(m::MappedData) = size(m.data)
@inline Base.axes(m::MappedData) = axes(m.data)
@inline Base.getindex(m::MappedData, I::Vararg{Int}) = m.func(@inbounds m.data[I...])
Adapt.adapt_structure(to, m::MappedData) = MappedData(m.func, adapt(to, m.data))

# `func = identity` returns `data` unchanged ⇒ the identity path is byte-identical to plain
# `interpolate` and pays no wrapper overhead.
@inline mapped_data(func, data) = MappedData(func, data)
@inline mapped_data(::typeof(identity), data) = data

# Field → node, with each source value mapped through `func` (func-space result, no inverse).
@inline function Oceananigans.Fields.interpolate(func::Base.Callable, at_node, from_field, from_loc, from_grid)
    fidx = FractionalIndices(at_node, from_grid, from_loc...)
    ix = interpolator(fidx.i)
    iy = interpolator(fidx.j)
    iz = interpolator(fidx.k)
    return _interpolate(mapped_data(func, from_field), ix, iy, iz)
end

# FieldTimeSeries → node, space + time, with each source value mapped through `func` (func-space,
# no inverse). Reimplements the time path because pre-#5726 the Oceananigans time-interpolation
# methods require `data::OffsetArray`; #5726 relaxes them to `AbstractArray` and reuses them.
@inline function Oceananigans.Fields.interpolate(func::Base.Callable, to_node, to_time_index::Time,
                                                  from_fts::FlavorOfFTS, from_loc, from_grid)
    data   = mapped_data(func, from_fts.data)
    interp = TimeInterpolator(from_fts.time_indexing, from_fts.times, to_time_index.time)
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
    m₁ = memory_index(from_fts.backend, from_fts.time_indexing, Nt, n₁)
    m₂ = memory_index(from_fts.backend, from_fts.time_indexing, Nt, n₂)

    ψ₁ = _interpolate(data, ix, iy, iz, m₁)
    ψ₂ = _interpolate(data, ix, iy, iz, m₂)
    return ifelse(n₁ == n₂, ψ₁, ψ₂ * ñ + ψ₁ * (1 - ñ))
end
