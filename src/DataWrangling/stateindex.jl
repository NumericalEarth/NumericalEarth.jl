# TODO: `stateindex` is a general utility for reading a number, array, `FieldTimeSeries`, function, or
# tuple of them at (i, j, k, time), and belongs in Oceananigans. It lives here, at the bottom of
# NumericalEarth's module stack, so that DataWrangling depends on nothing else in NumericalEarth.

const SomeKindOfFieldTimeSeries = Union{FieldTimeSeries,
                                        GPUAdaptedFieldTimeSeries}

const SKOFTS = SomeKindOfFieldTimeSeries

@inline stateindex(a::Union{Number, Nothing}, i, j, k, args...) = a
@inline stateindex(a::AbstractArray, i, j, k, args...) = @inbounds a[i, j, k]
@inline stateindex(a::SKOFTS, i, j, k, grid, time, args...) = @inbounds a[i, j, k, time]
@inline stateindex(a::SKOFTS, i, j, k, time_interpolator::TimeInterpolator) = @inbounds a[i, j, k, time_interpolator]

@inline function stateindex(a::Function, i, j, k, grid, time, (LX, LY, LZ), args...)
    # `_node` always returns the full (λ, φ, z) triple — with placeholder
    # values for Flat dimensions — whereas `node` drops Flat-dim entries
    # and produces a shorter tuple that breaks the destructuring below.
    λ, φ, z = _node(i, j, k, grid, LX(), LY(), LZ())
    return a(λ, φ, z, time)
end

@inline function stateindex(a::Union{Tuple, NamedTuple}, i, j, k, grid, time, args...)
    return map(value -> stateindex(value, i, j, k, grid, time, args...), a)
end
