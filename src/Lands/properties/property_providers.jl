#####
##### Land property helpers.
#####
##### Land properties are stored either as:
##### - `Number`             — uniform scalar
##### - `AbstractField`      — per-cell field (including `ConstantField`, `ZeroField`)
#####
##### Keeping this split explicit avoids wrapper types and lets kernels read
##### constants without indexing.

"""
    normalize_property(::Type{FT}, prop)

Normalize a user-provided property value to a consistent in-memory
representation: convert scalar numerics to type `FT`, keep fields as-is.
Unsupported input types raise a `MethodError` at construction time.
"""
@inline normalize_property(::Type{FT}, p::Number) where FT = convert(FT, p)
@inline normalize_property(::Type, p::AbstractField) = p

"""
    property_value(prop, i, j[, k=1])

Evaluate a land property at `(i, j, k)`. Defined for `Number`,
`AbstractArray`, `AbstractField`, and `Nothing` (a missing property,
read as `NaN` so a closure can fall back); unsupported types raise a
`MethodError` at compile time. We avoid a catch-all throwing fallback
here because GPUCompiler cannot lower the
`throw(ArgumentError("…\$(typeof(p))…"))` codepath (it needs runtime
`Symbol` construction), which trips GPU kernel compilation even when
the fallback isn't reached.

`AbstractArray` covers the GPU-adapted case: Oceananigans `Field`s are
adapted to their underlying `OffsetArray{T, 3, CuDeviceArray{T, 3}}`
on the device, which is not `<: AbstractField` even though it is the
same memory.
"""
@inline property_value(p::Number, i, j, k=1) = p
@inline property_value(p::AbstractArray, i, j, k=1) = @inbounds p[i, j, k]
@inline property_value(p::AbstractField, i, j, k=1) = @inbounds p[i, j, k]
@inline property_value(::Nothing, i, j, k=1) = NaN
