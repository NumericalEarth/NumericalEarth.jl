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

Evaluate a land property at `(i, j, k)`. Defined for `Number` and
`AbstractField`; unsupported types raise a `MethodError` at compile
time. We avoid a catch-all throwing fallback here because GPUCompiler
cannot lower the `throw(ArgumentError("…\$(typeof(p))…"))` codepath
(it needs runtime `Symbol` construction), which trips GPU kernel
compilation even when the fallback isn't reached.
"""
@inline property_value(p::Number, i, j, k=1) = p
@inline property_value(p::AbstractField, i, j, k=1) = @inbounds p[i, j, k]
