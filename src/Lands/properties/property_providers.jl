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
"""
@inline normalize_property(::Type{FT}, p::Number) where FT = convert(FT, p)
@inline normalize_property(::Type, p::AbstractField) = p
@inline normalize_property(::Type, p) = p

"""
    property_value(prop, i, j[, k=1])

Evaluate a land property at `(i, j, k)`.
"""
@inline property_value(p::Number, i, j, k=1) = p
@inline property_value(p::AbstractField, i, j, k=1) = @inbounds p[i, j, k]

