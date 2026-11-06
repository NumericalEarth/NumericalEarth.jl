using Oceananigans: Oceananigans, location
using Oceananigans.BoundaryConditions: BoundaryConditions
using Oceananigans.MultiRegion: MultiRegionField
using Oceananigans.Utils: Utils

"""
    @regional T

Define `getregion` for the struct `T`, field by field, so that a `T` holding multi-region fields
reaches a regional kernel launch (`@apply_regionally launch!(...)`) as its regional part.
"""
macro regional(T)
    fields = fieldnames(getfield(__module__, T))
    outer_args = [:(Utils._getregion(getfield(x, $(QuoteNode(f))), r)) for f in fields]
    inner_args = [:(Utils.getregion(getfield(x, $(QuoteNode(f))), r)) for f in fields]
    T = esc(T)
    return quote
        @inline $Utils.getregion(x::$T, r) = $T($(outer_args...))
        @inline $Utils._getregion(x::$T, r) = $T($(inner_args...))
    end
end

# TODO: move to Oceananigans, whose `regularize_boundary_condition(::Field, ...)` rejects a multi-region
# `Field` because its `indices` are a `MultiRegionObject`.
function BoundaryConditions.regularize_boundary_condition(condition::MultiRegionField, grid, loc, dim, side, args...)
    location(condition)[dim] === Nothing ||
        throw(ArgumentError("A multi-region Field used as a boundary condition on dimension $dim must be reduced along it."))
    return condition
end
