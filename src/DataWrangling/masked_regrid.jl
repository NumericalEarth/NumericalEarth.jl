using DocStringExtensions: TYPEDSIGNATURES

"""
$(TYPEDSIGNATURES)

Fill `target` with the area-weighted mean of the finite `source` cells under each
target cell. A target cell that no finite source cell overlaps is set to `NaN`.
"""
function masked_regrid!(target, source)
    LX, LY, LZ = location(source)
    finite_values = Field{LX, LY, LZ}(source.grid)
    finite_mask   = Field{LX, LY, LZ}(source.grid)
    finite = isfinite.(interior(source))
    interior(finite_values) .= ifelse.(finite, interior(source), 0)
    interior(finite_mask)   .= finite

    TX, TY, TZ = location(target)
    weight = Field{TX, TY, TZ}(target.grid)
    regrid!(target, finite_values)
    regrid!(weight, finite_mask)

    # A cell with no finite coverage divides 0 by 0 and comes out NaN.
    interior(target) ./= interior(weight)

    return target
end
