#####
##### The seafloor a three-dimensional ocean dataset marks with missing values
#####
#
# A reanalysis like GLORYS marks the seafloor implicitly: cells below it hold the product's missing
# value. `Field`/`FieldTimeSeries` inpaint by default, so `regrid_bathymetry` reads the raw field and
# recovers the seafloor from its mask here.

"""
$(TYPEDSIGNATURES)

Return the bottom height (m, negative below sea level) implied by `mask`, a three-dimensional boolean
field that is `true` where a dataset has no data. The seafloor of a column is the bottom face of its
deepest wet cell; a column that is wet nowhere sits at sea level.
"""
function bottom_height_from_mask(mask)
    grid = mask.grid
    bottom_height = Field{Center, Center, Nothing}(grid)
    launch!(architecture(grid), grid, :xy, _bottom_height_from_mask!, bottom_height, grid, mask)
    return bottom_height
end

# `z` is ordered bottom-to-top, so sweeping downward in `k` leaves the deepest wet cell's bottom face
# as the last value written.
@kernel function _bottom_height_from_mask!(bottom_height, grid, mask)
    i, j = @index(Global, NTuple)

    zᵇ = zero(grid)
    for k in size(grid, 3):-1:1
        wet = @inbounds !mask[i, j, k]
        zᵇ = ifelse(wet, znode(i, j, k, grid, Center(), Center(), Face()), zᵇ)
    end

    @inbounds bottom_height[i, j, 1] = zᵇ
end
