#####
##### The seafloor a three-dimensional ocean dataset marks with missing values
#####

# The bottom face of each column's deepest wet cell, where `mask` is `true` without data
function bottom_height_from_mask(mask)
    grid = mask.grid
    bottom_height = Field{Center, Center, Nothing}(grid)
    launch!(architecture(grid), grid, :xy, _bottom_height_from_mask!, bottom_height, grid, mask)
    return bottom_height
end

# `z` runs bottom to top, so the downward sweep ends at the bottom face of the deepest wet cell
@kernel function _bottom_height_from_mask!(bottom_height, grid, mask)
    i, j = @index(Global, NTuple)

    zᵇ = zero(grid)
    for k in size(grid, 3):-1:1
        wet = @inbounds !mask[i, j, k]
        zᵇ = ifelse(wet, znode(i, j, k, grid, Center(), Center(), Face()), zᵇ)
    end

    @inbounds bottom_height[i, j, 1] = zᵇ
end
