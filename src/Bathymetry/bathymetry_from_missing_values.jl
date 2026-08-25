#####
##### Bathymetry derived from where a three-dimensional ocean dataset has missing values
#####
#
# A reanalysis like GLORYS marks the seafloor implicitly: cells below it hold the product's missing
# value. That mask IS the bathymetry, and it is the only record of it — `Field`/`FieldTimeSeries`
# inpaint by default, filling those cells with propagated ocean values, so the seafloor cannot be
# recovered downstream. Deriving it here, from the raw field, is what lets a nested child's boundary
# bathymetry be made consistent with the parent state that drives it.

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

"""
$(TYPEDSIGNATURES)

Regrid onto `target_grid` the bathymetry implied by `metadatum`'s missing values — the seafloor a
three-dimensional ocean dataset carries implicitly through its land mask. The field is read without
inpainting, so the mask survives to be read.
"""
function bathymetry_from_missing_values(target_grid, metadatum::Metadatum)
    arch = architecture(target_grid)
    dataset_field = Field(metadatum, arch; inpainting = nothing)
    dataset_bottom_height = bottom_height_from_mask(DataWrangling.compute_mask(metadatum, dataset_field))

    bottom_height = Field{Center, Center, Nothing}(target_grid)
    interpolate!(bottom_height, dataset_bottom_height)

    return bottom_height
end
