#####
##### Per-class area fractions of a categorical (class-code) map
#####

using Oceananigans.Grids: constructor_arguments, halo_size, pop_flat_elements

# `source_grid` with its longitudes shifted by the multiple of 360 that lands it on `grid`'s.
# TODO: drop the relabel once Oceananigans' `regrid!` matches longitudes modulo 360.
function relabeled_source_grid(source_grid, grid)
    source_grid isa LatitudeLongitudeGrid || return source_grid
    longitude_shift = 360 * round((mean(x_domain(grid)) - mean(x_domain(source_grid))) / 360)
    args, kwargs = constructor_arguments(source_grid)
    kwargs[:longitude] = kwargs[:longitude] .+ longitude_shift
    return LatitudeLongitudeGrid(args[:architecture], args[:number_type]; kwargs...)
end

# The columns of `grid` as a grid with a `Flat` vertical axis, the only target `regrid!` takes
# from a horizontal source.
# TODO: drop it once Oceananigans' `regrid!` lands a `Flat`-z source on a three-dimensional grid.
function horizontal_grid(grid)
    args, kwargs = constructor_arguments(grid)
    TX, TY, _ = topology(grid)
    kwargs[:topology] = (TX, TY, Flat)
    kwargs[:size] = pop_flat_elements(size(grid), kwargs[:topology])
    kwargs[:halo] = pop_flat_elements(halo_size(grid), kwargs[:topology])
    delete!(kwargs, :z)
    return LatitudeLongitudeGrid(args[:architecture], args[:number_type]; kwargs...)
end

"""
    $(TYPEDSIGNATURES)

The area fraction of each of `codes` on `grid`, regridded conservatively from the class map
`classes`. A cell with no class counts toward none of them.
"""
function class_fractions(grid, classes::Field, codes)
    source_grid = relabeled_source_grid(classes.grid, grid)
    indicator = Field{Center, Center, Nothing}(source_grid)
    horizontal_fraction = Field{Center, Center, Nothing}(horizontal_grid(grid))
    return map(codes) do code
        interior(indicator) .= interior(classes) .== code
        regrid!(horizontal_fraction, indicator)
        fraction = Field{Center, Center, Nothing}(grid)
        interior(fraction) .= interior(horizontal_fraction)
        fill_halo_regions!(fraction)
        fraction
    end
end

#####
##### Area means over the valid cells of a field with gaps
#####

"""
    $(TYPEDSIGNATURES)

Set each cell of `target` to the area-weighted mean of the finite values of `source` under it:
`source` is regridded conservatively and normalized by the area its finite cells cover rather
than by the target cell's area. A cell with no finite value under it is `NaN`.
"""
function valid_area_mean!(target, source)
    source_grid = relabeled_source_grid(source.grid, target.grid)
    valid = Field{Center, Center, Nothing}(source_grid)
    valid_values = Field{Center, Center, Nothing}(source_grid)

    target_columns = horizontal_grid(target.grid)
    valid_area_fraction = Field{Center, Center, Nothing}(target_columns)
    regridded_values = Field{Center, Center, Nothing}(target_columns)

    interior(valid) .= isfinite.(interior(source))
    interior(valid_values) .= ifelse.(isfinite.(interior(source)), interior(source), zero(eltype(source)))
    regrid!(valid_area_fraction, valid)
    regrid!(regridded_values, valid_values)

    indices = (target.indices[1], target.indices[2], :)
    # A cell with no valid area divides zero by zero.
    interior(target) .= interior(regridded_values, indices...) ./ interior(valid_area_fraction, indices...)
    fill_halo_regions!(target)
    return target
end

#####
##### The majority class of regridded fractions
#####

"""
    majority_class!(majority_class, fractions, codes)

Set each cell of `majority_class` to the entry of `codes` whose `Field` in `fractions`
covers the largest fraction of it, ties going to the earlier entry. Cells whose fractions
sum below one half are `NaN`.
"""
function majority_class!(majority_class, fractions, codes)
    grid = majority_class.grid
    arch = child_architecture(architecture(grid))
    LX, LY, LZ = location(majority_class)
    largest_fraction = Field{LX, LY, LZ}(grid)
    total_fraction = Field{LX, LY, LZ}(grid)
    parameters = KernelParameters(interior_indices(majority_class))

    fill!(majority_class, 0)
    for (fraction, code) in zip(fractions, codes)
        launch!(arch, grid, parameters, _accumulate_majority_class!,
                majority_class, largest_fraction, total_fraction, fraction, code)
    end

    launch!(arch, grid, parameters, _mask_uncovered_class!, majority_class, total_fraction)
    fill_halo_regions!(majority_class)
    return majority_class
end

@kernel function _accumulate_majority_class!(target, largest_fraction, total_fraction, fraction, code)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        f = fraction[i, j, k]
        larger = f > largest_fraction[i, j, k]
        largest_fraction[i, j, k] = ifelse(larger, f, largest_fraction[i, j, k])
        target[i, j, k] = ifelse(larger, convert(eltype(target), code), target[i, j, k])
        total_fraction[i, j, k] += f
    end
end

# Cells outside the product's coverage have no valid pixels, so every fraction is
# zero and no class holds a majority.
@kernel function _mask_uncovered_class!(target, total_fraction)
    i, j, k = @index(Global, NTuple)
    FT = eltype(target)
    @inbounds covered = total_fraction[i, j, k] > convert(FT, 1//2)
    @inbounds target[i, j, k] = ifelse(covered, target[i, j, k], convert(FT, NaN))
end
