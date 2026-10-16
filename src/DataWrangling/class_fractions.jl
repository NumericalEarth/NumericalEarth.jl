#####
##### Per-class area fractions of a categorical (class-code) map
#####

using Oceananigans.Grids: constructor_arguments

"""
    $(TYPEDSIGNATURES)

The area fraction of each of `codes` on `grid`, regridded conservatively from the class map
`classes`. A cell with no class counts toward none of them.
"""
function class_fractions(grid, classes::Field, codes)
    source_grid = classes.grid

    # TODO: drop the relabel once Oceananigans' `regrid!` matches longitudes modulo 360.
    if source_grid isa LatitudeLongitudeGrid
        longitude_shift = 360 * round((mean(x_domain(grid)) - mean(x_domain(source_grid))) / 360)
        args, kwargs = constructor_arguments(source_grid)
        kwargs[:longitude] = kwargs[:longitude] .+ longitude_shift
        source_grid = LatitudeLongitudeGrid(args[:architecture], args[:number_type]; kwargs...)
    end

    indicator = Field{Center, Center, Nothing}(source_grid)
    return map(codes) do code
        interior(indicator) .= interior(classes) .== code
        fraction = Field{Center, Center, Nothing}(grid)
        regrid!(fraction, indicator)
        fill_halo_regions!(fraction)
        fraction
    end
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
