module NumericalEarthConservativeRegriddingExt

using ConservativeRegridding: Regridder
using NumericalEarth: NumericalEarth
using Oceananigans.Grids: AbstractGrid

# TODO: move the ownership-weighted Yin-Yang operators built from these intersections into the
# Oceananigans extension of ConservativeRegridding, as `Regridder(yin_yang_grid, grid)`.
function NumericalEarth.YinYangRegridding.cell_intersection_areas(destination_grid::AbstractGrid, source_grid::AbstractGrid)
    intersections = Regridder(destination_grid, source_grid).intersections
    entries = findall(!iszero, intersections)
    return map(first ∘ Tuple, entries), map(last ∘ Tuple, entries), intersections[entries]
end

end # module NumericalEarthConservativeRegriddingExt
