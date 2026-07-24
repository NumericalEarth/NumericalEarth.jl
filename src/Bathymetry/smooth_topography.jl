using DocStringExtensions: TYPEDSIGNATURES

@kernel function _smooth_topography!(smoothed, unsmoothed)
    i, j = @index(Global, NTuple)
    @inbounds smoothed[i, j, 1] = (unsmoothed[i-1, j-1, 1] + 2 * unsmoothed[i, j-1, 1] + unsmoothed[i+1, j-1, 1]
                             + 2 * (unsmoothed[i-1, j,  1] + 2 * unsmoothed[i, j,  1] + unsmoothed[i+1, j,  1])
                                  + unsmoothed[i-1, j+1, 1] + 2 * unsmoothed[i, j+1, 1] + unsmoothed[i+1, j+1, 1]) / 16
end

"""
$(TYPEDSIGNATURES)

Smooth the two-dimensional field `elevation` in place with `passes` applications of a
two-dimensional binomial (1-2-1) filter, and return it.

Each pass multiplies the amplitude of a Fourier mode with wavenumbers `(kˣ, kʸ)` by
`(1 + cos(kˣ Δx)) (1 + cos(kʸ Δy)) / 4`: the two-grid-length mode is annihilated exactly,
a three-grid-length mode is damped to 1/4 per pass, and well-resolved scales pass through
nearly unchanged. Use it to remove the grid-scale orographic roughness left by point-sampled
regridding (see [`regrid_topography`](@ref)), which on a terrain-following coordinate otherwise
excites standing grid-scale noise in the near-surface flow.

```jldoctest
using Oceananigans
using NumericalEarth

grid = RectilinearGrid(size=(8, 8), extent=(1, 1), topology=(Periodic, Periodic, Flat))
elevation = Field{Center, Center, Nothing}(grid)
set!(elevation, (x, y) -> sinpi(8x) * sinpi(8y))
smooth_topography!(elevation, passes=1)

# output
8×8×1 Field{Center, Center, Nothing} reduced over dims = (3,) on RectilinearGrid on CPU
├── grid: 8×8×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 3×3×0 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: Nothing
└── data: 14×14×1 OffsetArray(::Array{Float64, 3}, -2:11, -2:11, 1:1) with eltype Float64 with indices -2:11×-2:11×1:1
    └── max=0.0, min=0.0, mean=0.0
```
"""
function smooth_topography!(elevation; passes = 1)
    grid = elevation.grid
    arch = architecture(grid)
    scratch = Field{Center, Center, Nothing}(grid)
    fill_halo_regions!(elevation)
    for _ in 1:passes
        parent(scratch) .= parent(elevation)
        launch!(arch, grid, :xy, _smooth_topography!, elevation, scratch)
        fill_halo_regions!(elevation)
    end
    return elevation
end
