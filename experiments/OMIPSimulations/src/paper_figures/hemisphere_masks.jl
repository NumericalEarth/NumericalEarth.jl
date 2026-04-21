using Oceananigans.Grids: ynode

# Hemisphere predicates usable as `condition =` in Oceananigans reductions.
# Each has signature `(i, j, k, grid)` expected by `Average` / `Integral`,
# and works on any grid that defines `ynode(..., Center()^3)` — including
# `LatitudeLongitudeGrid`, `RectilinearGrid`, and `OrthogonalSphericalShellGrid`.

@inline northern_hemisphere(i, j, k, grid, args...) =
    ynode(i, j, k, grid, Center(), Center(), Center()) > 0

@inline southern_hemisphere(i, j, k, grid, args...) =
    ynode(i, j, k, grid, Center(), Center(), Center()) < 0

@inline global_ocean(i, j, k, grid, args...) = true
