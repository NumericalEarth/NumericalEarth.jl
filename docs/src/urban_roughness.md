# Urban roughness

`urban_roughness(grid, dataset; surface_layer_height, neighborhood=1000)` computes
momentum roughness length and zero-plane displacement from building data on a
`LatitudeLongitudeGrid`. It evaluates the closure on neighborhoods of the specified
width in meters, then preserves their area-mean neutral drag when reducing to the
model grid. `surface_layer_height` is the height in meters at which that drag is matched;
it must exceed displacement plus roughness length in every urban neighborhood. Coupled
fluxes evaluate the similarity profile at the atmosphere's surface-layer height, so the drag
is exact there when `surface_layer_height` equals it.

## GHSL building data

[`GHSBuilt`](@ref) pairs built-up surface with building height on the native Mollweide
pixels before aggregation. Its default uses the 10 m surface product and 100 m height,
both for 2018. This example uses the smaller 100 m surface product for 2020, paired with
the 2018 height. Loading `ArchGDAL` enables raster reading and reprojection; source tiles
are downloaded on first use and cached.

```@example urban_roughness
using NumericalEarth
using Oceananigans
using ArchGDAL

grid = LatitudeLongitudeGrid(size = (4, 4),
                             longitude = (-0.10, -0.06), latitude = (51.50, 51.54),
                             topology = (Bounded, Bounded, Flat))
dataset = GHSBuilt(surface=GHSBuiltS())
roughness = urban_roughness(grid, dataset; surface_layer_height=500, neighborhood=1000)
roughness.momentum_roughness_length
```

`momentum_roughness_length` and `zero_plane_displacement` match the neutral drag of the
whole model cell: its urban neighborhoods, which cover `urban_fraction` of the cell, and an
`unbuilt` surface over the rest, by default the closure's bare-soil roughness and zero
displacement. Pass `unbuilt = (ℓᵐ, d)` for another surface, such as `(1e-4, 0)` for water.
`urban_momentum_roughness_length` and `urban_zero_plane_displacement` describe the urban
neighborhoods alone and are `NaN` where a cell has none. `building_height` is their mean
height weighted by built area, also `NaN` where a cell has no built area.

```@example urban_roughness
roughness.zero_plane_displacement
```

Missing GHSL heights are averaged over the measured built area in each neighborhood;
the built fraction includes footprints without heights. A built neighborhood with no
measured height still raises an error.

`GlobalBuildingFootprints3D()` can be passed in place of `GHSBuilt()`. It supplies
measured height spread, maximum height, and frontal area from rasterized footprints.
The GHSL closure estimates these quantities from mean height and built fraction.

For morphometry alone, use `building_morphometry(grid, dataset)`;
omitting `dataset` defaults to `GlobalBuildingFootprints3D()`.
To evaluate fields that are already prepared, `urban_roughness(h, λᵖ)`
takes a height `Field` and returns the pair `(roughness_length, displacement)` on its grid.
