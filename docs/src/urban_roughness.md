# Urban roughness

`urban_roughness(grid, dataset; surface_layer_height, neighborhood=1000)` computes
momentum roughness length and zero-plane displacement from building data on a
`LatitudeLongitudeGrid`. It evaluates the closure on neighborhoods of the specified
width in meters, then preserves their area-mean neutral drag when reducing to the
model grid. `surface_layer_height` is the height in meters at which that drag is matched;
it must exceed displacement plus roughness length in every urban neighborhood.

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

The result also contains `zero_plane_displacement`, `urban_fraction` (the fraction of
each model cell covered by urban neighborhoods), and `building_height` (their mean
height weighted by built area). Cells without urban neighborhoods receive the closure's
bare-soil roughness and zero displacement.

```@example urban_roughness
roughness.zero_plane_displacement
```

`GlobalBuildingFootprints3D()` can be passed in place of `GHSBuilt()`. It supplies
measured height spread, maximum height, and frontal area from rasterized footprints.
The GHSL closure estimates these quantities from mean height and built fraction.

For morphometry alone, use `building_morphometry(grid, dataset)`;
omitting `dataset` defaults to `GlobalBuildingFootprints3D()`.
To evaluate fields that are already prepared, `urban_roughness(h, λᵖ)`
takes a height `Field` and returns the pair `(roughness_length, displacement)` on its grid.
