# Nested regional ocean

```@meta
DocTestSetup = quote
    using NumericalEarth
    using Oceananigans
end
```

A nested ocean is a regional hydrostatic ocean (the _child_) driven one-way at its open sides by a coarser
_parent_: an ocean reanalysis such as `GLORYSDaily()`, or any [`PrescribedOcean`](@ref). The parent is
never changed by the child.

[`nested_ocean_simulation`](@ref) builds the whole nest from a grid and a dataset, and
[`nested_ocean_model`](@ref) builds the model alone. The result is a `Simulation` of a [`NestedModel`](@ref),
which goes into an [`OceanOnlyModel`](@ref) or an [`OceanSeaIceModel`](@ref) like any other ocean simulation.
The [Bouvet Island example](@ref bouvet-nested-ocean) shows a complete setup.

A parent can also be built in memory. Here a uniform eastward flow surrounds a smaller child domain,
which is nested in it and stepped three times:

```jldoctest nested
using Oceananigans.Units
using Logging

parent_grid = LatitudeLongitudeGrid(size = (12, 12, 6), longitude = (-2, 2), latitude = (28, 32),
                                    z = (-1000, 0), topology = (Bounded, Bounded, Bounded))

parent_ocean = PrescribedOcean(parent_grid, [0.0, 10days])
set!(parent_ocean; T = 15, S = 35, u = 0.1, v = 0, η = 0)

child_grid = LatitudeLongitudeGrid(size = (20, 20, 6), halo = (7, 7, 7), longitude = (-1, 1), latitude = (29, 31),
                                   z = (-1000, 0), topology = (Bounded, Bounded, Bounded))

nested = nested_ocean_model(parent_ocean, child_grid)

simulation = Simulation(nested; Δt = 1minute, stop_iteration = 3)
with_logger(NullLogger()) do
    run!(simulation)
end
nested

# output
NestedModel(PrescribedOcean{Float64, CPU, LatitudeLongitudeGrid}(Nt = 2) → HydrostaticFreeSurfaceModel{CPU, LatitudeLongitudeGrid}(time = 3 minutes, iteration = 3))
├── parent: PrescribedOcean{Float64, CPU, LatitudeLongitudeGrid}(Nt = 2)
└── child:  HydrostaticFreeSurfaceModel{CPU, LatitudeLongitudeGrid}(time = 3 minutes, iteration = 3)
```

## What a time step does

Each time step of a `NestedModel`

1. steps the parent to the end of the child's step,
2. samples the parent onto narrow bands along the open sides of the child grid,
3. steps the child, whose boundary conditions and relaxation read from those bands.

When the parent is a dataset, only strips along the open sides are read from disk, and only a few
time samples are held in memory at once. The child starts from the parent interpolated onto the whole
child grid at the first date. On a curvilinear grid the parent's eastward and northward velocities are
rotated onto the grid's own axes.

## Open boundaries

The open sides default to the walls of the grid that touch water and may be set with `open_sides`.
On each open side:

| Field | Scheme | Default |
|:------|:-------|:--------|
| boundary-normal and tangential velocity | oblique radiation | nudged toward the parent over 9 hours on inflow, 360 days on outflow |
| tracers | tracer reservoir | relaxed toward the parent over 20 km on inflow |
| barotropic transport | Flather | |
| free surface | Chapman | |

`normal_velocity_scheme`, `tangential_velocity_scheme` and `tracer_scheme` may each be a `NamedTuple`
keyed by side, to set a scheme per side.

## Relaxation band

Over the outermost `relaxation_width` cells of each open side (default: 2/3°), the temperature, salinity
and the baroclinic part of the velocity relax toward the parent at `relaxation_rate` (default: 1/6 hours).
The barotropic flow is left to Flather. `relaxation_rate = nothing` turns the band off, and
`relaxation_rate = (inflow = 1/6hours, outflow = 0)` relaxes the tracers only where the flow enters.

## Bathymetry at the boundaries

A parent's boundary transport only fits the water column it was computed over. When the parent's
seafloor is known (by default, the dataset's own bathymetry, or the seafloor its missing temperature
values mark), the child's bottom is blended toward it within `bathymetry_blend_width` cells of each open
side (default: 1/3°), no shallower than `minimum_depth`. The Flather exterior transport is then the
parent's depth-mean velocity times the child's wet depth, and the boundary-normal velocity is corrected
to integrate to it.

## Inverse barometer

Ocean reanalyses such as GLORYS have no inverse-barometer response in their sea level, while a child
forced with atmospheric pressure does. The Flather exterior sea level therefore adds the child's own
inverse-barometer response to the parent's, so the pressure-driven sea level isn't read as a jump at the
boundary. Whether a dataset's sea level already has the response comes from
`sea_level_includes_inverse_barometer(dataset)`, which is `false` by default. For a parent that does
have it, such as the output of an outer nest forced with pressure, pass
`parent_includes_inverse_barometer = true`. Without atmospheric pressure the added sea level is zero.

## Sea ice

[`nested_sea_ice_simulation`](@ref) builds a sea ice simulation over a nested ocean, with the parent's
sea ice read from the same dataset ([`parent_sea_ice`](@ref)). At the open sides the ice thickness and
concentration relax toward the parent, inflowing ice carries no snow, and the boundary-normal ice
velocity lets ice leave but is not pulled inward ([`nested_sea_ice_boundary_conditions`](@ref)).
[`set_parent_sea_ice!`](@ref) sets the sea ice state from the parent.

## Constructors

```@docs
nested_ocean_simulation
nested_ocean_model
```
