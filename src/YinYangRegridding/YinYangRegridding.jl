"""
    YinYangRegridding

Regridding between an Oceananigans `YinYangGrid` and a global `LatitudeLongitudeGrid`, for surface
exchange through a latitude-longitude exchange grid. Two functions, kept apart on purpose:

- `interpolate_to_yin_yang!`: state and boundary data, latitude-longitude → Yin-Yang, bilinear, panel by
  panel. It conserves nothing.
- `transfer_flux!`: fluxes, in both directions, by ownership-aware area-weighted averages that preserve
  constants. The averages are row-normalized and only approximately conservative; an optional correction
  closes the global budget integrated with the ownership weights, and only that: it gives neither local
  flux conservation nor conservation of the flux through the seam between the panels.

Both take scalars at cell centers and horizontal vectors, which are regridded as Cartesian 3-vectors.
Every operator is a sparse table built once on the CPU and applied by a kernel.

TODO: these operators belong upstream. The interpolation of a geographic field onto the panels and the
Cartesian vector remap belong in `Oceananigans.MultiRegion` next to `yin_yang_remapping`, as
`remap!(yin_yang_field, geographic_field)` and `remap!((u, v), (uᵍ, vᵍ))`. The ownership-weighted
area-weighted operators belong in the Oceananigans extension of ConservativeRegridding, as
`Regridder(yin_yang_grid, grid)`, which already builds the per-panel cell intersections used here.
"""
module YinYangRegridding

export YinYangRegridder, interpolate_to_yin_yang!, transfer_flux!,
       GlobalRescaling, UniformRedistribution,
       surface_integral, surface_force_and_torque

using Oceananigans: Oceananigans, instantiated_location
using Oceananigans.Architectures: CPU, architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: Center, Face, Periodic, LatitudeLongitudeGrid, topology, λnode, φnode, λnodes, φnodes, halo_size
using Oceananigans.MultiRegion: YinYangGrid, owned_weights
using Oceananigans.Operators: Azᶜᶜᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.Utils: getregion, launch!

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using StaticArrays: SVector, SMatrix

using ..NumericalEarth: SIGNATURES

include("regridding_tables.jl")
include("region_geometry.jl")
include("yin_yang_regridder.jl")
include("flux_corrections.jl")
include("scalar_regridding.jl")
include("vector_regridding.jl")

end # module YinYangRegridding
