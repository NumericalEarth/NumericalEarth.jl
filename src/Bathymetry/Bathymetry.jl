module Bathymetry

export regrid_bathymetry, regrid_topography, smooth_topography!, ORCAGrid
export Basin
export atlantic_ocean_basin, indian_ocean_basin, southern_ocean_basin, pacific_ocean_basin, arctic_ocean_basin
export label_ocean_basins
export meridional_barrier

using DocStringExtensions: TYPEDSIGNATURES
using Downloads: Downloads, download
using ImageMorphology: ImageMorphology
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture, CPU, on_architecture
using Oceananigans.BoundaryConditions: BoundaryConditions
using Oceananigans.DistributedComputations: DistributedComputations, DistributedGrid,
                                            reconstruct_global_grid, all_reduce
using Oceananigans.Fields: Field, interior, interpolate!
using Oceananigans.Grids: x_domain, y_domain, topology, Face, Center,
                          Flat, Periodic, Bounded, LeftConnected, RightConnected,
                          RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: launch!
using NCDatasets: NCDatasets, Dataset
using Printf: Printf

using ..DataWrangling: Metadatum, native_grid, metadata_path,
                       dataset_variable_name, validate_dataset_coverage,
                       FieldRegridding, load_field_cache, save_field_cache
using ..DataWrangling.ETOPO: ETOPO2022

include("label_ocean_basins.jl")
include("regrid_bathymetry.jl")
include("ocean_basin.jl")
include("smooth_topography.jl")
include("orca_grid.jl")

end # module
