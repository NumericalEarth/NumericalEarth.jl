module Bathymetry

export regrid_bathymetry, regrid_topography, regrid_ocean_fraction, smooth_topography!, ORCAGrid

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
                          RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid,
                          λnodes, φnodes
using Oceananigans.Utils: launch!
using OffsetArrays: OffsetArrays, OffsetArray
using NCDatasets: NCDatasets, Dataset
using Printf: Printf

using ..DataWrangling: Metadatum, native_grid, metadata_path,
                       dataset_variable_name, validate_dataset_coverage,
                       FieldRegridding, load_field_cache, save_field_cache
using ..DataWrangling.ETOPO: ETOPO2022

include("regrid_bathymetry.jl")
include("smooth_topography.jl")
include("orca_grid.jl")

end # module
