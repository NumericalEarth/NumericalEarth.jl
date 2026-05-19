module Bathymetry

export regrid_bathymetry, ORCAGrid

using Downloads: Downloads
using ImageMorphology: ImageMorphology
using JLD2: JLD2, jldopen
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture, CPU, on_architecture
using Oceananigans.BoundaryConditions: BoundaryConditions
using Oceananigans.DistributedComputations: DistributedComputations, DistributedGrid,
                                            reconstruct_global_grid, all_reduce
using Oceananigans.Fields: Field, interior, interpolate!
using Oceananigans.Grids: x_domain, y_domain, topology, Face, Center,
                          Flat, Periodic, Bounded,
                          RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.Utils: launch!
using OffsetArrays: OffsetArrays, OffsetArray
using NCDatasets: NCDatasets, Dataset
using Printf: Printf
using Scratch: Scratch, @get_scratch!

using ..DataWrangling: Metadatum, native_grid, metadata_path, download,
                       dataset_variable_name, validate_dataset_coverage
using ..DataWrangling.ETOPO: ETOPO2022

include("regrid_bathymetry.jl")
include("orca_grid.jl")

end # module
