module Bathymetry

export regrid_bathymetry, ORCAGrid

import Oceananigans
import Downloads

using ImageMorphology: ImageMorphology
using JLD2: JLD2, jldopen
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture, on_architecture, CPU
using Oceananigans.BoundaryConditions: BoundaryConditions
using Oceananigans.DistributedComputations: DistributedComputations, DistributedGrid,
                                            reconstruct_global_grid, all_reduce
using Oceananigans.Fields: interpolate!, Field, interior
using Oceananigans.Grids: x_domain, y_domain,
                          topology, Bounded, Flat, Periodic, Center, Face,Periodic,
                          RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.Utils: launch!
using OffsetArrays: OffsetArrays, OffsetArray
using NCDatasets: NCDatasets, Dataset
using Printf: Printf
using Scratch: Scratch, @get_scratch!

using ..DataWrangling: Metadatum, native_grid, metadata_path, download_dataset,
                       dataset_variable_name, validate_dataset_coverage
using ..DataWrangling.ETOPO: ETOPO2022

include("regrid_bathymetry.jl")
include("orca_grid.jl")

end # module
