module Bathymetry

export regrid_bathymetry, ORCAGrid

using Downloads
using ImageMorphology
using JLD2
using KernelAbstractions: @kernel, @index
using Oceananigans
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedGrid, reconstruct_global_grid, all_reduce, @root
using Oceananigans.Fields: interpolate!
using Oceananigans.Grids: x_domain, y_domain, topology
using Oceananigans.Utils: launch!
using OffsetArrays
using NCDatasets
using Printf
using Scratch

using ..DataWrangling: Metadatum, native_grid, metadata_path, download_dataset
using ..DataWrangling.ETOPO: ETOPO2022

include("regrid_bathymetry.jl")
include("orca_grid.jl")

end # module
