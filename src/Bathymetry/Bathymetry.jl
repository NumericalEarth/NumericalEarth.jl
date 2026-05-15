module Bathymetry

export regrid_bathymetry, ORCAGrid,
       BathymetryPatchRecord, BathymetryPatchLog, BathymetryPatchSummary,
       compute_bathymetry_roughness, flag_unstable_columns, dilate_mask,
       smooth_flagged_bathymetry, summarize_bathymetry_patch,
       condition_bathymetry

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
using Oceananigans.Utils: launch!, KernelParameters
using OffsetArrays
using NCDatasets
using Printf
using Scratch

using ..DataWrangling: Metadatum, native_grid, metadata_path, download_dataset,
                      dataset_variable_name, validate_dataset_coverage
using ..DataWrangling.ETOPO: ETOPO2022

include("regrid_bathymetry.jl")
include("condition_bathymetry.jl")
include("orca_grid.jl")

end # module
