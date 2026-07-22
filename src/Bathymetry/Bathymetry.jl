module Bathymetry

export regrid_bathymetry, regrid_topography, bare_earth_elevation, ORCAGrid

using Downloads: Downloads, download
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
                          Flat, Periodic, Bounded, AbstractGrid,
                          RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.Utils: launch!, worksize
using OffsetArrays: OffsetArrays, OffsetArray
using NCDatasets: NCDatasets, Dataset
using Printf: Printf

using ..DataWrangling: DataWrangling, Metadatum, BoundingBox, native_grid, metadata_path,
                       dataset_variable_name, validate_dataset_coverage,
                       default_horizontal_padding
using ..DataWrangling.ETOPO: ETOPO2022
using ..DataWrangling.CopernicusDEM: GLO30

include("regrid_bathymetry.jl")
include("orca_grid.jl")

end # module
