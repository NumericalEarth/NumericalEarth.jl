module Bathymetry

export regrid_bathymetry, regrid_topography, smooth_topography!, bare_earth_elevation, ORCAGrid

using DocStringExtensions: TYPEDSIGNATURES
using Downloads: Downloads, download
using ImageMorphology: ImageMorphology
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture, CPU, on_architecture
using Oceananigans.BoundaryConditions: BoundaryConditions
using Oceananigans.DistributedComputations: DistributedComputations, DistributedGrid,
                                            reconstruct_global_grid, all_reduce
using Oceananigans.Fields: Field, interior, interpolate!, set!
using Oceananigans.Grids: x_domain, y_domain, topology, Face, Center,
                          Flat, Periodic, Bounded, LeftConnected, RightConnected, AbstractGrid,
                          RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.Utils: launch!
using OffsetArrays: OffsetArrays, OffsetArray
using NCDatasets: NCDatasets, Dataset
using Printf: Printf

using ..DataWrangling: DataWrangling, Metadatum, native_grid,
                       dataset_variable_name, validate_dataset_coverage,
                       validate_region_covers_grid, default_region,
                       read_windowed_variable, set_region_data!,
                       no_data_means_sea_level,
                       FieldRegridding, load_field_cache, save_field_cache
using ..DataWrangling.ETOPO: ETOPO2022
using ..DataWrangling.CopernicusDEM: GLO30

include("regrid_bathymetry.jl")
include("smooth_topography.jl")
include("orca_grid.jl")

end # module
