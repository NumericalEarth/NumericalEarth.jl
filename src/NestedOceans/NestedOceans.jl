module NestedOceans

export nested_ocean_model, nested_ocean_grid, ocean_state_exchanger, OceanStateExchanger,
       parent_ocean_variables

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, instantiated_location
using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, GravityWaveRadiationBoundaryCondition,
                                       NormalFlowBoundaryCondition, NormalRadiation, ValueBoundaryCondition
using Oceananigans.Fields: interpolate, set!
using Oceananigans.Grids: AbstractGrid, Center, Face, inactive_node, node
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.OutputReaders: Linear as LinearTimeIndexing
using Oceananigans.TimeSteppers: Clock, update_state!
using Oceananigans.Units: Time, days
using Oceananigans.Utils: launch!

using ..Bathymetry: bathymetry_from_missing_values, regrid_bathymetry
using ..DataWrangling: BoundingBox, Metadata, Metadatum, MetadataSet,
                       default_download_directory, default_horizontal_padding
using ..DataWrangling.ETOPO: ETOPO2022
using ..Grids: is_three_dimensional
using ..NestedModels: NestedModels, NestedModel, blend_parent_terrain!, davies_relaxation_mask,
                      parent_boundary_conditions, parent_forcings
using ..Oceans: PrescribedOcean, default_free_surface, ocean_model, update_prescribed_ocean_series!

include("parent_ocean_variables.jl")
include("ocean_state_exchanger.jl")
include("nested_ocean_grid.jl")
include("nested_ocean_model.jl")

end # module NestedOceans
