module NestedSimulations

export NestedModel, NestedSimulation,
       parent_boundary_conditions, parent_forcings

using Oceananigans
using Oceananigans.Fields: Face
using Oceananigans.Forcings: Relaxation
using Oceananigans.Grids: AbstractGrid
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Simulations: Simulation
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Units: Time

include("nested_model.jl")
include("nested_simulation.jl")
include("transformed_interpolate.jl")
include("interpolated_fts_boundary.jl")
include("parent_state_boundary.jl")
include("parent_boundary_conditions.jl")
include("parent_forcings.jl")

end # module
