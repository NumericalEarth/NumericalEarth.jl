module NestedModels

export NestedModel, NestedSimulation, nested_atmosphere_model,
       parent_boundary_conditions, parent_forcings

# Model-specific extension point (methods defined in the Breeze extension):
#   `nested_atmosphere_model(parent, child_grid; …)` builds a child atmosphere over `child_grid` driven
#       by `parent` (lateral BCs + Davies relaxation derived on the fly), wrapped in a `NestedModel`.
# The child's lateral BCs / interior relaxation are built from the exchanger's parent-derived prognostic
# `FieldTimeSeries` via the generic `parent_boundary_conditions` / `parent_forcings`.
function nested_atmosphere_model end

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
include("interpolated_fts_boundary.jl")
include("parent_state_boundary.jl")
include("parent_boundary_conditions.jl")
include("parent_forcings.jl")

end # module NestedModels
