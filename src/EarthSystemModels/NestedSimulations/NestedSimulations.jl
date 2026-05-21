module NestedSimulations

export NestedSimulation, parent_boundary_conditions, parent_forcings,
       child_simulation, default_parent_variables

using Oceananigans
using Oceananigans.Fields: Field, Center, Face, instantiated_location
using Oceananigans.Forcings: Relaxation
using Oceananigans.Grids: AbstractGrid, xnode, ynode, znode
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Models: NonhydrostaticModel
using Oceananigans.Simulations: Simulation, Callback, IterationInterval
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Units: Time

import Oceananigans.Simulations: run!
import Oceananigans.TimeSteppers: time_step!

include("nested_simulation.jl")
include("interpolated_fts_boundary.jl")
include("parent_boundary_conditions.jl")
include("parent_forcings.jl")
include("child_simulation.jl")

end # module
