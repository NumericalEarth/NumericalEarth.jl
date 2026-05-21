module NestedSimulations

export NestedSimulation, parent_boundary_conditions

using Oceananigans
using Oceananigans.Fields: Field, Center, Face, instantiated_location
using Oceananigans.Grids: AbstractGrid, xnode, ynode, znode
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Simulations: Simulation, Callback, IterationInterval
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Units: Time

import Oceananigans.Simulations: run!
import Oceananigans.TimeSteppers: time_step!

include("nested_simulation.jl")
include("parent_boundary_conditions.jl")

end # module
