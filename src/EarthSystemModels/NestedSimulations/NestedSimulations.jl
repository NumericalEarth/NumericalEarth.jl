module NestedSimulations

export NestedSimulation,
       parent_boundary_conditions,
       parent_clock,
       parent_field,
       parent_time_step!,
       parent_update_state!,
       parent_interpolate

using Oceananigans
using Oceananigans: AbstractModel
using Oceananigans.Fields: AbstractField, Center, Face, instantiated_location
using Oceananigans.Grids: AbstractGrid, xnode, ynode, znode
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Simulations: Simulation, Callback, IterationInterval
using Oceananigans.TimeSteppers: tick!, Clock
using Oceananigans.Units: Time

import Oceananigans.TimeSteppers: time_step!, update_state!
import Oceananigans.Simulations: run!
import Oceananigans.Models: fields

include("parent_interface.jl")
include("nested_simulation.jl")
include("parent_boundary_conditions.jl")

end # module
