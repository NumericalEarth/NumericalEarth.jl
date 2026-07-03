module NestedModels

export NestedModel, NestedSimulation, nested_atmosphere_model,
       parent_boundary_conditions, parent_forcings, blend_parent_terrain!,
       exchange_state!, total_density, reconstruct_parent_state

# Model-specific extension point (methods defined in the Breeze extension):
#   `nested_atmosphere_model(parent, child_grid; …)` builds a child atmosphere over `child_grid` driven
#       by `parent` (lateral BCs + Davies relaxation derived on the fly), wrapped in a `NestedModel`.
# The child's lateral BCs / interior relaxation are built from the exchanger's parent-derived prognostic
# `FieldTimeSeries` via the generic `parent_boundary_conditions` / `parent_forcings`.
function nested_atmosphere_model end

# Total air density ρ = ρᵈ + Σρqˣ from a state exchanger's density-weighted prognostics at time index
# `n` (method in the Breeze extension). Centralizes the partial-density sum so recovering specific
# quantities (qᵛ = ρqᵛ/ρ) stays correct as condensate species are added to the prognostic set.
function total_density end

# Reconstruct the parent-derived child prognostic state at a `time` from the parent's FULL-in-memory raw
# fields — bypassing a state exchanger's 2-level resident window, whose residency aliases across arbitrary
# time queries (e.g. per-frame post-run reads). Uses the exchanger's own constants / standard pressure /
# condensate sources, so the result matches what the child integrated for any condensate/pˢᵗ config
# (method in the Breeze extension).
function reconstruct_parent_state end

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
include("parent_boundary_conditions.jl")
include("parent_forcings.jl")
include("parent_terrain.jl")

end # module NestedModels
