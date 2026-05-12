#####
##### `AbstractSurfaceProperties` — slab-land radiative/aerodynamic
##### property closure interface.
#####
##### Provides the radiative and aerodynamic properties the atmosphere
##### needs to close shortwave / longwave / momentum / scalar fluxes:
#####
#####   - `albedo(surface, state)`
#####   - `emissivity(surface, state)`
#####   - `momentum_roughness_length(surface, state)`
#####   - `scalar_roughness_length(surface, state)`
#####
##### Pulling these into a typed sub-component (rather than holding them
##### as bare Fields on `SlabLand`) lets behaviors like "snow modifies
##### albedo" be implemented as decorators or as concrete closures
##### without bolting conditionals into every reader.
#####

abstract type AbstractSurfaceProperties end

#####
##### Interface — per-closure overrides
#####

prognostic_variables(::AbstractSurfaceProperties) = ()
flux_variables(::AbstractSurfaceProperties) = ()

initial_state(::AbstractSurfaceProperties, ::Symbol, grid) = CenterField(grid)
initial_flux(::AbstractSurfaceProperties, ::Symbol, grid) = CenterField(grid)

step!(::AbstractSurfaceProperties, state, fluxes, surface, grid, Δt) = nothing
update_diagnostics!(::AbstractSurfaceProperties, state, fluxes, surface, grid) = nothing

#####
##### Atmosphere-facing accessors
#####

function albedo end
function emissivity end
function momentum_roughness_length end
function scalar_roughness_length end
