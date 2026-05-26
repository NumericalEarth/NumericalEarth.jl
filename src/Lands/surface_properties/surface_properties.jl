#####
##### `AbstractSurfaceProperties` — slab-land aerodynamic property
##### closure interface.
#####
##### Provides the aerodynamic properties the atmosphere needs to close
##### momentum and scalar fluxes:
#####
#####   - `momentum_roughness_length(surface, state)`
#####   - `scalar_roughness_length(surface, state)`
#####
##### Radiative properties (albedo, emissivity) live on the top-level
##### `radiation` component as `radiation.surface_properties.land`,
##### mirroring how ocean and sea-ice radiative properties are handled.
#####

abstract type AbstractSurfaceProperties end

#####
##### Interface — per-closure overrides
#####

flux_variables(::AbstractSurfaceProperties) = ()

initial_flux(::AbstractSurfaceProperties, ::Symbol, grid) = CenterField(grid)

step!(::AbstractSurfaceProperties, land, Δt) = nothing
step!(surface::AbstractSurfaceProperties, land, Δt, time) = step!(surface, land, Δt)
update_diagnostics!(::AbstractSurfaceProperties, land) = nothing

#####
##### Atmosphere-facing accessors
#####

function momentum_roughness_length end
function scalar_roughness_length end
