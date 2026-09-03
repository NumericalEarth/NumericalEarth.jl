using Oceananigans.Grids: grid

#####
##### Temperature units
#####

struct DegreesCelsius end
struct DegreesKelvin end

const celsius_to_kelvin = 273.15
@inline convert_to_kelvin(::DegreesCelsius, T::FT) where FT = T + convert(FT, celsius_to_kelvin)
@inline convert_to_kelvin(::DegreesKelvin, T) = T

@inline convert_from_kelvin(::DegreesCelsius, T::FT) where FT = T - convert(FT, celsius_to_kelvin)
@inline convert_from_kelvin(::DegreesKelvin, T) = T

#####
##### generic defaults
#####

# Default: build the exchange grid from the ocean. When the model has no
# ocean / sea ice, fall back to the land grid (used by AtmosphereLandModel).
exchange_grid(atmosphere, ocean, sea_ice, land=nothing) = grid(ocean)
exchange_grid(atmosphere, ::Nothing, ::Nothing, land) = land.grid

# Prescribed fields are FieldTimeSeries; set a `Number` into every time slice.
# `nothing` leaves the field untouched.
set_prescribed_field!(fts, ::Nothing) = nothing
set_prescribed_field!(fts, value::Number) = Oceananigans.set!(fts, value)

#####
##### Functions extended by sea-ice and ocean models
#####

reference_density(::Nothing) = 0
heat_capacity(::Nothing) = 0
ocean_temperature(ocean) = ZeroField()
ocean_salinity(ocean) = ZeroField()
ocean_surface_temperature(ocean) = ZeroField()
ocean_surface_salinity(ocean) = ZeroField()
ocean_surface_velocities(ocean) = ZeroField(), ZeroField()

"""
$(TYPEDSIGNATURES)

Sea surface height ``η`` [m] of the ocean, on a `(Center, Center, Nothing)` field so that a sea-ice
momentum equation can read it at its own topmost index. Its slope enters the ice momentum balance as
``- g ∇η``, which is `f × uᵍ` for a geostrophic surface current: without it the ice cannot ride the
ocean's dynamic topography and the uncompensated Coriolis force is absorbed by the ice-ocean drag.
"""
ocean_surface_height(ocean) = ZeroField()

"""
$(TYPEDSIGNATURES)

Refill the sea surface height field the sea-ice surface-tilt term reads from the ocean state.
"""
ocean_surface_height!(ηˢ, ocean) = nothing

"""
$(TYPEDSIGNATURES)

Ocean velocities averaged over the top `reference_depth` metres, the reference velocities of the sea
ice-ocean drag. The quadratic law `ρ Cᴰ |uⁱ - uᵒ| (uⁱ - uᵒ)` carries a coefficient defined against the
velocity of the under-ice boundary layer, which is tens of metres thick, so referencing it to the
topmost cell brakes the ice against a film the ice itself accelerates. `reference_depth = nothing`
returns [`ocean_surface_velocities`](@ref) unchanged.
"""
surface_layer_velocities(ocean, reference_depth) = ocean_surface_velocities(ocean)
surface_layer_velocities(ocean, ::Nothing) = ocean_surface_velocities(ocean)

"""
$(TYPEDSIGNATURES)

Recompute the sea ice-ocean drag reference velocities from the ocean state. Called once per coupled
step from `update_state!`; a no-op for models whose reference is a plain view of the ocean's own field.
"""
refresh_drag_reference_velocities!(model) = nothing

"""
$(TYPEDSIGNATURES)

Recompute the sea surface height the sea-ice momentum equation reads for its surface-tilt term from
the ocean state. Called once per coupled step from `update_state!`; a no-op when the ice carries no
tilt term.
"""
refresh_ocean_surface_height!(model) = nothing
temperature_units(ocean) = DegreesCelsius()

#####
##### Functions extended by sea-ice models
#####

sea_ice_thickness(::Nothing) = ZeroField()
sea_ice_concentration(::Nothing) = ZeroField()
intercepted_snowfall(::Nothing) = ZeroField()
function default_sea_ice end

#####
##### Functions extended by atmosphere models
#####

function thermodynamics_parameters end
function surface_layer_height end
function boundary_layer_height end

surface_layer_height(::Nothing) = 0
boundary_layer_height(::Nothing) = 0

# Grid-aware surface-layer height, built once and cached in `interfaces.properties`.
# The generic fallback ignores the exchange grid and returns the scalar height
# (prescribed atmospheres carry a fixed measurement height); atmosphere models with
# per-column geometry (e.g. Breeze on a terrain-following grid) override this to
# materialize a 2-D field on the exchange grid.
surface_layer_height(atmosphere, exchange_grid) = surface_layer_height(atmosphere)

#####
##### Functions extended by all component models
#####

"""
    component_model(component)

Return the bare component model from a wrapper. ESM components are sometimes
passed as a bare model (e.g. `Breeze.AtmosphereModel`, `Breeze.RadiativeTransferModel`)
and sometimes as a `Simulation` wrapping that model (e.g. `Simulation{<:Breeze.AtmosphereModel}`).
Component-interface methods that need the underlying model — to reach for
`.grid`, `.velocities`, boundary conditions, etc. — call `component_model(x)` so
they can share one implementation between the wrapped and unwrapped forms. The
default unwraps a `Simulation`; the identity fallback covers bare models.
"""
@inline component_model(sim::Simulation) = sim.model
@inline component_model(component) = component

function interpolate_state! end
function update_net_fluxes! end

# Fallbacks for a  generic component model
update_net_fluxes!(coupled_model, component) = nothing
interpolate_state!(exchanger, grid, component, coupled_model) = nothing

# Fallback for radiative coupling when no radiation is configured.
apply_air_land_radiative_fluxes!(::Any) = nothing

#####
##### Surface (skin) temperature diagnostic
#####

function surface_temperature end
surface_temperature(::Any) = nothing

#####
##### Clock type consistency across components
#####

"""
    adopt_clock(component, clock)

Return `component` reconfigured so that its time is tracked with the same time type as the authoritative
`EarthSystemModel` `clock`. `EarthSystemModel` construction calls this on every component so their clocks
cannot drift apart over long runs — e.g. `Float32` and `Float64` clocks accumulating `Δt` differently across
thousands of days.

Behavior depends on how the component tracks time:

  - the generic method leaves `component` untouched, so a component with its own clock representation (e.g.
    SpeedyWeather or Veros, which track time internally) only extends this method if it needs coercion;
  - a `Simulation`, whose clock type is fixed by its grid, errors on a mismatch since it cannot be coerced;
  - prescribed components that carry an Oceananigans `Clock` extend this method through `reclock`, which
    coerces the clock to the model time type and warns when the type actually changes.
"""
adopt_clock(component, clock) = component
adopt_clock(::Nothing, clock) = nothing

function adopt_clock(simulation::Simulation, clock)
    same_time_type(simulation.model.clock.time, clock.time) && return simulation
    throw(ArgumentError(string(
        "the simulation clock tracks time as ", typeof(simulation.model.clock.time),
        " but the EarthSystemModel clock uses ", typeof(clock.time), ". A Simulation's clock type ",
        "follows its grid and cannot be coerced; rebuild the simulation on a grid ",
        "with float type ", typeof(clock.time), ", or construct the EarthSystemModel with a matching `clock`.")))
end

same_time_type(::TT, ::ST) where {TT, ST} = ST === TT

# Return a clock matching `clock`'s time type (or nothing if clocks are the same)
function matching_clock(old::Clock, clock)
    same_time_type(old.time, clock.time) && return nothing
    TT = typeof(clock.time)
    return Clock{TT}(time = convert(TT, old.time),
                     last_Δt = old.last_Δt,
                     last_stage_Δt = old.last_stage_Δt,
                     iteration = old.iteration,
                     stage = old.stage)
end

warn_clock_coercion(component, new_clock) = @warn string(summary(component), " tracks time as ",  typeof(component.clock.time),
                                                         " but the EarthSystemModel clock uses ", typeof(new_clock.time),
                                                         "; coercing the component clock to keep components synchronized.")

function reclock(component, clock)
    new_clock = matching_clock(component.clock, clock)
    isnothing(new_clock) && return component
    warn_clock_coercion(component, new_clock)
    names = fieldnames(typeof(component))
    args = ntuple(i -> names[i] === :clock ? new_clock : getfield(component, i), length(names))
    return typeof(component).name.wrapper(args...)
end
