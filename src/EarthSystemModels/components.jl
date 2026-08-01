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
