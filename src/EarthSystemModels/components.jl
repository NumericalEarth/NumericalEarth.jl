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

exchange_grid(atmosphere, ocean, sea_ice) = ocean.model.grid

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

function interpolate_state! end
function update_net_fluxes! end

# Fallbacks for a  generic component model
update_net_fluxes!(coupled_model, component) = nothing
interpolate_state!(exchanger, grid, component, coupled_model) = nothing

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
    TT = typeof(clock.time)
    simulation_time = simulation.model.clock.time
    typeof(simulation_time) === TT && return simulation
    throw(ArgumentError(string(
        "the simulation clock tracks time as ", typeof(simulation_time),
        " but the EarthSystemModel clock uses ", TT, ". A Simulation's clock type ",
        "follows its grid and cannot be coerced; rebuild the simulation on a grid ",
        "with float type ", TT, ", or construct the EarthSystemModel with a matching `clock`.")))
end

# Return a clock matching `clock`'s time type, copied from `old`, or `nothing` when `old` already has that
# time type. Components extending `adopt_clock` use this to decide whether — and to what — to coerce.
function matching_clock(old::Clock, clock)
    TT = typeof(clock.time)
    typeof(old.time) === TT && return nothing
    return Clock{TT}(time = convert(TT, old.time),
                     last_Δt = old.last_Δt,
                     last_stage_Δt = old.last_stage_Δt,
                     iteration = old.iteration,
                     stage = old.stage)
end

warn_clock_coercion(component, new_clock) =
    @warn string(summary(component), " tracks time as ", typeof(component.clock.time),
                 " but the EarthSystemModel clock uses ", typeof(new_clock.time),
                 "; coercing the component clock to keep components synchronized.")

# Rebuild a component that stores an Oceananigans `Clock` in its `clock` field, giving it a clock with the
# same time type as `clock`. Relies on the default field-order constructor, so it works for any component
# whose type parameters are all inferable from its fields. Components with a type parameter that is not
# field-inferable (e.g. a separate float-type parameter) extend `adopt_clock` directly and rebuild themselves.
function reclock(component, clock)
    new_clock = matching_clock(component.clock, clock)
    isnothing(new_clock) && return component
    warn_clock_coercion(component, new_clock)
    names = fieldnames(typeof(component))
    args = ntuple(i -> names[i] === :clock ? new_clock : getfield(component, i), length(names))
    return typeof(component).name.wrapper(args...)
end
