# The land interface's state may be prognostic; its fluxes and diagnostics are checkpointed so they are consistent immediately after pickup.
Oceananigans.prognostic_state(interfaces::ComponentInterfaces) =
    (; atmosphere_land_interface=prognostic_state(interfaces.atmosphere_land_interface))

Oceananigans.prognostic_state(interface::AtmosphereInterface) =
    (; fluxes=prognostic_state(interface.fluxes), state=prognostic_state(interface.state))

Oceananigans.prognostic_state(interface::TiledLandInterface) =
    (; vegetated=prognostic_state(interface.vegetated), bare=prognostic_state(interface.bare),
       fluxes=prognostic_state(interface.fluxes), state=prognostic_state(interface.state))

function Oceananigans.prognostic_state(fields::Union{AtmosphereSurfaceFluxes, TemperatureHumidityState, CanopyState})
    names = fieldnames(typeof(fields))
    return NamedTuple{names}(map(name -> prognostic_state(getproperty(fields, name)), names))
end

# A restored state replaces the massless initial state, also at iteration 0.
function Oceananigans.restore_prognostic_state!(interfaces::ComponentInterfaces, state::NamedTuple)
    for name in keys(state)
        restore_prognostic_state!(getproperty(interfaces, name), state[name])
    end
    interfaces.state_is_set = true
    return interfaces
end

function Oceananigans.restore_prognostic_state!(interface::Union{AtmosphereInterface,
                                                               TiledLandInterface, AtmosphereSurfaceFluxes,
                                                               TemperatureHumidityState, CanopyState}, state::NamedTuple)
    for name in keys(state)
        restore_prognostic_state!(getproperty(interface, name), state[name])
    end
    return interface
end
