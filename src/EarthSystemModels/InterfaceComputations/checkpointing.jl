# Land interface diagnostics contain the completed step's average fluxes and temperatures.
Oceananigans.prognostic_state(interfaces::ComponentInterfaces) =
    (; atmosphere_land_interface=prognostic_state(interfaces.atmosphere_land_interface))

Oceananigans.prognostic_state(interface::AtmosphereInterface) =
    (; fluxes=prognostic_state(interface.fluxes), temperature=prognostic_state(interface.temperature),
       specific_humidity=prognostic_state(interface.specific_humidity))

function Oceananigans.prognostic_state(fields::Union{AtmosphereSurfaceFluxes, CanopyInterfaceDiagnostics})
    names = fieldnames(typeof(fields))
    return NamedTuple{names}(map(name -> prognostic_state(getproperty(fields, name)), names))
end

function Oceananigans.restore_prognostic_state!(interface::Union{ComponentInterfaces, AtmosphereInterface,
                                                               AtmosphereSurfaceFluxes,
                                                               CanopyInterfaceDiagnostics}, state::NamedTuple)
    for name in keys(state)
        restore_prognostic_state!(getproperty(interface, name), state[name])
    end
    return interface
end
