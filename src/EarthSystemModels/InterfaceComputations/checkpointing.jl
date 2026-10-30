# The land interface's storage is prognostic; its diagnostics are checkpointed so they are consistent immediately after pickup.
Oceananigans.prognostic_state(interfaces::ComponentInterfaces) =
    (; atmosphere_land_interface=prognostic_state(interfaces.atmosphere_land_interface))

Oceananigans.prognostic_state(interface::AtmosphereInterface) =
    (; fluxes=prognostic_state(interface.fluxes), temperature=prognostic_state(interface.temperature),
       specific_humidity=prognostic_state(interface.specific_humidity), storage=prognostic_state(interface.storage))

Oceananigans.prognostic_state(interface::TiledLandInterface) =
    (; vegetated=prognostic_state(interface.vegetated), bare=prognostic_state(interface.bare),
       fluxes=prognostic_state(interface.fluxes), temperature=prognostic_state(interface.temperature))

function Oceananigans.prognostic_state(fields::Union{AtmosphereSurfaceFluxes, CanopyInterfaceDiagnostics})
    names = fieldnames(typeof(fields))
    return NamedTuple{names}(map(name -> prognostic_state(getproperty(fields, name)), names))
end

function Oceananigans.restore_prognostic_state!(interface::Union{ComponentInterfaces, AtmosphereInterface,
                                                               TiledLandInterface, AtmosphereSurfaceFluxes,
                                                               CanopyInterfaceDiagnostics}, state::NamedTuple)
    for name in keys(state)
        restore_prognostic_state!(getproperty(interface, name), state[name])
    end
    return interface
end
