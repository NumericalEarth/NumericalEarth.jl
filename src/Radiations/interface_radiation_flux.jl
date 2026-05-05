"""
    InterfaceRadiationFlux{F}

Container for the diagnostic radiative fluxes at an air–surface interface.
The same struct type is instantiated per surface (ocean, sea ice, snow, ...).

Fields
======
- `upwelling_longwave    :: F`   ϵσT⁴
- `downwelling_longwave  :: F`   ϵℐꜜˡʷ (absorbed by the surface)
- `downwelling_shortwave :: F`   (1−α)ℐꜜˢʷ (transmitted into the surface)
"""
struct InterfaceRadiationFlux{F}
    upwelling_longwave    :: F
    downwelling_longwave  :: F
    downwelling_shortwave :: F
end

function InterfaceRadiationFlux(grid)
    F = Field{Center, Center, Nothing}
    return InterfaceRadiationFlux(F(grid), F(grid), F(grid))
end

InterfaceRadiationFlux(::Nothing) = InterfaceRadiationFlux(ntuple(_ -> ZeroField(), 3)...)

Adapt.adapt_structure(to, fluxes::InterfaceRadiationFlux) =
    InterfaceRadiationFlux(Adapt.adapt(to, fluxes.upwelling_longwave),
                           Adapt.adapt(to, fluxes.downwelling_longwave),
                           Adapt.adapt(to, fluxes.downwelling_shortwave))

on_architecture(arch, fluxes::InterfaceRadiationFlux) =
    InterfaceRadiationFlux(on_architecture(arch, fluxes.upwelling_longwave),
                           on_architecture(arch, fluxes.downwelling_longwave),
                           on_architecture(arch, fluxes.downwelling_shortwave))
