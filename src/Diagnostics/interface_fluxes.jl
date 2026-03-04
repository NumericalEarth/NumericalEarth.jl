
###########################
### Heat fluxes
###########################

"""
    net_ocean_heat_flux(esm::EarthSystemModel)

Return the net heat flux (W m⁻²) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_heat_flux(esm::EarthSystemModel)
    Jᵀ = esm.ocean.model.tracers.T.boundary_conditions.top.condition # temperature flux
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    frazil_heat_flux = esm.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
    return ρᵒᶜ * cᵒᶜ * Jᵀ + frazil_heat_flux
end

"""
    sea_ice_ocean_heat_flux(esm::EarthSystemModel)

Return the sea ice-ocean heat flux (W m⁻²) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_heat_flux(esm::EarthSystemModel)
    frazil_heat_flux = esm.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
    return esm.interfaces.sea_ice_ocean_interface.fluxes.interface_heat + frazil_heat_flux
end

"""
    atmosphere_ocean_heat_flux(esm::EarthSystemModel)

Return the atmosphere-ocean heat flux (W m⁻²) at the atmosphere-ocean
interface in a coupled `esm`.
"""
atmosphere_ocean_heat_flux(esm::EarthSystemModel) =
    net_ocean_heat_flux(esm) - sea_ice_ocean_heat_flux(esm)


###########################
### Freshwater mass fluxes
###########################

"""
    net_ocean_freshwater_flux(esm::EarthSystemModel)

Return the net freshwater mass flux (kg m⁻² s⁻¹) at the ocean's surface in a coupled `esm`.
"""
net_ocean_freshwater_flux(esm::EarthSystemModel) =
    sea_ice_ocean_freshwater_flux(esm) + atmosphere_ocean_freshwater_flux(esm)

"""
    sea_ice_ocean_freshwater_flux(esm::EarthSystemModel)

Return the sea ice-ocean freshwater mass flux (kg m⁻² s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
sea_ice_ocean_freshwater_flux(esm::EarthSystemModel) =
    esm.interfaces.sea_ice_ocean_interface.fluxes.freshwater_flux

"""
    atmosphere_ocean_freshwater_flux(esm::EarthSystemModel)

Return the atmosphere-ocean freshwater mass flux (kg m⁻² s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_freshwater_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    return ρᵒᶜ * esm.interfaces.atmosphere_ocean_interface.fluxes.freshwater_flux
end
