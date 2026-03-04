
###########################
### Temperature fluxes
###########################

"""
    frazil_temperature_flux(esm::EarthSystemModel)

Return the two-dimensional frazil temperature flux (K m s⁻¹) in a coupled `esm`.
"""
function frazil_temperature_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    frazil_temperature_flux = 1 / (ρᵒᶜ * cᵒᶜ) * frazil_heat_flux(esm)
    return Field(frazil_temperature_flux)
end

"""
    net_ocean_temperature_flux(esm::EarthSystemModel)

Return the net temperature flux (K m s⁻¹) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_temperature_flux(esm::EarthSystemModel)
    Jᵀ = esm.ocean.model.tracers.T.boundary_conditions.top.condition
    net_ocean_temperature_flux = Jᵀ + frazil_temperature_flux(esm)
    return Field(net_ocean_temperature_flux)
end


"""
    sea_ice_ocean_temperature_flux(esm::EarthSystemModel)

Return the sea ice-ocean temperature flux (K m s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_temperature_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    sea_ice_ocean_temperature_flux = 1 / (ρᵒᶜ * cᵒᶜ) * sea_ice_ocean_heat_flux(esm)
    return Field(sea_ice_ocean_temperature_flux)
end


"""
    atmosphere_ocean_temperature_flux(esm::EarthSystemModel)

Return the atmosphere-ocean temperature flux (K m s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_temperature_flux(esm::EarthSystemModel)
    atmosphere_ocean_temperature_flux =
        net_ocean_temperature_flux(esm) - sea_ice_ocean_temperature_flux(esm)
    return Field(atmosphere_ocean_temperature_flux)
end


###########################
### Heat fluxes
###########################

"""
    frazil_heat_flux(esm::EarthSystemModel)

Return the two-dimensional frazil heat flux (W m⁻²) in a coupled `esm`.
"""
function frazil_heat_flux(esm::EarthSystemModel)
    frazil_heat_flux = esm.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
    return frazil_heat_flux
end

"""
    net_ocean_heat_flux(esm::EarthSystemModel)

Return the net heat flux (W m⁻²) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    net_ocean_heat_flux = ρᵒᶜ * cᵒᶜ * net_ocean_temperature_flux(esm)
    return Field(net_ocean_heat_flux)
end

"""
    sea_ice_ocean_heat_flux(esm::EarthSystemModel)

Return the sea ice-ocean heat flux (W m⁻²) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_heat_flux(esm::EarthSystemModel)
    sea_ice_ocean_heat_flux =
        esm.interfaces.sea_ice_ocean_interface.fluxes.interface_heat + frazil_heat_flux(esm)
    return Field(sea_ice_ocean_heat_flux)
end

"""
    atmosphere_ocean_heat_flux(esm::EarthSystemModel)

Return the atmosphere-ocean heat flux (W m⁻²) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    atmosphere_ocean_heat_flux = ρᵒᶜ * cᵒᶜ * atmosphere_ocean_temperature_flux(esm)
    return Field(atmosphere_ocean_heat_flux)
end

###########################
### Freshwater volume fluxes
###########################

"""
    net_ocean_freshwater_flux(esm::EarthSystemModel)

Return the net freshwater volume flux (m s⁻¹) at the ocean's surface in a coupled `esm`.
"""
net_ocean_freshwater_flux(esm::EarthSystemModel) = sea_ice_ocean_freshwater_flux(esm) + atmosphere_ocean_freshwater_flux(esm)

"""
    sea_ice_ocean_freshwater_flux(esm::EarthSystemModel)

Return the sea ice-ocean freshwater volume flux (m s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
sea_ice_ocean_freshwater_flux(esm::EarthSystemModel) = Field(esm.interfaces.sea_ice_ocean_interface.fluxes.freshwater_flux)

"""
    atmosphere_ocean_freshwater_flux(esm::EarthSystemModel)

Return the atmosphere-ocean freshwater volume flux (m s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
atmosphere_ocean_freshwater_flux(esm::EarthSystemModel) = Field(esm.interfaces.atmosphere_ocean_interface.fluxes.freshwater_flux)
