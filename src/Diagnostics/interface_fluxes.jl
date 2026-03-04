
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
    return 1 / (ρᵒᶜ * cᵒᶜ) * frazil_heat_flux(esm)
end

"""
    net_ocean_temperature_flux(esm::EarthSystemModel)

Return the net temperature flux (K m s⁻¹) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_temperature_flux(esm::EarthSystemModel)
    Jᵀ = esm.ocean.model.tracers.T.boundary_conditions.top.condition
    return Jᵀ + frazil_temperature_flux(esm)
end


"""
    sea_ice_ocean_temperature_flux(esm::EarthSystemModel)

Return the sea ice-ocean temperature flux (K m s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_temperature_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    return 1 / (ρᵒᶜ * cᵒᶜ) * sea_ice_ocean_heat_flux(esm)
end


"""
    atmosphere_ocean_temperature_flux(esm::EarthSystemModel)

Return the atmosphere-ocean temperature flux (K m s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_temperature_flux(esm::EarthSystemModel)
    return net_ocean_temperature_flux(esm) - sea_ice_ocean_temperature_flux(esm)
end


###########################
### Heat fluxes
###########################

"""
    frazil_heat_flux(esm::EarthSystemModel)

Return the two-dimensional frazil heat flux (W m⁻²) in a coupled `esm`.
"""
function frazil_heat_flux(esm::EarthSystemModel)
    return esm.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
end

"""
    net_ocean_heat_flux(esm::EarthSystemModel)

Return the net heat flux (W m⁻²) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    return ρᵒᶜ * cᵒᶜ * net_ocean_temperature_flux(esm)
end

"""
    sea_ice_ocean_heat_flux(esm::EarthSystemModel)

Return the sea ice-ocean heat flux (W m⁻²) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_heat_flux(esm::EarthSystemModel)
    return esm.interfaces.sea_ice_ocean_interface.fluxes.interface_heat + frazil_heat_flux(esm)
end

"""
    atmosphere_ocean_heat_flux(esm::EarthSystemModel)

Return the atmosphere-ocean heat flux (W m⁻²) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    return ρᵒᶜ * cᵒᶜ * atmosphere_ocean_temperature_flux(esm)
end


###########################
### Salinity fluxes
###########################

"""
    net_ocean_salinity_flux(esm::EarthSystemModel)

Return the net salinity flux (g/kg m s⁻¹) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_salinity_flux(esm::EarthSystemModel)
    return esm.ocean.model.tracers.S.boundary_conditions.top.condition
end


"""
    sea_ice_ocean_salinity_flux(esm::EarthSystemModel)

Return the sea ice-ocean salinity flux (g/kg m s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_salinity_flux(esm::EarthSystemModel)
    return esm.interfaces.sea_ice_ocean_interface.fluxes.salt
end

"""
    atmosphere_ocean_salinity_flux(esm::EarthSystemModel)

Return the atmosphere-ocean salinity flux (g/kg m s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_salinity_flux(esm::EarthSystemModel)
    return net_ocean_salinity_flux(esm) - sea_ice_ocean_salinity_flux(esm)
end


###########################
### Freshwater mass fluxes
###########################

"""
    net_ocean_freshwater_flux(esm::EarthSystemModel)

Return the net freshwater mass flux (kg m⁻² s⁻¹) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    S₀ = convert(typeof(ρᵒᶜ), reference_salinity)
    return - ρᵒᶜ / S₀ * net_ocean_salinity_flux(esm)
end

"""
    sea_ice_ocean_freshwater_flux(esm::EarthSystemModel)

Return the sea ice-ocean freshwater mass flux (kg m⁻² s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    S₀ = convert(typeof(ρᵒᶜ), reference_salinity)
    return - ρᵒᶜ / S₀ * sea_ice_ocean_salinity_flux(esm)
end

"""
    atmosphere_ocean_freshwater_flux(esm::EarthSystemModel)

Return the atmosphere-ocean freshwater mass flux (kg m⁻² s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    S₀ = convert(typeof(ρᵒᶜ), reference_salinity)
    return - ρᵒᶜ / S₀ * atmosphere_ocean_salinity_flux(esm)
end
