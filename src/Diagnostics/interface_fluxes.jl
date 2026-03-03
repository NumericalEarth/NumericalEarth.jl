
###########################
### Temperature fluxes
###########################

function frazil_temperature_flux(esm::EarthSystemModel)
    ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes
    frazil_temperature_flux = getfield(ice_ocean_fluxes, :frazil_heat)
    return frazil_temperature_flux
end

function net_ocean_temperature_flux(esm::EarthSystemModel)
    Jᵀ = esm.ocean.model.tracers.T.boundary_conditions.top.condition
    ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes
    net_ocean_temperature_flux = Jᵀ + frazil_temperature_flux(esm)
    return Field(net_ocean_temperature_flux)
end

function sea_ice_ocean_temperature_flux(esm::EarthSystemModel)
    ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes
    sea_ice_ocean_temperature_flux =
        getfield(ice_ocean_fluxes, :interface_heat) + frazil_heat_flux(esm::EarthSystemModel)
    return Field(sea_ice_ocean_temperature_flux)
end

function atmosphere_ocean_temperature_flux(esm::EarthSystemModel)
    atmosphere_ocean_temperature_flux = net_ocean_temperature_flux(esm) - sea_ice_ocean_temperature_flux(esm)
    return Field(atmosphere_ocean_temperature_flux)
end


###########################
### Heat fluxes
###########################

function frazil_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    frazil_heat_flux = ρᵒᶜ * cᵒᶜ * frazil_temperature_flux(esm)
    return Field(frazil_heat_flux)
end

function net_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    net_ocean_heat_flux = ρᵒᶜ * cᵒᶜ * net_ocean_temperature_flux(esm)
    return Field(net_ocean_heat_flux)
end

function sea_ice_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    sea_ice_ocean_heat_flux = ρᵒᶜ * cᵒᶜ * sea_ice_ocean_temperature_flux(esm)
    return Field(sea_ice_ocean_heat_flux)
end

function atmosphere_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    ocean_heat_flux = ρᵒᶜ * cᵒᶜ * atmosphere_ocean_temperature_flux(esm)
    return Field(atmosphere_ocean_heat_flux)
end


###########################
### Salinity fluxes
###########################

function net_ocean_salinity_flux(esm::EarthSystemModel)
    Jˢ = esm.ocean.model.tracers.S.boundary_conditions.top.condition
    return Jˢ
end

function sea_ice_ocean_salinity_flux(esm::EarthSystemModel)
    ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes
    sea_ice_ocean_salinity_flux = getfield(ice_ocean_fluxes, :salt)
    return sea_ice_ocean_salinity_flux
end

function atmosphere_ocean_salinity_flux(esm::EarthSystemModel)
    atmosphere_ocean_salinity_flux = net_ocean_salinity_flux(esm) - sea_ice_ocean_salinity_flux(esm)
    return Field(atmosphere_ocean_salinity_flux)
end


###########################
### Freshwater mass fluxes
###########################

function net_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    S₀ = convert(typeof(ρᵒᶜ), reference_salinity)
    net_ocean_frashwater_flux = - ρᵒᶜ / S₀ * net_ocean_salinity_flux(esm)
    return Field(net_ocean_frashwater_flux)
end

function sea_ice_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    S₀ = convert(typeof(ρᵒᶜ), reference_salinity)
    sea_ice_ocean_freshwater_flux = - ρᵒᶜ / S₀ * sea_ice_ocean_salinity_flux(esm)
    return Field(sea_ice_ocean_freshwater_flux)
end

function atmosphere_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    S₀ = convert(typeof(ρᵒᶜ), reference_salinity)
    atmosphere_ocean_freshwater_flux = - ρᵒᶜ / S₀ * atmosphere_ocean_salinity_flux(esm)
    return Field(atmosphere_ocean_freshwater_flux)
end
