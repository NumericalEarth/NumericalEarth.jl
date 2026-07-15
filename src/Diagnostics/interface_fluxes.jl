
import ..EarthSystemModels: net_ocean_heat_flux

@inline flux_field(condition) = condition
@inline flux_field(bc::MultipleFluxes) = bc.flux_field
@inline flux_field(bc::DiscreteBoundaryFunction) = flux_field(bc.func)

const NoSeaIceOceanInterfaceModel = Union{NoSeaIceInterfaceModel,
                                          NoOceanInterfaceModel,
                                          NoInterfaceModel}

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

frazil_temperature_flux(::NoSeaIceOceanInterfaceModel) = ZeroField()

"""
    net_ocean_temperature_flux(esm::EarthSystemModel)

Return the net temperature flux (K m s⁻¹) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_temperature_flux(esm::EarthSystemModel)
    Jᵀ = flux_field(esm.ocean.model.tracers.T.boundary_conditions.top.condition)
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

sea_ice_ocean_temperature_flux(::NoSeaIceOceanInterfaceModel) = ZeroField()

"""
    atmosphere_ocean_temperature_flux(esm::EarthSystemModel)

Return the atmosphere-ocean temperature flux (K m s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
atmosphere_ocean_temperature_flux(esm::EarthSystemModel) =
        net_ocean_temperature_flux(esm) - sea_ice_ocean_temperature_flux(esm)


###########################
### Heat fluxes
###########################

"""
    frazil_heat_flux(esm::EarthSystemModel)

Return the two-dimensional frazil heat flux (W m⁻²) in a coupled `esm`.
"""
frazil_heat_flux(esm::EarthSystemModel) =
    esm.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat

frazil_heat_flux(::NoSeaIceOceanInterfaceModel) = ZeroField()

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
    sea_ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes
    return sea_ice_ocean_fluxes.interface_heat + frazil_heat_flux(esm)
end

sea_ice_ocean_heat_flux(::NoSeaIceOceanInterfaceModel) = ZeroField()

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
net_ocean_salinity_flux(esm::EarthSystemModel) =
    flux_field(esm.ocean.model.tracers.S.boundary_conditions.top.condition)


"""
    sea_ice_ocean_salinity_flux(esm::EarthSystemModel)

Return the sea ice-ocean salinity flux (g/kg m s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
sea_ice_ocean_salinity_flux(esm::EarthSystemModel) =
    esm.interfaces.sea_ice_ocean_interface.fluxes.salt

sea_ice_ocean_salinity_flux(::NoSeaIceOceanInterfaceModel) = ZeroField()

"""
    atmosphere_ocean_salinity_flux(esm::EarthSystemModel)

Return the atmosphere-ocean salinity flux (g/kg m s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
atmosphere_ocean_salinity_flux(esm::EarthSystemModel) =
    net_ocean_salinity_flux(esm) - sea_ice_ocean_salinity_flux(esm)


###########################
### Freshwater mass fluxes
###########################

"""
    net_ocean_freshwater_flux(esm::EarthSystemModel)

Return the net freshwater mass flux (kg m⁻² s⁻¹) at the ocean's surface in a coupled `esm`.
"""
function net_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    Sᵒᶜ = convert(typeof(ρᵒᶜ), reference_salinity)
    return - ρᵒᶜ / Sᵒᶜ * net_ocean_salinity_flux(esm)
end

"""
    sea_ice_ocean_freshwater_flux(esm::EarthSystemModel)

Return the sea ice-ocean freshwater mass flux (kg m⁻² s⁻¹) at the sea ice-ocean interface
in a coupled `esm`.
"""
function sea_ice_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    Sᵒᶜ = convert(typeof(ρᵒᶜ), reference_salinity)
    return - ρᵒᶜ / Sᵒᶜ * sea_ice_ocean_salinity_flux(esm)
end

sea_ice_ocean_freshwater_flux(::NoSeaIceOceanInterfaceModel; kwargs...) = ZeroField()

"""
    atmosphere_ocean_freshwater_flux(esm::EarthSystemModel)

Return the atmosphere-ocean freshwater mass flux (kg m⁻² s⁻¹) at the atmosphere-ocean
interface in a coupled `esm`.
"""
function atmosphere_ocean_freshwater_flux(esm::EarthSystemModel; reference_salinity = 35)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    Sᵒᶜ = convert(typeof(ρᵒᶜ), reference_salinity)
    return - ρᵒᶜ / Sᵒᶜ * atmosphere_ocean_salinity_flux(esm)
end
