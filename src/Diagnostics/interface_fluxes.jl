using ..SeaIces: FreezingLimitedEarthSystemModel

@inline flux_field(condition) = condition
@inline flux_field(bc::MultipleFluxes) = bc.flux_field
@inline flux_field(bc::DiscreteBoundaryFunction) = flux_field(bc.func)

const NoSeaIceOceanInterfaceModel = Union{NoSeaIceInterfaceModel,
                                          NoOceanInterfaceModel,
                                          NoInterfaceModel}

ocean_freshwater_temperature_flux(esm, grid) = ZeroField()
ocean_freshwater_temperature_flux(esm, ::MutableGridOfSomeKind) =
    esm.interfaces.net_fluxes.ocean.freshwater_heat_content

ocean_freshwater_temperature_flux(esm::EarthSystemModel) =
    ocean_freshwater_temperature_flux(esm, esm.ocean.model.grid)

@inline function top_advective_temperature_flux(i, j, k, grid, advection, fields)
    kᴺ = grid.Nz + 1
    area = Azᶜᶜᶠ(i, j, kᴺ, grid)
    flux = _advective_tracer_flux_z(i, j, kᴺ, grid, advection, fields.w, fields.T)
    return flux / area
end

ocean_top_advective_temperature_flux(esm, T, ::MutableGridOfSomeKind) = ZeroField()

function ocean_top_advective_temperature_flux(esm, T, grid)
    model = esm.ocean.model
    fields = (w = model.transport_velocities.w, T)
    return KernelFunctionOperation{Center, Center, Nothing}(top_advective_temperature_flux,
                                                            grid,
                                                            model.advection.T,
                                                            fields)
end

ocean_top_advective_temperature_flux(esm::EarthSystemModel, T=esm.ocean.model.tracers.T) =
    ocean_top_advective_temperature_flux(esm, T, esm.ocean.model.grid)

"""
    ocean_top_advective_heat_flux(esm::EarthSystemModel, T=esm.ocean.model.tracers.T)

Return the outward-positive heat flux through the fixed upper computational
boundary. This is zero when the vertical grid follows the free surface. Pass the
temperature field used by the corresponding time-integration stage when checking
a discrete time-step budget.
"""
function ocean_top_advective_heat_flux(esm::EarthSystemModel, T=esm.ocean.model.tracers.T)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    return ρᵒᶜ * cᵒᶜ * ocean_top_advective_temperature_flux(esm, T)
end

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

function frazil_temperature_flux(esm::FreezingLimitedEarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    return 1 / (ρᵒᶜ * cᵒᶜ) * frazil_heat_flux(esm)
end

"""
    net_ocean_temperature_flux(esm::EarthSystemModel)

Return the complete net temperature flux (K m s⁻¹) out of the ocean surface.
Freshwater heat is included for mutable grids and is zero for fixed grids.
"""
function net_ocean_temperature_flux(esm::EarthSystemModel)
    Jᵀ = flux_field(esm.ocean.model.tracers.T.boundary_conditions.top.condition)
    Jᴴ = ocean_freshwater_temperature_flux(esm)
    return Jᵀ - Jᴴ + frazil_temperature_flux(esm)
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

frazil_heat_flux(esm::FreezingLimitedEarthSystemModel) = esm.sea_ice.frazil_heat

frazil_heat_flux(::NoSeaIceOceanInterfaceModel) = ZeroField()

"""
    net_ocean_heat_flux(esm::EarthSystemModel)

Return the complete outward-positive heat flux (W m⁻²) at the ocean surface.
This includes surface exchange, frazil heat, and freshwater heat on a mutable
grid.
"""
function net_ocean_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    return ρᵒᶜ * cᵒᶜ * net_ocean_temperature_flux(esm)
end

"""
    ocean_freshwater_heat_flux(esm::EarthSystemModel)

Return the heat carried into the ocean by freshwater at the ocean surface
temperature (W m⁻²), positive into the ocean.
"""
function ocean_freshwater_heat_flux(esm::EarthSystemModel)
    ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
    cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
    Jᴴ = ocean_freshwater_temperature_flux(esm)
    return ρᵒᶜ * cᵒᶜ * Jᴴ
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
