struct TracerFluxUnits end
struct HeatFreshwaterMassUnits end

"""
    interface_flux_outputs(coupled_model::EarthSystemModel;
                           units = HeatFreshwaterMassUnits(),
                           separate_sea_ice = false,
                           reference_salinity = 35)

Return 2D heat and freshwater mass fluxes _or_ the temperature and salt fluxes respectively,
derived from the ocean--sea ice model's top tracer boundary conditions. Note that the difference,
e.g., of heat and temperature fluxes is just a multiplicative factor; same for the difference
between freshwater mass fluxes and salt fluxes.

Arguments
=========

* `coupled_model`: An `EarthSystemModel`.


Keyword Arguments
=================

* `separate_sea_ice`: If set to `true`, then returns separate fluxes for the ocean and
                      sea ice model components. If `false` (default), the sum of the
                      tracer fluxes for the ocean and sea ice model components are output.

* `units`: If `TracerFluxUnits()`, then each of the fluxes are output in units of `tracer`
           multiplied by a velocity per unit area, i.e., `tracer_unit` m⁻¹ s⁻¹.
           If `HeatFreshwaterMassUnits()` (default), then the temperature fluxes are converted
           to heat fluxes (W m⁻²) and the salt fluxes are converted to freshwater mass
           fluxes (kg m⁻² s⁻¹).

* `reference_salinity`: Reference salinity ``S₀`` used to convert the salt fluxes to freshwater
                        mass fluxes, i.e., ``-ρ₀ Jˢ / S₀``, where ``Jˢ`` is the salt fluxes.
                        Default: 35 g/kg.
"""
function interface_flux_outputs(coupled_model::EarthSystemModel;
                                units = HeatFreshwaterMassUnits(),
                                separate_sea_ice = false,
                                reference_salinity = 35)

    (units isa HeatFreshwaterMassUnits || units isa TracerFluxUnits) ||
        throw(ArgumentError("units must be `HeatFreshwaterMassUnits()` or `TracerFluxUnits()`"))

    temperature_outputs = temperature_flux_outputs(coupled_model; units, separate_sea_ice)
    salinity_outputs = salinity_flux_outputs(coupled_model; units, separate_sea_ice, reference_salinity)

    return merge(temperature_outputs, salinity_outputs)
end


function temperature_flux_outputs(coupled_model::EarthSystemModel;
                                  units = HeatFreshwaterMassUnits(),
                                  separate_sea_ice = false)

    temperature_flux = coupled_model.ocean.model.tracers.T.boundary_conditions.top.condition

    ρ₀ = coupled_model.interfaces.ocean_properties.reference_density
    cₚ = coupled_model.interfaces.ocean_properties.heat_capacity

    convert_temperature_flux(Jᵀ, ::TracerFluxUnits) = Jᵀ
    convert_temperature_flux(Jᵀ, ::HeatFreshwaterMassUnits) = Field(ρ₀ * cₚ * Jᵀ)

    ice_ocean_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes
    required = (:frazil_heat, :interface_heat)

    for name in required
        hasproperty(ice_ocean_fluxes, name) ||
            throw(ArgumentError("Missing required interface flux field: $(name)."))
    end

    frazil_heat_flux = getfield(ice_ocean_fluxes, :frazil_heat)
    heat_flux = temperature_flux + frazil_heat_flux

    outputs = (; heat_flux = convert_temperature_flux(heat_flux, units))

    if separate_sea_ice
        sea_ice_heat_flux = getfield(ice_ocean_fluxes, :interface_heat) + frazil_heat_flux
        ocean_heat_flux = heat_flux - sea_ice_heat_flux

        outputs = merge(outputs, (; ocean_heat_flux = convert_temperature_flux(ocean_heat_flux, units),
                                    sea_ice_heat_flux = convert_temperature_flux(sea_ice_heat_flux, units)))
    end

    return outputs
end

function salinity_flux_outputs(coupled_model::EarthSystemModel;
                               units = HeatFreshwaterMassUnits(),
                               separate_sea_ice = false,
                               reference_salinity = 35)

    salinity_flux = coupled_model.ocean.model.tracers.S.boundary_conditions.top.condition

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρ₀ = ocean_properties.reference_density
    S₀ = convert(typeof(ρ₀), reference_salinity)

    convert_salinity_flux(Jˢ, ::TracerFluxUnits) = Jˢ
    convert_salinity_flux(Jˢ, ::HeatFreshwaterMassUnits) = Field(-ρ₀ * Jˢ / S₀)

    ice_ocean_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes
    required = (:salt,)

    for name in required
        hasproperty(ice_ocean_fluxes, name) ||
            throw(ArgumentError("Missing required interface flux field: $(name)."))
    end

    freshwater_flux = salinity_flux

    outputs = (; freshwater_flux = convert_salinity_flux(freshwater_flux, units))

    if separate_sea_ice
        sea_ice_freshwater_flux = getfield(ice_ocean_fluxes, :salt)
        ocean_freshwater_flux = freshwater_flux - sea_ice_freshwater_flux

        outputs = merge(outputs, (; ocean_freshwater_flux = convert_salinity_flux(ocean_freshwater_flux, units),
                                    sea_ice_freshwater_flux = convert_salinity_flux(sea_ice_freshwater_flux, units)))
    end

    return outputs
end
