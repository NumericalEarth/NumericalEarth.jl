struct TracerFlux end
struct HeatFreshwaterMass end

"""
    InterfaceFluxOutputs(coupled_model::EarthSystemModel;
                         units = TracerFlux(),
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

* `units`: If `TracerFlux()`, then each of the fluxes are output in units of `tracer`
           multiplied by a velocity per unit area, i.e., `tracer_unit` m⁻¹ s⁻¹.
           If `HeatFreshwaterMass()` (default), then the temperature fluxes are converted
           to heat fluxes (W m⁻²) and the salt fluxes are converted to freshwater mass
           fluxes (kg m⁻² s⁻¹).

* `reference_salinity`: Reference salinity ``S₀`` used to convert the salt fluxes to freshwater
                        mass fluxes, i.e., ``-ρ₀ Jˢ / S₀``, where ``Jˢ`` is the salt fluxes.
                        Default: 35 g/kg.
"""
function InterfaceFluxOutputs(coupled_model::EarthSystemModel;
                              units = HeatFreshwaterMass(),
                              separate_sea_ice = false,
                              reference_salinity = 35)

    (units isa HeatFreshwaterMass || units isa TracerFlux) ||
        throw(ArgumentError("units must be `HeatFreshwaterMass()` or `TracerFlux()`"))

    T_top_flux = coupled_model.ocean.model.tracers.T.boundary_conditions.top.condition
    S_top_flux = coupled_model.ocean.model.tracers.S.boundary_conditions.top.condition

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρ₀ = ocean_properties.reference_density
    cₚ = ocean_properties.heat_capacity
    S₀ = convert(typeof(ρ₀), reference_salinity)

    convert_temperature_flux(Jᵀ, ::TracerFlux) = Jᵀ
    convert_temperature_flux(Jᵀ, ::HeatFreshwaterMass) = Field(ρ₀ * cₚ * Jᵀ)
       convert_salinity_flux(Jˢ, ::TracerFlux) = Jˢ
       convert_salinity_flux(Jˢ, ::HeatFreshwaterMass) = Field(-ρ₀ * Jˢ / S₀)

    heat_flux = convert_temperature_flux(T_top_flux, units)
    freshwater_flux = convert_salinity_flux(S_top_flux, units)

    outputs = (; heat_flux, freshwater_flux)

    if separate_sea_ice
        io_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes
        required = (:frazil_heat, :interface_heat, :salt)

        for name in required
            hasproperty(io_fluxes, name) || throw(ArgumentError("Missing required interface flux field: $(name)."))
        end

        sea_ice_heat_flux = convert_temperature_flux(getfield(io_fluxes, :frazil_heat), units) + convert_temperature_flux(getfield(io_fluxes, :interface_heat), units)
        sea_ice_freshwater_flux = convert_salinity_flux(getfield(io_fluxes, :salt), units)
        ocean_heat_flux = heat_flux - sea_ice_heat_flux
        ocean_freshwater_flux = freshwater_flux - sea_ice_freshwater_flux

        outputs = merge(outputs, (; ocean_heat_flux,
                                    sea_ice_heat_flux,
                                    ocean_freshwater_flux,
                                    sea_ice_freshwater_flux))
    end

    return outputs
end
