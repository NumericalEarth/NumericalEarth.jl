"""
    InterfaceFluxOutputs(model::EarthSystemModel;
                         units = :tracer_flux,
                         separate_sea_ice = false,
                         reference_salinity = 35)

Return 2D heat and freshwater mass fluxes _or_ the temperature and salt fluxes respectively,
derived from the ocean--sea ice model's top tracer boundary conditions. Note that the difference,
e.g., of heat and temperature fluxes is just a multiplicative factor; same for the difference
between freshwater mass fluxes and salt fluxes.

Arguments
=========

* `model`: An `EarthSystemModel`.


Keyword Arguments
=================

* `separate_sea_ice`: If set to `true`, then returns separate fluxes for the ocean and
                      sea ice model components. If `false` (default), the sum of the
                      tracer fluxes for the ocean and sea ice model components are output.

* `units`: If `:tracer_flux`, then each of the fluxes are output in units of `tracer`
           multiplied by a velocity per unit area, i.e., `tracer_unit` m⁻¹ s⁻¹.
           If `:heat_freshwater_mass` (default), then the temperature fluxes are converted
           to heat fluxes (W m⁻²) and the salt fluxes are converted to freshwater mass
           fluxes (kg m⁻² s⁻¹).

* `reference_salinity`: Reference salinity ``S₀`` used to convert the salt fluxes to freshwater
                        mass fluxes, i.e., ``-ρₒ Jˢ / S₀``, where ``Jˢ`` is the salt fluxes.
                        Default: 35 gr/kg.
"""
function InterfaceFluxOutputs(coupled_model::EarthSystemModel;
                              units = :tracer_flux,
                              separate_sea_ice = false,
                              reference_salinity = 35)

    units in (:physical, :tracer) || throw(ArgumentError("`units` must be `:physical` or `:tracer`."))

    T_top_flux = coupled_model.ocean.model.tracers.T.boundary_conditions.top.condition
    S_top_flux = coupled_model.ocean.model.tracers.S.boundary_conditions.top.condition

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρₒ = ocean_properties.reference_density
    cₒ = ocean_properties.heat_capacity
    S₀ = convert(typeof(ρₒ), reference_salinity)

    convert_temperature_flux(Jᵀ) = units === :physical ? Field(ρₒ * cₒ * Jᵀ) : Jᵀ
    convert_salinity_flux(Jˢ) = units === :physical ? Field(-ρₒ * Jˢ / S₀) : Jˢ

    heat_flux = convert_temperature_flux(T_top_flux)
    freshwater_flux = convert_salinity_flux(S_top_flux)

    outputs = (; heat_flux, freshwater_flux)

    if separate_sea_ice
        io_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes
        required = (:frazil_heat, :interface_heat, :salt)

        for name in required
            hasproperty(io_fluxes, name) || throw(ArgumentError("Missing required interface flux field: $(name)."))
        end

        sea_ice_heat_flux = convert_temperature_flux(getfield(io_fluxes, :frazil_heat)) + convert_temperature_flux(getfield(io_fluxes, :interface_heat))
        sea_ice_freshwater_flux = convert_salinity_flux(getfield(io_fluxes, :salt))
        ocean_heat_flux = heat_flux - sea_ice_heat_flux
        ocean_freshwater_flux = freshwater_flux - sea_ice_freshwater_flux

        outputs = merge(outputs, (; ocean_heat_flux,
                                    sea_ice_heat_flux,
                                    ocean_freshwater_flux,
                                    sea_ice_freshwater_flux))
    end

    return outputs
end
