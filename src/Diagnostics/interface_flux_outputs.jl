"""
    InterfaceFluxOutputs(coupled_model; isolate_sea_ice=false, units=:physical, reference_salinity=35)

Return 2D surface flux outputs derived from ocean top tracer boundary conditions.

By default (`units=:physical`), outputs are converted to physical units:
- heat flux: `W m⁻²`
- freshwater-equivalent flux: `kg m⁻² s⁻¹`

Set `units=:tracer` to return raw tracer-flux units.
If `isolate_sea_ice=true`, the output also includes ocean and sea-ice components.
"""
function InterfaceFluxOutputs(coupled_model; isolate_sea_ice=false, units=:physical, reference_salinity=35)
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

    if isolate_sea_ice
        ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
        required = (:ocean_temperature_flux, :sea_ice_temperature_flux, :ocean_salinity_flux, :sea_ice_salinity_flux)

        for name in required
            hasproperty(ao_fluxes, name) || throw(ArgumentError("Missing required interface flux field: $(name)."))
        end

        ocean_heat_flux = convert_temperature_flux(getfield(ao_fluxes, :ocean_temperature_flux))
        sea_ice_heat_flux = convert_temperature_flux(getfield(ao_fluxes, :sea_ice_temperature_flux))
        ocean_freshwater_flux = convert_salinity_flux(getfield(ao_fluxes, :ocean_salinity_flux))
        sea_ice_freshwater_flux = convert_salinity_flux(getfield(ao_fluxes, :sea_ice_salinity_flux))

        outputs = merge(outputs, (; ocean_heat_flux,
                                   sea_ice_heat_flux,
                                   ocean_freshwater_flux,
                                   sea_ice_freshwater_flux))
    end

    return outputs
end
