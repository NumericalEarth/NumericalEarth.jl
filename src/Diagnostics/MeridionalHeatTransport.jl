"""
    Meridional_Heat_Transport(coupled_model)

Return the cumulative meridional integral of the zonal-depth integrated
temperature minus the zonally integrated interface heat-flux tendency.
"""
function Meridional_Heat_Transport(coupled_model)
    ocean = coupled_model.ocean

    T_int = Integral(ocean.model.tracers.T, dims=(1, 3))

    flux_outputs = InterfaceFluxOutputs(coupled_model;
                                        isolate_sea_ice=false,
                                        units=:physical,
                                        reference_salinity=35)

    heat_flux = haskey(flux_outputs, "heat_flux") ? flux_outputs["heat_flux"] : flux_outputs[:heat_flux]
    flux_int = Integral(heat_flux * ocean.model.clock.Δt, dims=(1))

    return CumulativeIntegral(T_int - flux_int, dims=(2))
end
