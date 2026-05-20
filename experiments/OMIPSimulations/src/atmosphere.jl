"""
    omip_atmosphere(arch, sea_ice; forcing_dir, start_date, end_date, backend_size=30)

Build a JRA55-forced prescribed atmosphere and a matching
`JRA55PrescribedRadiation`. Ocean surface uses the OMIP-2 standard values
(emissivity = 1.0, albedo = 0.06); sea-ice surface uses the CCSM3
temperature / snow / thickness-dependent `SeaIceAlbedo` evaluated against
the live sea-ice surface temperature (snow surface if present, ice top
otherwise) with emissivity = 1.0.

Returns the tuple `(atmosphere, radiation)`.
"""
function omip_atmosphere(arch, sea_ice;
                         forcing_dir,
                         start_date,
                         end_date,
                         repeat_year_forcing = false,
                         backend_size = 30)

    dataset = repeat_year_forcing ? RepeatYearJRA55() : MultiYearJRA55()

    kw = (; dir = forcing_dir,
            dataset,
            start_date,
            end_date,
            time_indices_in_memory = backend_size,
            prefetch = true)

    atmosphere = JRA55PrescribedAtmosphere(arch; kw...)

    # CCSM3 sea-ice albedo reads live model fields, so the surface
    # temperature must come from whichever layer the atmosphere actually
    # sees: snow top if a snow model is present, ice top otherwise.
    hi = sea_ice.model.ice_thickness
    hs = sea_ice.model.snow_thickness
    snow_thermo = sea_ice.model.snow_thermodynamics
    Ts = isnothing(snow_thermo) ? sea_ice.model.ice_thermodynamics.top_surface_temperature :
                                  snow_thermo.top_surface_temperature
    sea_ice_albedo = SeaIceAlbedo(hi, hs, Ts)

    radiation = JRA55PrescribedRadiation(arch;
                                         kw...,
                                         ocean_surface   = SurfaceRadiationProperties(0.06, 1.00),
                                         sea_ice_surface = SurfaceRadiationProperties(sea_ice_albedo, 1.0))

    return atmosphere, radiation
end
