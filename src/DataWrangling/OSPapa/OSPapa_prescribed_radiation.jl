using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties, default_stefan_boltzmann_constant

"""
    OSPapaPrescribedRadiation(architecture = CPU(), FT = Float32;
                              start_date = first_date(OSPapaHourly(), :shortwave_radiation),
                              end_date   = last_date(OSPapaHourly(), :shortwave_radiation),
                              dir = download_OSPapa_cache,
                              max_gap_hours = 72,
                              ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                              sea_ice_surface = nothing,
                              stefan_boltzmann_constant = default_stefan_boltzmann_constant)

Construct a `PrescribedRadiation` from Ocean Station Papa buoy SW/LW
observations on a single-column grid.
"""
function OSPapaPrescribedRadiation(architecture = CPU(), FT = Float32;
                                   start_date = first_date(OSPapaHourly(), :shortwave_radiation),
                                   end_date   = last_date(OSPapaHourly(), :shortwave_radiation),
                                   dir = download_OSPapa_cache,
                                   max_gap_hours = 72,
                                   ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                                   sea_ice_surface = nothing,
                                   stefan_boltzmann_constant = default_stefan_boltzmann_constant)

    mdkw = (; dataset = OSPapaHourly(), start_date, end_date, dir)
    surface_grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))

    function ospapa_fts(name)
        md = Metadata(name; mdkw...)
        download_dataset(md)
        fts = FieldTimeSeries(md, surface_grid; time_indices_in_memory = length(md))
        fill_gaps!(fts; max_gap = max_gap_hours)
        return fts
    end

    swa = ospapa_fts(:shortwave_radiation)
    lwa = ospapa_fts(:longwave_radiation)

    return PrescribedRadiation(swa, lwa;
                               ocean_surface,
                               sea_ice_surface,
                               stefan_boltzmann_constant)
end
