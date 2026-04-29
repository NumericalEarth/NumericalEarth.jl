"""
    omip_atmosphere(arch; forcing_dir, start_date, end_date, repeat_year_forcing=false, backend_size=30)

Set up a JRA55 prescribed atmosphere with river and iceberg forcing,
together with a default `Radiation` model. Returns the tuple
`(atmosphere, radiation)`.
"""
function omip_atmosphere(arch;
                         forcing_dir,
                         start_date,
                         end_date,
                         repeat_year_forcing = false,
                         backend_size = 30)

    dataset = repeat_year_forcing ? RepeatYearJRA55() : MultiYearJRA55()

    atmosphere = JRA55PrescribedAtmosphere(arch;
                                           dir = forcing_dir,
                                           dataset,
                                           time_indices_in_memory = backend_size,
                                           include_rivers_and_icebergs = true,
                                           prefetch = true,
                                           start_date,
                                           end_date)

    radiation = Radiation()

    return atmosphere, radiation
end
