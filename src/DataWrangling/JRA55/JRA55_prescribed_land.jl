using NumericalEarth.Lands: PrescribedLand

export JRA55PrescribedLand

JRA55PrescribedLand(arch::Distributed; kw...) =
    JRA55PrescribedLand(child_architecture(arch); kw...)

"""
    JRA55PrescribedLand([architecture = CPU()];
                        dataset = RepeatYearJRA55(),
                        start_date = first_date(dataset, :river_freshwater_flux),
                        end_date = last_date(dataset, :river_freshwater_flux),
                        dir = download_JRA55_cache,
                        time_indices_in_memory = 10,
                        time_indexing = Cyclical(),
                        region = nothing,
                        other_kw...)

Return a [`PrescribedLand`](@ref) representing JRA55 reanalysis land surface data
(river runoff and iceberg calving freshwater fluxes). 
"""
function JRA55PrescribedLand(architecture = CPU();
                             dataset = RepeatYearJRA55(),
                             start_date = first_date(dataset, :river_freshwater_flux),
                             end_date = last_date(dataset, :river_freshwater_flux),
                             dir = download_JRA55_cache,
                             time_indices_in_memory = 10,
                             time_indexing = Cyclical(),
                             region = nothing,
                             other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    JRA55FieldTimeSeries(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), architecture; kw...)

    Fri = JRA55FieldTimeSeries(:river_freshwater_flux)
    Fic = JRA55FieldTimeSeries(:iceberg_freshwater_flux)

    freshwater_flux = (; rivers = Fri, icebergs = Fic)

    return PrescribedLand(freshwater_flux)
end
