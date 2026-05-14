using NumericalEarth.Lands: PrescribedLand

export JRA55PrescribedLand

JRA55PrescribedLand(arch::Distributed, FT=Float32; kw...) =
    JRA55PrescribedLand(child_architecture(arch), FT; kw...)

"""
    JRA55PrescribedLand([architecture = CPU(), FT = Float32];
                        dataset = RepeatYearJRA55(),
                        start_date = first_date(dataset, :river_freshwater_flux),
                        end_date = last_date(dataset, :river_freshwater_flux),
                        backend = JRA55NetCDFBackend(10),
                        time_indexing = Cyclical(),
                        other_kw...)

Return a [`PrescribedLand`](@ref) representing JRA55 reanalysis land surface data
(river runoff and iceberg calving freshwater fluxes).
"""
function JRA55PrescribedLand(architecture=CPU(), FT=Float32;
                             dataset = RepeatYearJRA55(),
                             start_date = first_date(dataset, :river_freshwater_flux),
                             end_date = last_date(dataset, :river_freshwater_flux),
                             backend = JRA55NetCDFBackend(10),
                             time_indexing = Cyclical(),
                             other_kw...)

    kw = (; time_indexing, backend, start_date, end_date, dataset)
    kw = merge(kw, other_kw)

    Fri = JRA55FieldTimeSeries(:river_freshwater_flux, architecture, FT;   kw...)
    Fic = JRA55FieldTimeSeries(:iceberg_freshwater_flux, architecture, FT; kw...)

    freshwater_flux = (; rivers = Fri, icebergs = Fic)

    return PrescribedLand(freshwater_flux)
end
