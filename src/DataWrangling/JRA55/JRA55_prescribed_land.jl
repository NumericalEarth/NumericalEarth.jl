using ...Lands: PrescribedLand, build_flux_routing

"""
    JRA55PrescribedLand(grid;
                        dataset = RepeatYearJRA55(),
                        start_date = first_date(dataset, :river_freshwater_flux),
                        end_date = last_date(dataset, :river_freshwater_flux),
                        dir = download_JRA55_cache,
                        time_indices_in_memory = 10,
                        time_indexing = Cyclical(),
                        region = nothing,
                        maximum_search_radius = 5,
                        spread_radius = 1.2,
                        maximum_spread_cells = nothing,
                        outlet_detection_snapshots = 365,
                        other_kw...)

Return a [`PrescribedLand`](@ref) holding the JRA55-do river runoff and iceberg calving fluxes, routed
onto the coastline of the target ocean `grid` by [`build_flux_routing`](@ref).

Keyword Arguments
=================
- `maximum_search_radius`: search distance in `grid` cells for the ocean cell receiving a mouth. Default: `5`.
- `spread_radius`: radius in degrees over which each mouth's discharge is divided equally. Default: `1.2`.
- `maximum_spread_cells`: cap on that footprint, nearest first. Default: `nothing` (uncapped).
- `outlet_detection_snapshots`: leading snapshots scanned for discharging cells, so that intermittent and seasonally frozen rivers are not missed. Default: `365`.

See also [`GloFASPrescribedLand`](@ref).
"""
function JRA55PrescribedLand(grid;
                             dataset = RepeatYearJRA55(),
                             start_date = first_date(dataset, :river_freshwater_flux),
                             end_date = last_date(dataset, :river_freshwater_flux),
                             dir = download_JRA55_cache,
                             time_indices_in_memory = 10,
                             time_indexing = Cyclical(),
                             region = nothing,
                             maximum_search_radius = 5,
                             spread_radius = 1.2,
                             maximum_spread_cells = nothing,
                             outlet_detection_snapshots = 365,
                             other_kw...)

    arch = child_architecture(grid)
    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    JRA55FieldTimeSeries(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), arch; kw...)

    Fri = JRA55FieldTimeSeries(:river_freshwater_flux)
    Fic = JRA55FieldTimeSeries(:iceberg_freshwater_flux)

    freshwater_flux = (; rivers = Fri, icebergs = Fic)
    river_routing = map(fts -> build_flux_routing(grid, fts; maximum_search_radius, spread_radius,
                                                  maximum_spread_cells, outlet_detection_snapshots),
                        freshwater_flux)

    return PrescribedLand(freshwater_flux; river_routing)
end
