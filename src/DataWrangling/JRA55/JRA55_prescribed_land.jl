using ...Lands: PrescribedLand, build_river_routing, ever_positive_mask, outlet_indices_from_mask,
                source_cell_areas

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

Return a [`PrescribedLand`](@ref) representing JRA55 reanalysis land surface data
(river runoff and iceberg calving freshwater fluxes), routed onto the coastline of
`grid` (the target ocean grid).

JRA55-do provides these as per-area mass fluxes (kg m⁻² s⁻¹) on coastal cells of the
forcing grid. Every cell that discharges at any point over the first
`outlet_detection_snapshots` records is treated as a river mouth and spread over the
active ocean cells of `grid` around it, depositing a volume-conserving mass flux (see
[`build_river_routing`](@ref)).

Keyword Arguments
=================
- `maximum_search_radius`: maximum distance (in `grid` cells) to search for an active
  ocean cell when placing a river mouth. Default: `5`.
- `spread_radius`: radius (in degrees) of the plume footprint over which each mouth's
  discharge is divided equally. Default: `1.2`.
- `maximum_spread_cells`: cap on the number of cells in that footprint, nearest first.
  Default: `nothing` (no cap).
- `outlet_detection_snapshots`: number of leading snapshots scanned for discharging
  cells, so that intermittent and seasonally frozen rivers are not missed. Default: `365`.

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
    river_routing = map(fts -> build_flux_routing(grid, fts; maximum_search_radius, spread_radius, maximum_spread_cells, outlet_detection_snapshots), 
                        freshwater_flux)

    return PrescribedLand(freshwater_flux; river_routing)
end

function build_flux_routing(grid, flux_time_series; maximum_search_radius = 5, spread_radius = 1.2,
                            maximum_spread_cells = nothing, outlet_detection_snapshots = 365)

    outlet_mask = ever_positive_mask(flux_time_series, outlet_detection_snapshots)
    outlet_i, outlet_j, outlet_λ, outlet_φ = outlet_indices_from_mask(outlet_mask, flux_time_series.grid)
    outlet_weight = source_cell_areas(flux_time_series.grid, outlet_i, outlet_j)

    return build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                               maximum_search_radius, spread_radius, maximum_spread_cells)
end
