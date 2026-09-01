using Oceananigans.Architectures: architecture
using ...Lands: PrescribedLand, ever_positive_mask, outlet_indices_from_mask, source_cell_areas,
                build_river_routing

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
                        n_spread_cells = nothing,
                        n_outlet_snapshots = 365,
                        flux_diversion = nothing,
                        other_kw...)

Return a [`PrescribedLand`](@ref) representing JRA55 reanalysis land surface data
(river runoff and iceberg calving freshwater fluxes), routed onto the coastline of
`grid` (the target ocean grid).

JRA55-do provides these as per-area mass fluxes (kg m⁻² s⁻¹) on coastal cells of the
forcing grid. Each nonzero cell is treated as a river mouth and mapped to the nearest
active ocean cell of `grid`, depositing a volume-conserving mass flux (see
[`build_river_routing`](@ref)). See also [`GloFASPrescribedLand`](@ref).

`flux_diversion` sends a fraction of the discharge that would land in one basin to another instead:
a NamedTuple `(; fraction, from, to)` whose masks are `Nx × Ny` Boolean arrays on `grid`. The
re-targeting is per mouth, so the global freshwater input is conserved at every time step.
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
                             n_spread_cells = nothing,
                             n_outlet_snapshots = 365,
                             flux_diversion = nothing,
                             other_kw...)

    arch = architecture(grid)
    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    JRA55FieldTimeSeries(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), arch; kw...)

    Fri = JRA55FieldTimeSeries(:river_freshwater_flux)
    Fic = JRA55FieldTimeSeries(:iceberg_freshwater_flux)

    freshwater_flux = (; rivers = Fri, icebergs = Fic)
    river_routing = map(fts -> build_flux_routing(grid, fts; maximum_search_radius, spread_radius, n_spread_cells,
                                                        n_outlet_snapshots, flux_diversion), freshwater_flux)

    return PrescribedLand(freshwater_flux; river_routing)
end

# Route a per-area mass-flux component: cells that discharge at any point over the first
# `n_outlet_snapshots` are mouths, weighted by their source-cell area so the mass delivered to the
# ocean equals ∫ flux dA at the source.
function build_flux_routing(grid, flux_fts; maximum_search_radius = 5, spread_radius = 1.2,
                            n_spread_cells = nothing, n_outlet_snapshots = 365, flux_diversion = nothing)
    outlet_mask = ever_positive_mask(flux_fts, n_outlet_snapshots)
    outlet_i, outlet_j, outlet_λ, outlet_φ = outlet_indices_from_mask(outlet_mask, flux_fts.grid)
    outlet_weight = source_cell_areas(flux_fts.grid, outlet_i, outlet_j)
    return build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                               maximum_search_radius, spread_radius, n_spread_cells, flux_diversion)
end
