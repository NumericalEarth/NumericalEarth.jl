using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: AbstractGrid
using NumericalEarth.Lands: PrescribedLand, ever_positive_mask, outlet_indices_from_mask,
                            source_cell_areas, build_river_routing

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

"""
    $(TYPEDSIGNATURES)

Build JRA55 river and iceberg forcing routed onto wet cells of `grid`.
`spread_radius` sets the radius in degrees around each receiving cell. Discharge
is shared in proportion to column depth, capped at 50 m, and converted using
source and receiving cell areas to conserve mass. Unreachable mouths are reported.
The first `n_outlet_snapshots` identify cells that discharge during the year.
Pass forcing date selections and cache options through `kw`.
"""
function JRA55PrescribedLand(grid::AbstractGrid;
                             maximum_search_radius = 5,
                             spread_radius = 1.2,
                             n_spread_cells = nothing,
                             n_outlet_snapshots = 365,
                             kw...)
    land = JRA55PrescribedLand(architecture(grid); kw...)
    routing = map(land.freshwater_flux) do flux
        outlet_mask = ever_positive_mask(flux, n_outlet_snapshots)
        outlet_i, outlet_j, outlet_λ, outlet_φ = outlet_indices_from_mask(outlet_mask, flux.grid)
        outlet_weight = source_cell_areas(flux.grid, outlet_i, outlet_j)
        build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                           maximum_search_radius, spread_radius, n_spread_cells)
    end
    return PrescribedLand(land.freshwater_flux; river_routing=routing)
end
