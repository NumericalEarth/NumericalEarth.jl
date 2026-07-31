using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field

"""
    GloFASPrescribedLand(grid;
                         dataset = GloFASReanalysis(),
                         start_date = first_date(dataset, :river_discharge),
                         end_date = last_date(dataset, :river_discharge),
                         dir = download_GloFAS_cache,
                         time_indices_in_memory = 10,
                         time_indexing = Cyclical(),
                         region = nothing,
                         freshwater_density = 1000,
                         maximum_search_radius = 5,
                         spread_radius = 1.2,
                         maximum_spread_cells = 8,
                         other_kw...)

Return a [`PrescribedLand`](@ref) that forces the ocean with GloFAS river
discharge routed onto the coastline of `grid` (the target ocean grid).

GloFAS provides river discharge (m³ s⁻¹) already accumulated downstream to river
mouths by the LISFLOOD channel-routing model, driven by ERA5 runoff. The mouths
are located from the dataset's land/ocean boundary and mapped to the active ocean
cells of `grid` surrounding each mouth; the discharge is then deposited as a
freshwater mass flux that conserves volume (see [`build_river_routing`](@ref)).

Keyword Arguments
=================
- `freshwater_density`: density converting the discharge (m³ s⁻¹) to a mass flux (kg m⁻² s⁻¹). Default: `1000`.
- `maximum_search_radius`: search distance in `grid` cells for the ocean cell receiving a mouth. Default: `5`.
- `spread_radius`: radius in degrees over which each mouth's discharge is divided equally. Default: `1.2`.
- `maximum_spread_cells`: cap on that footprint, nearest first. Default: `8`.

See also [`JRA55PrescribedLand`](@ref).
"""
function GloFASPrescribedLand(grid;
                              dataset = GloFASReanalysis(),
                              start_date = first_date(dataset, :river_discharge),
                              end_date = last_date(dataset, :river_discharge),
                              dir = download_GloFAS_cache,
                              time_indices_in_memory = 10,
                              time_indexing = Cyclical(),
                              region = nothing,
                              freshwater_density = 1000,
                              maximum_search_radius = 5,
                              spread_radius = 1.2,
                              maximum_spread_cells = 8,
                              other_kw...)

    arch = architecture(grid)
    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    discharge_meta = Metadata(:river_discharge; dataset, start_date, end_date, dir, region)
    discharge = FieldTimeSeries(discharge_meta, arch; kw...)

    # River mouths are located from the land/ocean boundary of the first snapshot
    # (ocean cells are NaN), then mapped to coastal cells of the target grid.
    snapshot = Field(first(discharge_meta), arch)
    outlet_i, outlet_j, outlet_λ, outlet_φ = coastal_outlet_indices(snapshot)

    outlet_weight = fill(convert(eltype(grid), freshwater_density), length(outlet_i))
    routing = build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                                  maximum_search_radius, spread_radius, maximum_spread_cells)

    freshwater_flux = (; rivers = discharge)

    return PrescribedLand(freshwater_flux; river_routing = (; rivers = routing))
end
