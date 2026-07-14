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
                         other_kw...)

Return a [`PrescribedLand`](@ref) that forces the ocean with GloFAS river
discharge routed onto the coastline of `grid` (the target ocean grid).

GloFAS provides river discharge (m³ s⁻¹) already accumulated downstream to river
mouths by the LISFLOOD channel-routing model, driven by ERA5 runoff. The mouths
are located from the dataset's land/ocean boundary and mapped to the nearest
active ocean cells of `grid`; the discharge is then deposited as a freshwater
mass flux that conserves volume (see [`build_river_routing`](@ref)).

Keyword Arguments
=================
- `freshwater_density`: density used to convert discharge (m³ s⁻¹) to a mass flux
  (kg m⁻² s⁻¹). Default: `1000`.
- `maximum_search_radius`: maximum distance (in `grid` cells) to search for an
  active ocean cell when placing a river mouth. Default: `5`.

See also [`JRA55PrescribedLand`](@ref) for the pre-routed JRA55 alternative.
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

    routing = build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ;
                                  freshwater_density, maximum_search_radius)

    freshwater_flux = (; rivers = discharge)

    return PrescribedLand(freshwater_flux; river_routing = routing)
end
