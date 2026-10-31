#####
##### Prescribed land and soil properties built from the JRA55, GloFAS, and OpenLandMap datasets
#####

using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.Fields: set!
using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.OutputReaders: Cyclical, FieldTimeSeries

using ..DataWrangling: Metadata, Metadatum, BoundingBox, first_date, last_date,
                       default_download_directory, inpaint_mask!
using ..DataWrangling.JRA55: RepeatYearJRA55, download_JRA55_cache
using ..DataWrangling.GloFAS: GloFASReanalysis, download_GloFAS_cache
using ..DataWrangling.OpenLandMap: OpenLandMapSoilDB

#####
##### JRA55 and GloFAS river runoff
#####

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
                        other_kw...)

Return a [`PrescribedLand`](@ref) holding the JRA55-do river runoff and iceberg calving fluxes, routed
onto the coastline of the target ocean `grid` by [`build_flux_routing`](@ref).

Keyword Arguments
=================
- `maximum_search_radius`: search distance in `grid` cells for the ocean cell receiving a mouth. Default: `5`.
- `spread_radius`: radius in degrees over which each mouth's discharge is divided equally. Default: `1.2`.
- `maximum_spread_cells`: cap on that footprint, nearest first. Default: `nothing` (uncapped).

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
                             other_kw...)

    arch = child_architecture(grid)
    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    JRA55FieldTimeSeries(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), arch; kw...)

    Fri = JRA55FieldTimeSeries(:river_freshwater_flux)
    Fic = JRA55FieldTimeSeries(:iceberg_freshwater_flux)

    freshwater_flux = (; rivers = Fri, icebergs = Fic)
    river_routing = routable_grid(grid) ?
        map(fts -> build_flux_routing(grid, fts; maximum_search_radius, spread_radius, maximum_spread_cells),
            freshwater_flux) : nothing

    return PrescribedLand(freshwater_flux; river_routing)
end

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
    river_routing = routable_grid(grid) ?
        (; rivers = build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                                        maximum_search_radius, spread_radius, maximum_spread_cells)) : nothing

    freshwater_flux = (; rivers = discharge)

    return PrescribedLand(freshwater_flux; river_routing)
end

#####
##### Hydraulic parameters straight from the dataset
#####

"""
$(TYPEDSIGNATURES)

Effective van Genuchten parameter fields for `grid` computed from `dataset`: the four
texture variables (`:sand_fraction`, `:silt_fraction`, `:clay_fraction`, `:bulk_density`)
are read over `region` onto a lattice with `grid`'s horizontal cells and the dataset's
depth layers, their gaps are inpainted, and the per-layer pedotransfer reduction runs
over `slab_depth`. Returns a NamedTuple of `porosity`, `residual_liquid_fraction`,
`inverse_air_entry_head`, `pore_size_uniformity`, `matching_point_conductivity`, and
`pore_connectivity_exponent`, each a `Field{Center, Center, Nothing}` on `grid`.
`dir` is the download directory; remaining
keyword arguments (`ptf`, `matching_heads`) pass to the reduction.
"""
function soil_hydraulic_properties(grid, dataset::OpenLandMapSoilDB;
                                         slab_depth,
                                         region = BoundingBox(grid),
                                         dir = default_download_directory(dataset),
                                         kw...)
    z = DataWrangling.z_interfaces(dataset)
    Nx, Ny, _ = size(grid)
    lattice = LatitudeLongitudeGrid(architecture(grid), eltype(grid);
                                    size = (Nx, Ny, length(z) - 1),
                                    longitude = λnodes(grid, Face(), Center(), Center()),
                                    latitude = φnodes(grid, Center(), Face(), Center()),
                                    z,
                                    topology = (Bounded, Bounded, Bounded))

    texture = map((:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)) do name
        field = Field(Metadatum(name; dataset, region, dir), lattice)
        gaps  = Field{Center, Center, Center}(lattice, Bool)
        interior(gaps) .= .!isfinite.(interior(field))
        inpaint_mask!(field, gaps)
        return field
    end

    layered = soil_hydraulic_properties(texture...; slab_depth, kw...)

    surface = (porosity = Field{Center, Center, Nothing}(grid),
               residual_liquid_fraction = Field{Center, Center, Nothing}(grid),
               inverse_air_entry_head = Field{Center, Center, Nothing}(grid),
               pore_size_uniformity = Field{Center, Center, Nothing}(grid),
               matching_point_conductivity = Field{Center, Center, Nothing}(grid),
               pore_connectivity_exponent = Field{Center, Center, Nothing}(grid))
    set!(surface, layered)
    return surface
end
