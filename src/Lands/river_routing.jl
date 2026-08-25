using Oceananigans.Grids: inactive_node, λnodes, φnodes
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Fields: interior

#####
##### River routing: map river-mouth discharge onto coastal ocean cells
#####

"""
    RiverRouting

A static map from river-mouth cells on a forcing dataset's native grid to the
active (wet) cells of a target ocean grid, used to deposit river discharge as a
conservative freshwater mass flux (kg m⁻² s⁻¹).

Contributions are grouped by destination ocean cell so the scatter writes each
ocean cell exactly once (no atomics). For destination cell `c`, the contributing
river mouths are `contribution_outlet_{i,j}[offsets[c]:offsets[c+1]-1]` with
`contribution_weight = outlet_weight / Aᵒᶜᵉᵃⁿ` (see [`build_river_routing`](@ref)),
chosen so the area integral of the deposited flux equals the total mass delivered.
"""
struct RiverRouting{I, W}
    contribution_outlet_i :: I
    contribution_outlet_j :: I
    contribution_weight   :: W
    target_i :: I
    target_j :: I
    offsets  :: I
end

# A routed land carries a `NamedTuple` of `RiverRouting`, one per freshwater component
# (e.g. `(; rivers, icebergs)`), so each component scatters through its own mouth map.
const RoutedPrescribedLand = PrescribedLand{<:Any, <:Any, <:Any, <:Any, <:NamedTuple}

#####
##### Outlet (river-mouth) detection
#####

"""
    coastal_outlet_indices(discharge)

Return `(outlet_i, outlet_j, outlet_λ, outlet_φ)` for the river-mouth cells of a
`discharge` `Field` whose ocean cells are `NaN` (the GloFAS convention). A river
mouth is a finite (land/river) cell with at least one `NaN` (ocean) horizontal
neighbor — i.e. the point where the routed river network meets the coast.
"""
function coastal_outlet_indices(discharge)
    grid = discharge.grid
    arch = architecture(grid)

    outlet = Field{Center, Center, Nothing}(grid, Bool)
    fill!(outlet, false)
    launch!(arch, grid, :xy, _mark_coastal_outlets!, outlet, discharge)

    return outlet_indices_from_mask(Array(interior(outlet))[:, :, 1], grid)
end

@kernel function _mark_coastal_outlets!(outlet, discharge)
    i, j = @index(Global, NTuple)
    @inbounds begin
        finite = !isnan(discharge[i, j, 1])
        ocean_neighbor = isnan(discharge[i-1, j, 1]) | isnan(discharge[i+1, j, 1]) |
                         isnan(discharge[i, j-1, 1]) | isnan(discharge[i, j+1, 1])
        outlet[i, j, 1] = finite & ocean_neighbor
    end
end

"""
    outlet_indices_from_mask(outlet_mask, grid)

Return `(outlet_i, outlet_j, outlet_λ, outlet_φ)` for the `true` cells of a 2-D `outlet_mask` on `grid`.
"""
function outlet_indices_from_mask(outlet_mask, grid)
    indices = findall(outlet_mask)

    outlet_i = [I[1] for I in indices]
    outlet_j = [I[2] for I in indices]

    λc = Array(λnodes(grid, Center(), Center(), Center()))
    φc = Array(φnodes(grid, Center(), Center(), Center()))
    nodes = [node_λφ(λc, φc, outlet_i[n], outlet_j[n]) for n in eachindex(outlet_i)]

    return outlet_i, outlet_j, first.(nodes), last.(nodes)
end

#####
##### Building the routing map (construction-time, on CPU)
#####

"""
    build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                        maximum_search_radius = 5, spread_radius = 1.2, maximum_spread_cells = nothing)

Map each mouth at `(outlet_λ, outlet_φ)` onto the active ocean cells of `target_grid`, returning a [`RiverRouting`](@ref) that deposits
`outlet_weight[n] * value[outletₙ] / Aᵒᶜᵉᵃⁿ`. The weight is the freshwater density for a volumetric discharge (m³ s⁻¹), the source-cell
area for a per-area mass flux (kg m⁻² s⁻¹).
"""
function build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                             maximum_search_radius = 5,
                             spread_radius = 1.2,
                             maximum_spread_cells = nothing)

    arch = architecture(target_grid)
    FT = eltype(target_grid)
    kᴺ = size(target_grid, 3)

    wet_field  = Field{Center, Center, Nothing}(target_grid, Bool)
    area_field = Field{Center, Center, Nothing}(target_grid)
    launch!(arch, target_grid, :xy, _compute_wet_mask_and_area!,
            wet_field, area_field, target_grid, kᴺ)

    wet  = Array(interior(wet_field))[:, :, 1]
    area = Array(interior(area_field))[:, :, 1]

    λc = Array(λnodes(target_grid, Center(), Center(), Center()))
    φc = Array(φnodes(target_grid, Center(), Center(), Center()))

    Nx, Ny = size(wet)
    ocean_cells = wet_cells(wet, λc, φc)
    maximum_degrees = maximum_search_radius * (360 / Nx + 180 / Ny) / 2

    # Split each mouth's discharge equally over its plume footprint so no single coastal cell receives
    # a runaway freshwater flux (which drives salinity to zero and crashes the run).
    contributions = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int, FT}}}()
    dropped = 0
    for n in eachindex(outlet_i)
        targets = spread_target_cells(ocean_cells, outlet_λ[n], outlet_φ[n],
                                      maximum_degrees, spread_radius, maximum_spread_cells)
        if isempty(targets)
            dropped += 1
            continue
        end
        w = convert(FT, outlet_weight[n]) / length(targets)
        for (i★, j★) in targets
            push!(get!(contributions, (i★, j★), Tuple{Int, Int, FT}[]), (outlet_i[n], outlet_j[n], w))
        end
    end

    if dropped > 0
        @warn string(dropped, " of ", length(outlet_i), " river mouths had no active ocean ",
                     "cell in range and were dropped.")
    end

    target_i = Int[]
    target_j = Int[]
    offsets = Int[1]
    contribution_outlet_i = Int[]
    contribution_outlet_j = Int[]
    contribution_weight = FT[]

    for ((i★, j★), mouths) in contributions
        push!(target_i, i★)
        push!(target_j, j★)
        A = convert(FT, area[i★, j★])
        for (oi, oj, w) in mouths
            push!(contribution_outlet_i, oi)
            push!(contribution_outlet_j, oj)
            push!(contribution_weight, w / A)
        end
        push!(offsets, length(contribution_outlet_i) + 1)
    end

    return RiverRouting(on_architecture(arch, contribution_outlet_i),
                        on_architecture(arch, contribution_outlet_j),
                        on_architecture(arch, contribution_weight),
                        on_architecture(arch, target_i),
                        on_architecture(arch, target_j),
                        on_architecture(arch, offsets))
end

"""
    build_flux_routing(target_grid, flux_time_series;
                       maximum_search_radius = 5, spread_radius = 1.2,
                       maximum_spread_cells = nothing, outlet_detection_snapshots = 365)

Route a component stored as a per-area mass flux (kg m⁻² s⁻¹) on coastal cells — the JRA55-do convention — onto `target_grid`. Mouths are
the cells positive in any of the first `outlet_detection_snapshots` records, weighted by their source-cell area. Remaining keyword arguments
go to [`build_river_routing`](@ref).
"""
function build_flux_routing(target_grid, flux_time_series;
                            maximum_search_radius = 5,
                            spread_radius = 1.2,
                            maximum_spread_cells = nothing,
                            outlet_detection_snapshots = 365)

    source_grid = on_architecture(CPU(), flux_time_series.grid)
    kᴺ = size(source_grid, 3)

    # Scan a full seasonal cycle so intermittent and seasonally frozen rivers stay in the map.
    outlet_mask = Array(interior(flux_time_series[1]))[:, :, 1] .> 0
    for n in 2:min(outlet_detection_snapshots, length(flux_time_series.times))
        outlet_mask .|= Array(interior(flux_time_series[n]))[:, :, 1] .> 0
    end

    outlet_i, outlet_j, outlet_λ, outlet_φ = outlet_indices_from_mask(outlet_mask, source_grid)
    outlet_weight = [Azᶜᶜᶜ(outlet_i[n], outlet_j[n], kᴺ, source_grid) for n in eachindex(outlet_i)]

    return build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                               maximum_search_radius, spread_radius, maximum_spread_cells)
end

@kernel function _compute_wet_mask_and_area!(wet, area, grid, kᴺ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        wet[i, j, 1] = !inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())
        area[i, j, 1] = Azᶜᶜᶜ(i, j, kᴺ, grid)
    end
end

node_λφ(λc::AbstractVector, φc::AbstractVector, i, j) = (λc[i], φc[j])
node_λφ(λc::AbstractMatrix, φc::AbstractMatrix, i, j) = (λc[i, j], φc[i, j])

wrap180(λ) = λ - 360 * floor((λ + 180) / 360)

# Approximate squared distance on the sphere (equirectangular, degrees).
function squared_distance(λ₁, φ₁, λ₂, φ₂)
    Δλ = wrap180(λ₂ - λ₁) * cosd((φ₁ + φ₂) / 2)
    Δφ = φ₂ - φ₁
    return Δλ^2 + Δφ^2
end

function wet_cells(wet, λc, φc)
    Nx, Ny = size(wet)
    wet_i = Int[]; wet_j = Int[]
    wet_λ = Float64[]; wet_φ = Float64[]
    for j in 1:Ny, i in 1:Nx
        wet[i, j] || continue
        λ, φ = node_λφ(λc, φc, i, j)
        push!(wet_i, i); push!(wet_j, j)
        push!(wet_λ, λ); push!(wet_φ, φ)
    end
    return (i = wet_i, j = wet_j, λ = wet_λ, φ = wet_φ)
end

"""
    spread_target_cells(wet, λₒ, φₒ, maximum_degrees, spread_radius, maximum_cells)

The ocean cells a mouth at `(λₒ, φₒ)` discharges into: the wet cells within `spread_radius` degrees of its
landing cell, nearest first, capped at `maximum_cells`. `spread_radius = nothing` spreads over the whole
`maximum_degrees` reach. Empty when no wet cell lies within `maximum_degrees`.
"""
function spread_target_cells(wet, λₒ, φₒ, maximum_degrees, spread_radius, maximum_cells)
    reach = maximum_degrees^2
    nearest = 0
    nearest_distance = Inf
    reachable = Int[]
    for n in eachindex(wet.i)
        # Cells further than `maximum_degrees` in latitude alone are out of reach, so skip the metric.
        abs(wet.φ[n] - φₒ) < maximum_degrees || continue
        d = squared_distance(λₒ, φₒ, wet.λ[n], wet.φ[n])
        d < nearest_distance && (nearest_distance = d; nearest = n)
        d < reach && push!(reachable, n)
    end
    nearest_distance < reach || return Tuple{Int, Int}[]

    # The footprint is centered on the landing cell, so mouths relocated onto the shelf (the Ob and
    # Yenisei move 2-3°) keep a full footprint.
    λ★, φ★ = wet.λ[nearest], wet.φ[nearest]
    footprint = isnothing(spread_radius) ? reach : spread_radius^2
    targets = [(squared_distance(λ★, φ★, wet.λ[n], wet.φ[n]), n) for n in reachable]
    filter!(t -> first(t) ≤ footprint, targets)
    sort!(targets; by = first)

    number_of_targets = isnothing(maximum_cells) ? length(targets) : min(maximum_cells, length(targets))
    return [(wet.i[targets[m][2]], wet.j[targets[m][2]]) for m in 1:number_of_targets]
end

#####
##### Conservative scatter of river discharge onto the ocean grid
#####

"""Scatter each prescribed freshwater component onto coastal ocean cells, conserving volume."""
function EarthSystemModels.interpolate_state!(exchanger, grid, land::RoutedPrescribedLand, coupled_model)
    arch = architecture(grid)
    land_freshwater_flux = exchanger.state.freshwater_flux
    time = Time(coupled_model.clock.time)

    fill!(land_freshwater_flux, 0)

    for name in keys(land.freshwater_flux)
        scatter_freshwater_flux!(land_freshwater_flux, land.freshwater_flux[name],
                                 land.river_routing[name], arch, grid, time)
    end

    return nothing
end

function scatter_freshwater_flux!(land_freshwater_flux, discharge, routing, arch, grid, time)
    n_targets = length(routing.target_i)
    n_targets == 0 && return nothing

    launch!(arch, grid, (n_targets,),
            _scatter_river_discharge!,
            land_freshwater_flux.data,
            discharge,
            time,
            routing.contribution_outlet_i,
            routing.contribution_outlet_j,
            routing.contribution_weight,
            routing.target_i,
            routing.target_j,
            routing.offsets)

    return nothing
end

# One thread per destination ocean cell sums all mouths routed to it (written exactly
# once within a launch); components accumulate across launches, so the write is `+=`.
@kernel function _scatter_river_discharge!(flux, discharge, time,
                                           contribution_outlet_i,
                                           contribution_outlet_j,
                                           contribution_weight,
                                           target_i, target_j, offsets)
    c = @index(Global)
    @inbounds begin
        accumulated = zero(eltype(flux))
        for k in offsets[c]:(offsets[c+1] - 1)
            iₒ = contribution_outlet_i[k]
            jₒ = contribution_outlet_j[k]
            Q = discharge[iₒ, jₒ, 1, time]   # temporal interpolation at the exact mouth cell
            Q = ifelse(isnan(Q), zero(Q), Q)
            accumulated += contribution_weight[k] * Q
        end
        flux[target_i[c], target_j[c], 1] += accumulated
    end
end
