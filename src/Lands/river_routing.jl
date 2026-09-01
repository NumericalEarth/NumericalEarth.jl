using Oceananigans.Grids: inactive_node, λnodes, φnodes
using Oceananigans.Operators: Azᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: interior

#####
##### River routing: map river-mouth discharge onto coastal ocean cells
#####

"""
    RiverRouting

A static map from river-mouth cells on a forcing dataset's native grid to the
active (wet) cells of a target ocean grid, used to deposit volumetric river
discharge (m³ s⁻¹) as a conservative freshwater mass flux (kg m⁻² s⁻¹).

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

    outlet_mask = Array(interior(outlet))[:, :, 1]
    indices = findall(outlet_mask)

    outlet_i = [I[1] for I in indices]
    outlet_j = [I[2] for I in indices]

    λc = Array(λnodes(grid, Center(), Center(), Center()))
    φc = Array(φnodes(grid, Center(), Center(), Center()))
    outlet_λ = [λc[i] for i in outlet_i]
    outlet_φ = [φc[j] for j in outlet_j]

    return outlet_i, outlet_j, outlet_λ, outlet_φ
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
    positive_outlet_indices(flux)

Return `(outlet_i, outlet_j, outlet_λ, outlet_φ)` for the coastal runoff cells of a
per-area freshwater `flux` `Field` (the JRA55 convention: runoff is a positive mass
flux at coastal cells and zero elsewhere). Every strictly positive cell is a mouth.
"""
positive_outlet_indices(flux) = outlet_indices_from_mask(Array(interior(flux))[:, :, 1] .> 0, flux.grid)

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
    outlet_λ = [λc[i] for i in outlet_i]
    outlet_φ = [φc[j] for j in outlet_j]

    return outlet_i, outlet_j, outlet_λ, outlet_φ
end

"""
    ever_positive_mask(flux_fts, n_snapshots)

Cells of `flux_fts` positive in *any* of its first `n_snapshots`. A cell dry in the first snapshot but
discharging later — an intermittent river, or a high-latitude one frozen at the start date — would
otherwise never enter the routing map and its water would be dropped for the whole run, so the mask is
taken over a full seasonal cycle rather than a single time.
"""
function ever_positive_mask(flux_fts, n_snapshots)
    mask = Array(interior(flux_fts[1]))[:, :, 1] .> 0
    for n in 2:min(n_snapshots, length(flux_fts.times))
        mask .|= Array(interior(flux_fts[n]))[:, :, 1] .> 0
    end
    return mask
end

"""
    source_cell_areas(grid, outlet_i, outlet_j)

Horizontal areas (m²) of the `grid` cells at the given outlet indices — the
per-mouth `outlet_weight` for routing a per-area mass flux (kg m⁻² s⁻¹).
"""
function source_cell_areas(grid, outlet_i, outlet_j)
    arch = architecture(grid)
    area_field = Field{Center, Center, Nothing}(grid)
    launch!(arch, grid, :xy, _compute_source_area!, area_field, grid, size(grid, 3))
    area = Array(interior(area_field))[:, :, 1]
    return [area[outlet_i[n], outlet_j[n]] for n in eachindex(outlet_i)]
end

@kernel function _compute_source_area!(area, grid, kᴺ)
    i, j = @index(Global, NTuple)
    @inbounds area[i, j, 1] = Azᶜᶜᶜ(i, j, kᴺ, grid)
end

#####
##### Building the routing map (construction-time, on CPU)
#####

"""
    build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                        maximum_search_radius = 5, spread_radius = 1.2, n_spread_cells = nothing)

Map each river mouth at `(outlet_λ, outlet_φ)` onto the active ocean cells of `target_grid` within `maximum_search_radius`
cells, returning a [`RiverRouting`](@ref). River mouths with no active ocean cell in range are dropped (and reported), so the
global freshwater budget is conserved up to the dropped discharge.

Discharge is divided over every wet cell within `spread_radius` degrees of the mouth's landing cell, capped at
`n_spread_cells` (nearest first) when that is not `nothing`. The footprint is set by a geographic radius rather than a cell
count so the freshwater flux per unit area does not grow as the grid refines. Within the footprint, each cell's share is
proportional to its water-column depth capped at `maximum_weighting_depth` (default 50 m): a thin estuary column holds
less volume to buffer the same per-area dilution, so weighting by depth keeps shallow coastal cells from being freshened
to zero while deeper shelf cells absorb the bulk of the discharge. The shares are normalized per mouth, so the total
delivered mass is unchanged.

`outlet_weight[n]` is the per-mouth factor that converts the outlet's stored value into a mass discharge (kg s⁻¹):
the deposited flux is `outlet_weight[n] * value[outlet_n] / Aᵒᶜᵉᵃⁿ`. For a volumetric discharge (m³ s⁻¹) it is the
freshwater density; for a per-area mass flux (kg m⁻² s⁻¹) it is the source-cell area. Both conserve the total mass delivered.
"""
function build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                             maximum_search_radius = 5,
                             spread_radius = 1.2,
                             n_spread_cells = nothing,
                             maximum_weighting_depth = 50,
                             flux_diversion = nothing)

    arch = architecture(target_grid)
    FT = eltype(target_grid)
    kᴺ = size(target_grid, 3)

    wet_field   = Field{Center, Center, Nothing}(target_grid, Bool)
    area_field  = Field{Center, Center, Nothing}(target_grid)
    depth_field = Field{Center, Center, Nothing}(target_grid)
    launch!(arch, target_grid, :xy, _compute_wet_mask_area_and_depth!,
            wet_field, area_field, depth_field, target_grid, kᴺ)

    wet   = Array(interior(wet_field))[:, :, 1]
    area  = Array(interior(area_field))[:, :, 1]
    depth = Array(interior(depth_field))[:, :, 1]

    λc = Array(λnodes(target_grid, Center(), Center(), Center()))
    φc = Array(φnodes(target_grid, Center(), Center(), Center()))

    Nx, Ny = size(wet)
    wet_i, wet_j, wet_λ, wet_φ = wet_cells(wet, λc, φc)
    max_degrees = maximum_search_radius * (360 / Nx + 180 / Ny) / 2

    # Split each mouth's discharge over its plume footprint, each cell weighted by its column depth
    # (capped at `maximum_weighting_depth`), so no single coastal cell — and in particular no thin
    # estuary cell — receives a runaway freshwater flux that drives its salinity to zero.
    diverting = !isnothing(flux_diversion) && flux_diversion.fraction > 0
    if diverting
        receiver_i, receiver_j, receiver_λ, receiver_φ = masked_wet_cells(flux_diversion.to, wet, λc, φc)
        isempty(receiver_i) && error("The diversion destination mask holds no wet cell of the target grid.")
        # Mouths at similar latitudes relocate to similar places, so a diverted footprint the size of a
        # river's own would stack several mouths onto the same cells. A wider one keeps the flux per
        # unit area at or below what the receiving basin's own rivers already deliver.
        n_receivers = get(flux_diversion, :spread_cells, 8 * something(n_spread_cells, 8))
    end

    contributions = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int, FT}}}()
    dropped = 0
    for n in eachindex(outlet_i)
        targets = spread_target_cells(wet_i, wet_j, wet_λ, wet_φ, outlet_λ[n], outlet_φ[n],
                                      max_degrees, spread_radius, n_spread_cells)
        if isempty(targets)
            dropped += 1
            continue
        end
        shares = [min(depth[i★, j★], maximum_weighting_depth) for (i★, j★) in targets]
        total_share = sum(shares)
        total_share > 0 || (shares = ones(FT, length(targets)); total_share = length(targets))
        diverted = zero(FT)
        for (m, (i★, j★)) in enumerate(targets)
            w = convert(FT, outlet_weight[n] * shares[m] / total_share)
            if diverting && flux_diversion.from[i★, j★]
                diverted += w * convert(FT, flux_diversion.fraction)
                w *= convert(FT, 1 - flux_diversion.fraction)
            end
            w > 0 && push!(get!(contributions, (i★, j★), Tuple{Int, Int, FT}[]), (outlet_i[n], outlet_j[n], w))
        end
        diverted > 0 || continue
        receivers = diversion_target_cells(receiver_i, receiver_j, receiver_λ, receiver_φ,
                                           outlet_λ[n], outlet_φ[n],
                                           max_degrees, spread_radius, n_receivers)
        receiver_shares = [min(depth[i★, j★], maximum_weighting_depth) for (i★, j★) in receivers]
        total_receiver_share = sum(receiver_shares)
        total_receiver_share > 0 ||
            (receiver_shares = ones(FT, length(receivers)); total_receiver_share = length(receivers))
        for (m, (i★, j★)) in enumerate(receivers)
            w = convert(FT, diverted * receiver_shares[m] / total_receiver_share)
            w > 0 && push!(get!(contributions, (i★, j★), Tuple{Int, Int, FT}[]), (outlet_i[n], outlet_j[n], w))
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
        for (oi, oj, s) in mouths
            push!(contribution_outlet_i, oi)
            push!(contribution_outlet_j, oj)
            push!(contribution_weight, s / A)
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

@kernel function _compute_wet_mask_area_and_depth!(wet, area, depth, grid, kᴺ)
    i, j = @index(Global, NTuple)
    D = zero(grid)
    for k in 1:kᴺ
        inactive = inactive_node(i, j, k, grid, Center(), Center(), Center())
        D += ifelse(inactive, zero(grid), Δzᶜᶜᶜ(i, j, k, grid))
    end
    @inbounds begin
        wet[i, j, 1] = !inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())
        area[i, j, 1] = Azᶜᶜᶜ(i, j, kᴺ, grid)
        depth[i, j, 1] = D
    end
end

node_λφ(λc::AbstractVector, φc::AbstractVector, i, j) = (λc[i], φc[j])
node_λφ(λc::AbstractMatrix, φc::AbstractMatrix, i, j) = (λc[i, j], φc[i, j])

wrap180(λ) = λ - 360 * floor((λ + 180) / 360)

function squared_distance(λ₁, φ₁, λ₂, φ₂)
    Δλ = wrap180(λ₂ - λ₁) * cosd((φ₁ + φ₂) / 2)
    Δφ = φ₂ - φ₁
    return Δλ^2 + Δφ^2
end

"""
    masked_wet_cells(mask, wet, λc, φc)

Wet cells of the target grid that lie inside `mask`, as `(i, j, λ, φ)` vectors.
"""
function masked_wet_cells(mask, wet, λc, φc)
    Nx, Ny = size(wet)
    is = Int[]; js = Int[]; λs = Float64[]; φs = Float64[]
    for j in 1:Ny, i in 1:Nx
        (wet[i, j] && mask[i, j]) || continue
        λ, φ = node_λφ(λc, φc, i, j)
        push!(is, i); push!(js, j); push!(λs, λ); push!(φs, φ)
    end
    return is, js, λs, φs
end

"""
    diversion_target_cells(receiver_i, receiver_j, receiver_λ, receiver_φ, λₒ, φₒ,
                           max_degrees, spread_radius, n_spread_cells; longitude_shift = 130)

The cells that receive water diverted away from a mouth at `(λₒ, φₒ)`. The mouth is relocated to its
counterpart in the destination basin — same latitude, `longitude_shift` degrees west — and the
discharge is then spread with the same footprint a real river gets, so each mouth lands in its own
place and the flux per unit area stays in the range the ocean already handles. Selecting by latitude
alone instead would funnel every mouth in a band onto one footprint. The caller weights the returned
cells by column depth exactly as it weights a mouth's own footprint.
"""
function diversion_target_cells(receiver_i, receiver_j, receiver_λ, receiver_φ, λₒ, φₒ,
                                max_degrees, spread_radius, n_spread_cells; longitude_shift = 130)
    λ★ = wrap180(λₒ - longitude_shift)
    targets = spread_target_cells(receiver_i, receiver_j, receiver_λ, receiver_φ, λ★, φₒ,
                                  max_degrees, spread_radius, n_spread_cells)
    isempty(targets) || return targets
    # No destination cell within reach of the counterpart position — the basins do not face each other
    # at every latitude. Fall back to the nearest cells, still a footprint: a single cell would take a
    # whole mouth's discharge and drive its salinity to zero on the first step.
    order = sortperm([squared_distance(λ★, φₒ, receiver_λ[m], receiver_φ[m]) for m in eachindex(receiver_λ)])
    keep = order[1:min(n_spread_cells isa Number ? n_spread_cells : 8, length(order))]
    return [(receiver_i[m], receiver_j[m]) for m in keep]
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
    return wet_i, wet_j, wet_λ, wet_φ
end

"""
    spread_target_cells(wet_i, wet_j, wet_λ, wet_φ, λₒ, φₒ, max_degrees, spread_radius, maximum_cells)

Return the ocean cells a mouth at `(λₒ, φₒ)` discharges into: every wet cell within `spread_radius`
degrees of the mouth's landing cell, nearest first, capped at `maximum_cells` unless that is `nothing`.
Empty when no wet cell lies within `max_degrees` of the mouth.

`spread_radius = nothing` instead takes the `maximum_cells` cells nearest the outlet itself, a footprint
fixed in cell count rather than in area.
"""
function spread_target_cells(wet_i, wet_j, wet_λ, wet_φ, λₒ, φₒ, max_degrees, spread_radius, maximum_cells)
    reach = max_degrees^2
    nearest = 0
    nearest_distance = Inf
    reachable = Tuple{Float64, Int}[]
    for n in eachindex(wet_i)
        d = squared_distance(λₒ, φₒ, wet_λ[n], wet_φ[n])
        d < nearest_distance && (nearest_distance = d; nearest = n)
        d < reach && push!(reachable, (d, n))
    end
    nearest_distance < reach || return Tuple{Int, Int}[]

    targets = if isnothing(spread_radius)
        sort!(reachable; by = first)   # cell-count footprint, ranked from the outlet
    else
        # Spread around the landing cell rather than the outlet, so mouths relocated onto the shelf
        # (the Ob and Yenisei move 2-3°) still get a full footprint instead of collapsing onto one cell.
        λ★, φ★ = wet_λ[nearest], wet_φ[nearest]
        footprint = spread_radius^2
        centred = [(squared_distance(λ★, φ★, wet_λ[n], wet_φ[n]), n) for (_, n) in reachable]
        filter!(t -> first(t) <= footprint, centred)
        sort!(centred; by = first)
    end

    nkeep = isnothing(maximum_cells) ? length(targets) : min(maximum_cells, length(targets))
    return [(wet_i[targets[m][2]], wet_j[targets[m][2]]) for m in 1:nkeep]
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
        scatter_freshwater_flux!(land_freshwater_flux, land.freshwater_flux[name], land.river_routing[name], arch, grid, time)
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
