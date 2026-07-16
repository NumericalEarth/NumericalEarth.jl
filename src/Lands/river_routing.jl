using Oceananigans.Grids: inactive_node, λnodes, φnodes
using Oceananigans.Operators: Azᶜᶜᶜ
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
function positive_outlet_indices(flux)
    grid = flux.grid
    outlet_mask = Array(interior(flux))[:, :, 1] .> 0
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
                        maximum_search_radius = 5)

Map each river mouth at `(outlet_λ, outlet_φ)` to the nearest active ocean cell of `target_grid` within `maximum_search_radius`
cells, returning a [`RiverRouting`](@ref). River mouths with no active ocean cell in range are dropped (and reported), so the
global freshwater budget is conserved up to the dropped discharge.

`outlet_weight[n]` is the per-mouth factor that converts the outlet's stored value into a mass discharge (kg s⁻¹):
the deposited flux is `outlet_weight[n] * value[outlet_n] / Aᵒᶜᵉᵃⁿ`. For a volumetric discharge (m³ s⁻¹) it is the
freshwater density; for a per-area mass flux (kg m⁻² s⁻¹) it is the source-cell area. Both conserve the total mass delivered.
"""
function build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ, outlet_weight;
                             maximum_search_radius = 5,
                             n_spread_cells = 8)

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
    wet_i, wet_j, wet_λ, wet_φ = wet_cells(wet, λc, φc)
    max_degrees = maximum_search_radius * (360 / Nx + 180 / Ny) / 2

    # Split each mouth's discharge equally among its `n_spread_cells` nearest ocean cells so
    # no single coarse coastal cell receives a runaway freshwater flux (which crashes salinity).
    contributions = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int, FT}}}()
    dropped = 0
    for n in eachindex(outlet_i)
        targets = nearest_active_cells(wet_i, wet_j, wet_λ, wet_φ, outlet_λ[n], outlet_φ[n],
                                       max_degrees, n_spread_cells)
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
    return wet_i, wet_j, wet_λ, wet_φ
end

function nearest_active_cells(wet_i, wet_j, wet_λ, wet_φ, λₒ, φₒ, max_degrees, K)
    cap = max_degrees^2
    candidates = Tuple{Float64, Int}[]
    for n in eachindex(wet_i)
        d = squared_distance(λₒ, φₒ, wet_λ[n], wet_φ[n])
        d < cap && push!(candidates, (d, n))
    end
    sort!(candidates; by = first)
    nfound = min(K, length(candidates))
    return [(wet_i[candidates[m][2]], wet_j[candidates[m][2]]) for m in 1:nfound]
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
