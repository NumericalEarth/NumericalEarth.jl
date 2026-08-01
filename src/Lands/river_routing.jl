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
`contribution_weight = ρ_freshwater / Aᵒᶜᵉᵃⁿ`, chosen so the area integral of
the deposited flux equals the total discharge times the freshwater density.
"""
struct RiverRouting{I, W}
    contribution_outlet_i :: I
    contribution_outlet_j :: I
    contribution_weight   :: W
    target_i :: I
    target_j :: I
    offsets  :: I
end

const RoutedPrescribedLand = PrescribedLand{<:Any, <:Any, <:Any, <:Any, <:RiverRouting}

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

#####
##### Building the routing map (construction-time, on CPU)
#####

"""
    build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ;
                        freshwater_density = 1000,
                        maximum_search_radius = 5)

Map each river mouth at `(outlet_λ, outlet_φ)` to the nearest active ocean cell
of `target_grid` within `maximum_search_radius` cells, returning a
[`RiverRouting`](@ref). River mouths with no active ocean cell in range are
dropped (and reported), so the global freshwater budget is conserved up to the
dropped discharge.
"""
function build_river_routing(target_grid, outlet_i, outlet_j, outlet_λ, outlet_φ;
                             freshwater_density = 1000,
                             maximum_search_radius = 5)

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

    # Group contributing river mouths by destination ocean cell.
    contributions = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    dropped = 0
    for n in eachindex(outlet_i)
        i★, j★ = nearest_active_cell(wet, λc, φc, outlet_λ[n], outlet_φ[n], maximum_search_radius)
        if i★ == 0
            dropped += 1
            continue
        end
        push!(get!(contributions, (i★, j★), Tuple{Int, Int}[]), (outlet_i[n], outlet_j[n]))
    end

    if dropped > 0
        @warn string(dropped, " of ", length(outlet_i), " river mouths had no active ocean ",
                     "cell within ", maximum_search_radius, " cells and were dropped.")
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
        w = convert(FT, freshwater_density) / convert(FT, area[i★, j★])
        for (oi, oj) in mouths
            push!(contribution_outlet_i, oi)
            push!(contribution_outlet_j, oj)
            push!(contribution_weight, w)
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

# Index of the entry of sorted vector `a` closest to `x`.
function searchsortednearest(a, x)
    i = searchsortedfirst(a, x)
    i == 1 && return 1
    i > length(a) && return length(a)
    return abs(a[i-1] - x) ≤ abs(a[i] - x) ? i - 1 : i
end

# Approximate squared distance on the sphere (equirectangular, degrees).
function squared_distance(λ₁, φ₁, λ₂, φ₂)
    Δλ = (λ₂ - λ₁) * cosd((φ₁ + φ₂) / 2)
    Δφ = φ₂ - φ₁
    return Δλ^2 + Δφ^2
end

# Spiral search outward from the target cell containing (λₒ, φₒ) for the nearest
# active ocean cell within `R` cells (Chebyshev), ranked by metric distance.
function nearest_active_cell(wet, λc, φc, λₒ, φₒ, R)
    Nx, Ny = size(wet)
    i₀ = clamp(searchsortednearest(λc, λₒ), 1, Nx)
    j₀ = clamp(searchsortednearest(φc, φₒ), 1, Ny)

    best_i = 0
    best_j = 0
    best_d = Inf

    for r in 0:R, di in -r:r, dj in -r:r
        max(abs(di), abs(dj)) == r || continue
        i = i₀ + di
        j = j₀ + dj
        (1 ≤ i ≤ Nx && 1 ≤ j ≤ Ny) || continue
        wet[i, j] || continue
        d = squared_distance(λₒ, φₒ, λc[i], φc[j])
        if d < best_d
            best_d = d
            best_i = i
            best_j = j
        end
    end

    return best_i, best_j
end

#####
##### Conservative scatter of river discharge onto the ocean grid
#####

"""Scatter prescribed river-mouth discharge onto coastal ocean cells, conserving volume."""
function EarthSystemModels.interpolate_state!(exchanger, grid, land::RoutedPrescribedLand, coupled_model)
    arch = architecture(grid)
    clock = coupled_model.clock
    land_freshwater_flux = exchanger.state.freshwater_flux

    fill!(land_freshwater_flux, 0)

    routing = land.river_routing
    n_targets = length(routing.target_i)
    n_targets == 0 && return nothing

    discharge = first(land.freshwater_flux)
    time = Time(clock.time)

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

# One thread per destination ocean cell sums all river mouths routed to it, so
# each cell is written exactly once — no atomics needed.
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
        flux[target_i[c], target_j[c], 1] = accumulated
    end
end
