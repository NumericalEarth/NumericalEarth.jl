#####
##### Sill overflow parameterization at Denmark Strait
#####
#
# The tracer advection scheme mixes the overflow away on the multi-cell bottom steps below the sill, so the
# product is moved across the staircase as a tracer exchange instead of advected down it, following
# Danabasoglu, Large & Briegleb (2010) with the rotating hydraulic transport of Whitehead et al. (1974).
# The resolved velocity is untouched. Every update:
#
#     h_u = thickness above the sill of source water denser, at the sill pressure, than the interior at sill depth
#     g′  = g (ρ̄_d − ρ_I) / ρ₀                    ρ̄_d: mean density of that dense layer at the sill pressure
#     M_s = α g′ h_u² / 2f,   M_e = φ M_s
#     c_p = (M_s c̄_d + M_e c̄_E) / (M_s + M_e),   applied only if the product is denser than the P region
#
# The exchange is volume-neutral and conserves heat and salt: the dense layer and the entrainment region lose their
# water to the product region and receive its ambient water back,
#
#     V_d ∂c/∂t = M_s (c̄_P − c̄_d),   V_E ∂c/∂t = M_e (c̄_P − c̄_E),   V_P ∂c/∂t = M_s c̄_d + M_e c̄_E − (M_s + M_e) c̄_P

using Oceananigans
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: Center, λnode, φnode, znode, znodes
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Operators: Azᶜᶜᶜ, Vᶜᶜᶜ
using Printf
using SeawaterPolynomials: SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

struct SillRegion{FT}
    i :: UnitRange{Int}
    j :: UnitRange{Int}
    k :: UnitRange{Int}
    volumes :: Array{FT, 3}   # cell volumes over the index block, zero outside the region
end

mutable struct SillOverflowState{FT}
    source_transport      :: FT
    entrainment_transport :: FT
    upstream_height       :: FT
    reduced_gravity       :: FT
    product_temperature   :: FT
    product_salinity      :: FT
    applied               :: Bool
end

struct SillOverflow{F, A, FT, E}
    region_codes          :: F     # 1 source columns above the sill, 2 entrainment, 3 product, 0 elsewhere
    increments            :: A     # (Nz + 2) × 2: source levels, entrainment, product; columns T and S [units s⁻¹]
    source                :: SillRegion{FT}
    interior              :: SillRegion{FT}
    entrainment           :: SillRegion{FT}
    product               :: SillRegion{FT}
    level_depths          :: Vector{FT}
    source_level_areas    :: Vector{FT}
    sill_level            :: Int
    product_depth         :: FT
    coriolis_parameter    :: FT
    hydraulic_factor      :: FT
    entrainment_ratio     :: FT
    maximum_transport     :: FT
    halo                  :: NTuple{3, Int}
    equation_of_state     :: E
    state                 :: SillOverflowState{FT}
end

function sill_region_cells(cpu_grid; longitude, latitude, levels)
    Nx, Ny, Nz = size(cpu_grid)
    cells = CartesianIndex{3}[]
    for k in levels, j in 1:Ny, i in 1:Nx
        inactive_node(i, j, k, cpu_grid, Center(), Center(), Center()) && continue
        λ = λnode(i, j, k, cpu_grid, Center(), Center(), Center())
        λ = λ > 180 ? λ - 360 : λ
        φ = φnode(i, j, k, cpu_grid, Center(), Center(), Center())
        longitude[1] <= λ <= longitude[2] && latitude[1] <= φ <= latitude[2] && push!(cells, CartesianIndex(i, j, k))
    end
    return cells
end

function SillRegion(cpu_grid, cells)
    isempty(cells) && throw(ArgumentError("sill overflow region contains no wet cells"))
    i = minimum(c[1] for c in cells):maximum(c[1] for c in cells)
    j = minimum(c[2] for c in cells):maximum(c[2] for c in cells)
    k = minimum(c[3] for c in cells):maximum(c[3] for c in cells)
    volumes = zeros(eltype(cpu_grid), length(i), length(j), length(k))
    for c in cells
        volumes[c[1] - first(i) + 1, c[2] - first(j) + 1, c[3] - first(k) + 1] = Vᶜᶜᶜ(c[1], c[2], c[3], cpu_grid)
    end
    return SillRegion(i, j, k, volumes)
end

levels_between(z, shallowest, deepest) = findall(zₖ -> shallowest <= -zₖ <= deepest, z)

"""
    SillOverflow(grid; sill_depth, latitude, source_box, interior_box, entrainment_box, product_box, ...)

Overflow exchange across one sill. Boxes are `(longitude, latitude)` pairs of intervals; the entrainment and
product boxes also take a `depth` interval. Defaults are Denmark Strait, calibrated on eORCA1 in campaign 36 (C36-25).
"""
function SillOverflow(grid;
                      sill_depth = 690,
                      latitude = 66,
                      source_box = (longitude = (-26, -18), latitude = (66.5, 69)),
                      interior_box = (longitude = (-33, -25), latitude = (63.5, 65.5)),
                      entrainment_box = (longitude = (-36, -28), latitude = (63, 65), depth = (700, 1500)),
                      product_box = (longitude = (-36, -26), latitude = (62, 66.4), depth = (1500, 3000)),
                      hydraulic_factor = 1,
                      entrainment_ratio = 1,
                      maximum_transport = 5e6)

    FT = eltype(grid)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(cpu_grid)
    z = znodes(cpu_grid, Center())
    sill_level = argmin(abs.(z .+ sill_depth))

    source_cells      = sill_region_cells(cpu_grid; source_box..., levels = sill_level:Nz)
    interior_cells    = sill_region_cells(cpu_grid; interior_box..., levels = sill_level:sill_level)
    entrainment_cells = sill_region_cells(cpu_grid; entrainment_box.longitude, entrainment_box.latitude,
                                          levels = levels_between(z, entrainment_box.depth...))
    product_cells     = sill_region_cells(cpu_grid; product_box.longitude, product_box.latitude,
                                          levels = levels_between(z, product_box.depth...))

    codes = zeros(FT, Nx, Ny, Nz)
    for (code, cells) in ((1, source_cells), (2, entrainment_cells), (3, product_cells)), c in cells
        codes[c] == 0 || throw(ArgumentError("sill overflow regions overlap at $c"))
        codes[c] = code
    end
    region_codes = Field{Center, Center, Center}(grid)
    set!(region_codes, codes)
    fill_halo_regions!(region_codes)

    source = SillRegion(cpu_grid, source_cells)
    source_level_areas = zeros(FT, length(source.k))
    for c in source_cells
        source_level_areas[c[3] - first(source.k) + 1] += Azᶜᶜᶜ(c[1], c[2], c[3], cpu_grid)
    end

    product = SillRegion(cpu_grid, product_cells)
    product_depth = -sum(z[first(product.k) + k - 1] * sum(view(product.volumes, :, :, k)) for k in axes(product.volumes, 3)) /
                    sum(product.volumes)

    increments = on_architecture(architecture(grid), zeros(FT, Nz + 2, 2))
    state = SillOverflowState(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), false)

    return SillOverflow(region_codes, increments,
                        source, SillRegion(cpu_grid, interior_cells),
                        SillRegion(cpu_grid, entrainment_cells), product,
                        FT.(z), source_level_areas, sill_level, FT(product_depth), FT(2 * 7.292115e-5 * sind(latitude)),
                        FT(hydraulic_factor), FT(entrainment_ratio), FT(maximum_transport),
                        (cpu_grid.Hx, cpu_grid.Hy, cpu_grid.Hz), TEOS10EquationOfState(), state)
end

function region_block(field, region, halo)
    Hx, Hy, Hz = halo
    return Array(parent(field)[region.i .+ Hx, region.j .+ Hy, region.k .+ Hz])
end

regional_mean(c, region) = sum(c .* region.volumes) / sum(region.volumes)

function update_sill_overflow!(overflow::SillOverflow, T, S)
    ρ(Θ, Sᴬ, z) = SeawaterPolynomials.ρ(Θ, Sᴬ, z, overflow.equation_of_state)
    g = 9.81
    ρ₀ = 1027
    zˢ = overflow.level_depths[overflow.sill_level]

    Tᴵ = regional_mean(region_block(T, overflow.interior, overflow.halo), overflow.interior)
    Sᴵ = regional_mean(region_block(S, overflow.interior, overflow.halo), overflow.interior)
    ρᴵ = ρ(Tᴵ, Sᴵ, zˢ)

    source = overflow.source
    Tˢ = region_block(T, source, overflow.halo)
    Sˢ = region_block(S, source, overflow.halo)
    level_volume = vec(sum(source.volumes, dims = (1, 2)))
    level_temperature = vec(sum(Tˢ .* source.volumes, dims = (1, 2))) ./ max.(level_volume, eps())
    level_salinity    = vec(sum(Sˢ .* source.volumes, dims = (1, 2))) ./ max.(level_volume, eps())

    # the dense layer is contiguous upward from the sill level (source.k starts at the sill level)
    dense_levels = 0
    for n in eachindex(level_volume)
        level_volume[n] > 0 && ρ(level_temperature[n], level_salinity[n], zˢ) > ρᴵ || break
        dense_levels = n
    end
    dense = 1:dense_levels
    Vᵈ = sum(level_volume[dense]; init = 0.0)
    Tᵈ = sum(level_temperature[dense] .* level_volume[dense]; init = 0.0) / max(Vᵈ, eps())
    Sᵈ = sum(level_salinity[dense] .* level_volume[dense]; init = 0.0) / max(Vᵈ, eps())
    ρᵈ = ρ(Tᵈ, Sᵈ, zˢ)
    hᵘ = sum(level_volume[dense] ./ overflow.source_level_areas[dense]; init = 0.0)

    g′ = max(0, g * (ρᵈ - ρᴵ) / ρ₀)
    Mˢ = clamp(overflow.hydraulic_factor * g′ * hᵘ^2 / (2 * overflow.coriolis_parameter), 0, overflow.maximum_transport)
    Mᵉ = overflow.entrainment_ratio * Mˢ

    Tᴱ = regional_mean(region_block(T, overflow.entrainment, overflow.halo), overflow.entrainment)
    Sᴱ = regional_mean(region_block(S, overflow.entrainment, overflow.halo), overflow.entrainment)
    Tᴾ = regional_mean(region_block(T, overflow.product, overflow.halo), overflow.product)
    Sᴾ = regional_mean(region_block(S, overflow.product, overflow.halo), overflow.product)

    Tᵖ = (Mˢ * Tᵈ + Mᵉ * Tᴱ) / max(Mˢ + Mᵉ, eps())
    Sᵖ = (Mˢ * Sᵈ + Mᵉ * Sᴱ) / max(Mˢ + Mᵉ, eps())
    zᴾ = -overflow.product_depth
    applied = Mˢ > 0 && ρ(Tᵖ, Sᵖ, zᴾ) > ρ(Tᴾ, Sᴾ, zᴾ)
    Mˢ, Mᵉ = applied ? (Mˢ, Mᵉ) : (zero(Mˢ), zero(Mᵉ))

    Vᴱ = sum(overflow.entrainment.volumes)
    Vᴾ = sum(overflow.product.volumes)
    Nz = length(overflow.level_depths)
    increments = zeros(eltype(overflow.increments), Nz + 2, 2)
    for (column, cᵈ, cᴱ, cᴾ) in ((1, Tᵈ, Tᴱ, Tᴾ), (2, Sᵈ, Sᴱ, Sᴾ))
        for n in dense
            increments[source.k[n], column] = Mˢ * (cᴾ - cᵈ) / Vᵈ
        end
        increments[Nz + 1, column] = Mᵉ * (cᴾ - cᴱ) / Vᴱ
        increments[Nz + 2, column] = (Mˢ * cᵈ + Mᵉ * cᴱ - (Mˢ + Mᵉ) * cᴾ) / Vᴾ
    end
    copyto!(overflow.increments, increments)

    state = overflow.state
    state.source_transport = Mˢ
    state.entrainment_transport = Mᵉ
    state.upstream_height = hᵘ
    state.reduced_gravity = g′
    state.product_temperature = Tᵖ
    state.product_salinity = Sᵖ
    state.applied = applied

    return nothing
end

function update_sill_overflow!(simulation, overflow::SillOverflow)
    tracers = simulation.model.ocean.model.tracers
    update_sill_overflow!(overflow, tracers.T, tracers.S)
    s = overflow.state
    @info @sprintf("SILL OVERFLOW M_s=%.3f Sv M_e=%.3f Sv h_u=%.0f m g′=%.2e Θ_p=%.2f Sᴬ_p=%.3f applied=%s",
                   s.source_transport / 1e6, s.entrainment_transport / 1e6, s.upstream_height, s.reduced_gravity,
                   s.product_temperature, s.product_salinity, s.applied)
    return nothing
end

struct SillOverflowUpdate{O}
    overflow :: O
end

(u::SillOverflowUpdate)(simulation) = update_sill_overflow!(simulation, u.overflow)

@inline function sill_overflow_tendency(i, j, k, grid, clock, fields, p)
    @inbounds begin
        r  = p.region_codes[i, j, k]
        Gˢ = p.increments[k, p.column]
        Gᴱ = p.increments[p.Nz + 1, p.column]
        Gᴾ = p.increments[p.Nz + 2, p.column]
    end
    return (r == 1) * Gˢ + (r == 2) * Gᴱ + (r == 3) * Gᴾ
end

"""
    sill_overflow_forcing(grid, enabled)

Return `((T, S) forcings, overflow)`, or `(NamedTuple(), nothing)` when `enabled` is `false`. The caller registers
`SillOverflowUpdate(overflow)` as a callback.
"""
function sill_overflow_forcing(grid, enabled)
    enabled || return NamedTuple(), nothing
    overflow = SillOverflow(grid)
    Nz = size(grid, 3)
    parameters(column) = (; region_codes = overflow.region_codes, increments = overflow.increments, Nz, column)
    T = Forcing(sill_overflow_tendency; discrete_form = true, parameters = parameters(1))
    S = Forcing(sill_overflow_tendency; discrete_form = true, parameters = parameters(2))
    return (; T, S), overflow
end
