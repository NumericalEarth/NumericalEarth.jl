#####
##### Sill overflow parameterization for the Greenland–Scotland Ridge gateways
#####
#
# The tracer advection scheme mixes the overflow away on the multi-cell bottom steps below the sills, so the
# product is moved across the staircase as a tracer exchange instead of advected down it, following
# Danabasoglu, Large & Briegleb (2010) with the rotating hydraulic transport of Whitehead et al. (1974).
# The resolved velocity is untouched.
#
# Nothing vertical is prescribed. Each sill supplies three horizontal boxes and nothing else — the upstream
# `source_box` it drains, the `approach_box` just upstream of the throat that sets the hydraulic state, and the
# downstream `plume_box` its plume descends. Every depth, the latitude, the slope and every participating level
# come from the grid and the density field:
#
#     z_s   = deepest passage through the throat, from the bathymetry
#     h_u   = thickness of approach water denser, at z_s, than the downstream column at z_s
#     b′(z) = g (ρ(z) − ρ_I) / ρ₀ over that layer, ζ its height above the sill
#     M_w   = (α/f) ∫ b′ ζ dζ                               Whitehead et al. (1974) rotating hydraulic transport
#     M_r   = net downward volume flux the model already carries across z_n in the plume columns
#     M_s   = clamp(M_w − M_r, 0, M_max)
#     Fr    = √(sinθ / C_d),  φ = clamp(c_e (Fr − 1), 0, φ_max),  M_e = φ M_s   θ: the model's descent slope
#     z_n   = deepest level at which the product is denser than the ambient plume profile
#     band  = the levels bracketing z_n;  entrainment = the levels the plume descends through, band to z_s
#
# The approach and the source are separate on purpose. Whitehead's h and g′ are properties of the water just
# upstream of the control, so they are measured in a box hugging the sill; the water is then drawn from the
# whole basin, which keeps the transport small against the reservoir that has to supply it. Measuring both in
# the basin makes h_u the full reservoir depth and overestimates the transport — at the Faroe Bank Channel by
# more than a factor of two (C36-25 calibrates 404 m and 2.88 Sv against a basin-wide 756 m and 6.5 Sv).
#
# The exchange is volume-neutral and conserves heat and salt exactly: the dense upstream layer and the descent
# path give their water to the neutral band and receive its water back,
#
#     V_d ∂c/∂t = M_s (c̄_N − c̄_d),  V_E ∂c/∂t = M_e (c̄_N − c̄_E),  V_N ∂c/∂t = M_s c̄_d + M_e c̄_E − (M_s + M_e) c̄_N
#
# which sum to zero. The source lies above the sill and the plume at or below it, so the two regions of a sill
# are disjoint by construction. Cells are coded `2(s − 1) + r` for sill `s` and `r` ∈ {source, plume}, and the
# increments are carried per level, so one `Field` and one small array serve any number of sills.

using Oceananigans
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: Center, Face, λnode, φnode, znode, znodes
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Operators: Azᶜᶜᶜ, Vᶜᶜᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ
using Printf
using Statistics: median
using SeawaterPolynomials: SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

struct SillRegion{FT}
    i       :: UnitRange{Int}
    j       :: UnitRange{Int}
    k       :: UnitRange{Int}
    volumes :: Array{FT, 3}   # cell volumes over the index block, zero outside the region
    areas   :: Array{FT, 3}   # horizontal cell areas, zero outside the region
end

mutable struct SillOverflowState{FT}
    hydraulic_transport   :: FT
    resolved_descent      :: FT
    source_transport      :: FT
    entrainment_transport :: FT
    froude_number         :: FT
    upstream_height       :: FT
    reduced_gravity       :: FT
    product_temperature   :: FT
    product_salinity      :: FT
    neutral_depth         :: FT
    applied               :: Bool
end

struct SillOverflow{FT, E}
    code                :: Int
    name                :: String
    source              :: SillRegion{FT}
    approach            :: SillRegion{FT}
    plume               :: SillRegion{FT}
    level_depths        :: Vector{FT}
    sill_level          :: Int
    sill_depth          :: FT
    coriolis_parameter  :: FT
    hydraulic_factor    :: FT
    entrainment_factor  :: FT
    maximum_entrainment :: FT
    bottom_slope        :: FT
    drag_coefficient    :: FT
    maximum_transport   :: FT
    halo                :: NTuple{3, Int}
    equation_of_state   :: E
    state               :: SillOverflowState{FT}
end

struct SillOverflowSet{S, F, A}
    sills        :: S
    region_codes :: F
    increments   :: A
end

inside(λ, φ, box) = (box.longitude[1] <= λ <= box.longitude[2]) & (box.latitude[1] <= φ <= box.latitude[2])

function horizontal_node(i, j, cpu_grid)
    λ = λnode(i, j, 1, cpu_grid, Center(), Center(), Center())
    return (λ > 180 ? λ - 360 : λ), φnode(i, j, 1, cpu_grid, Center(), Center(), Center())
end

function sill_region_cells(cpu_grid, box, levels)
    Nx, Ny, _ = size(cpu_grid)
    cells = CartesianIndex{3}[]
    for k in levels, j in 1:Ny, i in 1:Nx
        inactive_node(i, j, k, cpu_grid, Center(), Center(), Center()) && continue
        λ, φ = horizontal_node(i, j, cpu_grid)
        inside(λ, φ, box) && push!(cells, CartesianIndex(i, j, k))
    end
    return cells
end

function SillRegion(cpu_grid, cells)
    isempty(cells) && throw(ArgumentError("sill overflow region contains no wet cells"))
    i = minimum(c[1] for c in cells):maximum(c[1] for c in cells)
    j = minimum(c[2] for c in cells):maximum(c[2] for c in cells)
    k = minimum(c[3] for c in cells):maximum(c[3] for c in cells)
    volumes = zeros(eltype(cpu_grid), length(i), length(j), length(k))
    areas   = zeros(eltype(cpu_grid), length(i), length(j), length(k))
    for c in cells
        n = (c[1] - first(i) + 1, c[2] - first(j) + 1, c[3] - first(k) + 1)
        volumes[n...] = Vᶜᶜᶜ(c[1], c[2], c[3], cpu_grid)
        areas[n...]   = Azᶜᶜᶜ(c[1], c[2], c[3], cpu_grid)
    end
    return SillRegion(i, j, k, volumes, areas)
end

"""
    column_depths(cpu_grid)

Depth of the deepest wet cell of every column, zero over land.
"""
function column_depths(cpu_grid)
    Nx, Ny, Nz = size(cpu_grid)
    z = znodes(cpu_grid, Center())
    depth = zeros(eltype(cpu_grid), Nx, Ny)
    for j in 1:Ny, i in 1:Nx, k in 1:Nz
        inactive_node(i, j, k, cpu_grid, Center(), Center(), Center()) || (depth[i, j] = max(depth[i, j], -z[k]))
    end
    return depth
end

"""
    descent_slope(cpu_grid, depth_field, columns, shallowest)

The bottom slope the plume descends: the median |∇h| over the columns below `shallowest`, taken from
neighbouring column depths and the grid's own spacings. The median rather than the mean, and only the columns
below the sill, because the plume box also spans flat basin floor whose zero gradient would otherwise halve
the slope — and the slope alone sets the Froude number, so a factor of two there switches entrainment off.
"""
function descent_slope(cpu_grid, depth_field, columns, shallowest)
    Nx, Ny = size(depth_field)
    slopes = eltype(depth_field)[]
    for c in columns
        depth_field[c] > shallowest || continue
        i, j = c[1], c[2]
        iᵉ, iʷ = mod1(i + 1, Nx), mod1(i - 1, Nx)
        jⁿ, jˢ = min(j + 1, Ny), max(j - 1, 1)
        (depth_field[iᵉ, j] > 0 && depth_field[iʷ, j] > 0 && depth_field[i, jⁿ] > 0 && depth_field[i, jˢ] > 0) || continue
        ∂xh = (depth_field[iᵉ, j] - depth_field[iʷ, j]) / (2 * Δxᶜᶜᶜ(i, j, 1, cpu_grid))
        ∂yh = (depth_field[i, jⁿ] - depth_field[i, jˢ]) / (2 * Δyᶜᶜᶜ(i, j, 1, cpu_grid))
        push!(slopes, sqrt(∂xh^2 + ∂yh^2))
    end
    isempty(slopes) && return 1e-2
    return clamp(median(slopes), 1e-3, 0.2)
end

function box_columns(cpu_grid, box)
    Nx, Ny, _ = size(cpu_grid)
    columns = CartesianIndex{2}[]
    latitudes = eltype(cpu_grid)[]
    for j in 1:Ny, i in 1:Nx
        λ, φ = horizontal_node(i, j, cpu_grid)
        inside(λ, φ, box) && (push!(columns, CartesianIndex(i, j)); push!(latitudes, φ))
    end
    isempty(columns) && throw(ArgumentError("sill overflow box contains no columns"))
    return columns, sum(latitudes) / length(latitudes)
end

function basins_connect(depth_field, sources, targets, corridor, depth)
    Nx, Ny = size(depth_field)
    target = falses(Nx, Ny)
    for c in targets
        target[c] = true
    end
    seen = trues(Nx, Ny)
    for c in corridor
        seen[c] = false
    end
    stack = CartesianIndex{2}[]
    for c in sources
        depth_field[c] >= depth && !seen[c] && (seen[c] = true; push!(stack, c))
    end
    while !isempty(stack)
        c = pop!(stack)
        target[c] && return true
        i, j = c[1], c[2]
        for n in (CartesianIndex(mod1(i - 1, Nx), j), CartesianIndex(mod1(i + 1, Nx), j),
                  CartesianIndex(i, max(j - 1, 1)), CartesianIndex(i, min(j + 1, Ny)))
            seen[n] || depth_field[n] < depth || (seen[n] = true; push!(stack, n))
        end
    end
    return false
end

"""
    sill_depth(depth_field, source_columns, plume_columns, corridor, z)

The bottleneck of the deepest connected path from the upstream basin to the downstream one: the largest depth
at which the two are still joined by wet columns of `corridor`. This is the definition of a sill, and it needs
no box drawn around the throat — the bathymetry the model actually has decides where the control is and how
deep it sits. The corridor is the span of the sill's own source and plume boxes, which keeps each strait from
finding its neighbour's deeper passage.
"""
function sill_depth(depth_field, source_columns, plume_columns, corridor, z)
    for depth in sort(unique(-z); rev = true)
        basins_connect(depth_field, source_columns, plume_columns, corridor, depth) && return depth
    end
    return 0.0
end

"""
Denmark Strait and the Faroe Bank Channel, the two branches that carry the Nordic overflow. Each sill gives the
upstream reservoir it drains, the approach just upstream of its throat, and the downstream columns its plume
descends; the sill depth, the latitude, the descent slope and every participating level come from the grid and
the density field. The reservoirs are the Iceland/Greenland Seas and the Norwegian Sea rather than boxes hugging
the straits, so that the transport drawn off is small against the water available to supply it; the approach
boxes are the calibrated ones of C36-25, and they alone set the hydraulic state.
"""
default_sill_configurations() = (
    (name = "Denmark Strait",
     source_box   = (longitude = (-30, -12), latitude = (66.5, 72.0)),
     approach_box = (longitude = (-30, -20), latitude = (66.5, 69.0)),
     plume_box    = (longitude = (-40, -25), latitude = (58.0, 66.4))),
    (name = "Faroe Bank Channel",
     source_box   = (longitude = (-8, 4), latitude = (61.5, 66.0)),
     approach_box = (longitude = (-8, -2), latitude = (62.0, 64.0)),
     plume_box    = (longitude = (-22, -10), latitude = (56.0, 61.5))),
)

function SillOverflow(grid, code, cfg;
                      hydraulic_factor = 1,
                      entrainment_factor = 1,
                      maximum_entrainment = 3,
                      drag_coefficient = 2.5e-3,
                      maximum_transport = 5e6)

    FT = eltype(grid)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(cpu_grid)
    z = znodes(cpu_grid, Center())

    depth_field = column_depths(cpu_grid)
    source_columns, _ = box_columns(cpu_grid, cfg.source_box)
    approach_columns, latitude = box_columns(cpu_grid, cfg.approach_box)
    plume_columns, _ = box_columns(cpu_grid, cfg.plume_box)
    corridor, _ = box_columns(cpu_grid, (longitude = extrema((cfg.source_box.longitude..., cfg.plume_box.longitude...)),
                                         latitude = extrema((cfg.source_box.latitude..., cfg.plume_box.latitude...))))
    zˢ = sill_depth(depth_field, source_columns, plume_columns, corridor, z)
    zˢ > 0 || throw(ArgumentError("$(cfg.name): the source and plume boxes are not connected"))
    sill_level = argmin(abs.(z .+ zˢ))

    source_cells   = sill_region_cells(cpu_grid, cfg.source_box, sill_level:Nz)
    approach_cells = sill_region_cells(cpu_grid, cfg.approach_box, sill_level:Nz)
    plume_cells    = sill_region_cells(cpu_grid, cfg.plume_box, 1:sill_level)

    state = SillOverflowState(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT),
                              zero(FT), zero(FT), zero(FT), zero(FT), false)

    overflow = SillOverflow{FT, typeof(TEOS10EquationOfState())}(
        code, cfg.name, SillRegion(cpu_grid, source_cells), SillRegion(cpu_grid, approach_cells),
        SillRegion(cpu_grid, plume_cells),
        FT.(z), sill_level, FT(zˢ), FT(2 * 7.292115e-5 * sind(latitude)),
        FT(hydraulic_factor), FT(entrainment_factor), FT(maximum_entrainment),
        FT(descent_slope(cpu_grid, depth_field, plume_columns, zˢ)), FT(drag_coefficient), FT(maximum_transport),
        (cpu_grid.Hx, cpu_grid.Hy, cpu_grid.Hz), TEOS10EquationOfState(), state)

    return overflow, (source_cells, plume_cells)
end

region_block(field, region, halo) =
    Array(parent(field)[region.i .+ halo[1], region.j .+ halo[2], region.k .+ halo[3]])

"""
    level_profile(c, region)

Volume-weighted mean of `c` on every level of `region`, with that level's volume. Index 1 is the region's
deepest level, so a source profile runs upward from the sill.
"""
function level_profile(c, region)
    v = vec(sum(region.volumes, dims = (1, 2)))
    m = vec(sum(c .* region.volumes, dims = (1, 2))) ./ max.(v, eps(eltype(v)))
    return m, v
end

band_mean(c, v, band) = sum(c[band] .* v[band]) / max(sum(v[band]), eps(eltype(v)))

"""
    resolved_descent(plume, wᴾ, n)

Net downward volume flux the model already carries across level `n` of the plume columns. Measured at the
neutral level rather than at the sill: at the sill it cancels against several Sverdrups of basin-scale
downwelling and switches the scheme off.
"""
function resolved_descent(plume, wᴾ, n)
    (n < 1 || n > length(plume.k)) && return 0.0
    return max(0.0, -sum(view(wᴾ, :, :, n) .* view(plume.areas, :, :, n)))
end

"""
    neutral_band(overflow, Θᵖ, Sᴬᵖ, Tᵃ, Sᵃ, volume)

The levels bracketing the depth at which the product stops being denser than the ambient plume column. Empty
when the product is lighter than every level, which switches the exchange off.
"""
function neutral_band(overflow::SillOverflow, Θᵖ, Sᴬᵖ, Tᵃ, Sᵃ, volume)
    ρ(Θ, Sᴬ, z) = SeawaterPolynomials.ρ(Θ, Sᴬ, z, overflow.equation_of_state)
    plume = overflow.plume
    neutral = 0
    # index 1 is the basin floor and `nˢ` the sill, so the plume descends with decreasing `n`
    for n in reverse(eachindex(volume))
        volume[n] > 0 || continue
        zₙ = overflow.level_depths[plume.k[n]]
        ρ(Θᵖ, Sᴬᵖ, zₙ) > ρ(Tᵃ[n], Sᵃ[n], zₙ) || break
        neutral = n
    end
    neutral == 0 && return 1:0
    return max(neutral - 1, 1):neutral
end

function update_sill_overflow!(overflow::SillOverflow, increments, T, S, w)
    ρ(Θ, Sᴬ, z) = SeawaterPolynomials.ρ(Θ, Sᴬ, z, overflow.equation_of_state)
    g = 9.81
    ρ₀ = 1027
    halo = overflow.halo
    source, approach, plume = overflow.source, overflow.approach, overflow.plume
    zˢ = overflow.sill_depth
    nˢ = length(plume.k)

    Tᵃ, pv = level_profile(region_block(T, plume, halo), plume)
    Sᵃ, _  = level_profile(region_block(S, plume, halo), plume)
    pv[nˢ] > 0 || return nothing
    ρᴵ = ρ(Tᵃ[nˢ], Sᵃ[nˢ], -zˢ)

    # The hydraulic state is the approach water denser than the downstream column at sill pressure; the density
    # field sets its thickness, so it can neither be truncated nor reach into water that cannot spill.
    Tᵈ, av = level_profile(region_block(T, approach, halo), approach)
    Sᵈ, _  = level_profile(region_block(S, approach, halo), approach)
    aa = vec(sum(approach.areas, dims = (1, 2)))
    dense_levels = 0
    for n in eachindex(av)
        av[n] > 0 && ρ(Tᵈ[n], Sᵈ[n], -zˢ) > ρᴵ || break
        dense_levels = n
    end
    dense_levels == 0 && return nothing
    dense = 1:dense_levels

    # Whitehead's g′h²/2f is (1/f)∫ b′ζ dζ over a homogeneous layer, so for a stratified one each level enters
    # with its own buoyancy anomaly times its height above the sill. Volume-weighting instead gives water a hair
    # denser than the reference the same weight as the water at the sill.
    Δz = av[dense] ./ max.(aa[dense], eps())
    hᵘ = sum(Δz)
    b′ = [max(0, g * (ρ(Tᵈ[n], Sᵈ[n], -zˢ) - ρᴵ) / ρ₀) for n in dense]
    ζ = cumsum(Δz) .- Δz ./ 2
    weight = b′ .* ζ .* Δz
    W = sum(weight)
    W > 0 || return nothing
    ŵ = weight ./ W

    g′ = sum(ŵ .* b′)
    Θᵈ = sum(ŵ .* Tᵈ[dense])
    Sᴬᵈ = sum(ŵ .* Sᵈ[dense])

    Mʷ = clamp(overflow.hydraulic_factor * W / overflow.coriolis_parameter, 0, overflow.maximum_transport)

    # The neutral level, the descent path and the entrained water each define the others, so iterate: start from
    # the unentrained product and refine.
    wᴾ = region_block(w, plume, halo)
    Θᵖ, Sᴬᵖ = Θᵈ, Sᴬᵈ
    Mʳ, Mˢ, Mᵉ, Fr = 0.0, Mʷ, 0.0, 0.0
    band = 1:0
    entrained = 1:0
    for _ in 1:2
        band = neutral_band(overflow, Θᵖ, Sᴬᵖ, Tᵃ, Sᵃ, pv)
        isempty(band) && break

        Mʳ = clamp(resolved_descent(plume, wᴾ, first(band) + 1), 0, Mʷ)
        Mˢ = clamp(Mʷ - Mʳ, 0, overflow.maximum_transport)

        # A buoyancy-drag balance on the descent slope, U³ = g′ sinθ Mˢ / (Cᵈ W) with hᵖ = Mˢ / (W U), leaves
        # Fr = √(sinθ / Cᵈ) — the width and the transport cancel and the slope alone sets how supercritical the
        # plume is. Taking the Froude number on the upstream layer instead leaves it far below one, switches
        # entrainment off, and the product then outruns every ambient density and piles up on the floor.
        Fr = sqrt(overflow.bottom_slope / overflow.drag_coefficient)
        φ = clamp(overflow.entrainment_factor * (Fr - 1), 0, overflow.maximum_entrainment)

        entrained = (last(band) + 1):nˢ
        Vᴱ = isempty(entrained) ? 0.0 : sum(pv[entrained])
        Mᵉ = Vᴱ > 0 ? φ * Mˢ : 0.0
        Θᴱ = Mᵉ > 0 ? band_mean(Tᵃ, pv, entrained) : 0.0
        Sᴬᴱ = Mᵉ > 0 ? band_mean(Sᵃ, pv, entrained) : 0.0
        Θᵖ = (Mˢ * Θᵈ + Mᵉ * Θᴱ) / max(Mˢ + Mᵉ, eps())
        Sᴬᵖ = (Mˢ * Sᴬᵈ + Mᵉ * Sᴬᴱ) / max(Mˢ + Mᵉ, eps())
    end

    state = overflow.state
    applied = !isempty(band) && Mˢ > 0
    if applied
        # the source increment is written to every source-box cell on the dense levels, so it is drawn from their volume
        sv = vec(sum(source.volumes, dims = (1, 2)))
        Vᵈ = sum(sv[approach.k[dense] .- first(source.k) .+ 1])
        Vᴺ = sum(pv[band])
        Vᴱ = isempty(entrained) ? 0.0 : sum(pv[entrained])
        Θᴺ, Sᴬᴺ = band_mean(Tᵃ, pv, band), band_mean(Sᵃ, pv, band)
        Θᴱ = Vᴱ > 0 ? band_mean(Tᵃ, pv, entrained) : 0.0
        Sᴬᴱ = Vᴱ > 0 ? band_mean(Sᵃ, pv, entrained) : 0.0
        cˢ, cᵖ = 2 * (overflow.code - 1) + 1, 2 * (overflow.code - 1) + 2

        for (column, cᵈ, cᴺ, cᴱ) in ((1, Θᵈ, Θᴺ, Θᴱ), (2, Sᴬᵈ, Sᴬᴺ, Sᴬᴱ))
            for n in dense
                increments[approach.k[n], cˢ, column] = Mˢ * (cᴺ - cᵈ) / Vᵈ
            end
            for n in band
                increments[plume.k[n], cᵖ, column] = (Mˢ * cᵈ + Mᵉ * cᴱ - (Mˢ + Mᵉ) * cᴺ) / Vᴺ
            end
            if Vᴱ > 0 && Mᵉ > 0
                for n in entrained
                    increments[plume.k[n], cᵖ, column] = Mᵉ * (cᴺ - cᴱ) / Vᴱ
                end
            end
        end
        state.neutral_depth = -overflow.level_depths[plume.k[first(band)]]
    end

    state.hydraulic_transport = Mʷ
    state.resolved_descent = Mʳ
    state.source_transport = applied ? Mˢ : zero(Mˢ)
    state.entrainment_transport = applied ? Mᵉ : zero(Mᵉ)
    state.froude_number = Fr
    state.upstream_height = hᵘ
    state.reduced_gravity = g′
    state.product_temperature = Θᵖ
    state.product_salinity = Sᴬᵖ
    state.applied = applied

    return nothing
end

function update_sill_overflow!(simulation, set::SillOverflowSet)
    model = simulation.model.ocean.model
    T, S, w = model.tracers.T, model.tracers.S, model.velocities.w
    increments = zeros(eltype(set.increments), size(set.increments))
    for overflow in set.sills
        update_sill_overflow!(overflow, increments, T, S, w)
        s = overflow.state
        @info @sprintf("SILL OVERFLOW %-19s M_w=%.3f M_r=%.3f M_s=%.3f M_e=%.3f Sv Fr=%.2f h_u=%.0f m g′=%.2e Θ_p=%.2f Sᴬ_p=%.3f z_n=%.0f m applied=%s",
                       overflow.name, s.hydraulic_transport / 1e6, s.resolved_descent / 1e6,
                       s.source_transport / 1e6, s.entrainment_transport / 1e6, s.froude_number,
                       s.upstream_height, s.reduced_gravity, s.product_temperature, s.product_salinity,
                       s.neutral_depth, s.applied)
    end
    copyto!(set.increments, increments)
    return nothing
end

struct SillOverflowUpdate{O}
    set :: O
end

(u::SillOverflowUpdate)(simulation) = update_sill_overflow!(simulation, u.set)

@inline function sill_overflow_tendency(i, j, k, grid, clock, fields, p)
    @inbounds begin
        r = p.region_codes[i, j, k]
        G = p.increments[k, max(Int(r), 1), p.column]
    end
    return ifelse(r == 0, zero(G), G)
end

"""
    sill_overflow_forcing(grid, enabled; configurations = default_sill_configurations())

Return `((T, S) forcings, set)`, or `(NamedTuple(), nothing)` when `enabled` is `false`. The caller registers
`SillOverflowUpdate(set)` as a callback.
"""
function sill_overflow_forcing(grid, enabled; configurations = default_sill_configurations())
    enabled || return NamedTuple(), nothing

    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    cpu_grid = on_architecture(CPU(), grid)
    codes = zeros(FT, Nx, Ny, Nz)
    sills = []
    for (s, cfg) in enumerate(configurations)
        overflow, (source_cells, plume_cells) = SillOverflow(grid, s, cfg)
        for (r, cells) in ((1, source_cells), (2, plume_cells)), c in cells
            codes[c] == 0 || throw(ArgumentError("sill overflow regions overlap at $c"))
            codes[c] = 2 * (s - 1) + r
        end
        push!(sills, overflow)
    end

    region_codes = Field{Center, Center, Center}(grid)
    set!(region_codes, codes)
    fill_halo_regions!(region_codes)

    increments = on_architecture(architecture(grid), zeros(FT, Nz, 2 * length(configurations), 2))
    set = SillOverflowSet(Tuple(sills), region_codes, increments)

    parameters(column) = (; region_codes, increments, column)
    T = Forcing(sill_overflow_tendency; discrete_form = true, parameters = parameters(1))
    S = Forcing(sill_overflow_tendency; discrete_form = true, parameters = parameters(2))
    return (; T, S), set
end
