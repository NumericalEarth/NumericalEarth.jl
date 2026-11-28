using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans: Center
using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Utils: getregion

#####
##### Exchange weights between a Yin-Yang grid and the cell centers of another grid (e.g. a tripolar ocean).
##### Weights are computed once on the host; applying them is a sparse matrix-vector product.
##### Panel values are indexed as vcat(vec(panel 1), vec(panel 2)).
#####

unit_vector(λ, φ) = (cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ))
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

panel_centers(yin_yang_grid, r) = begin
    panel = on_architecture(CPU(), getregion(yin_yang_grid, r))
    Nx, Ny = size(panel, 1), size(panel, 2)
    λ = [λnode(i, j, 1, panel, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]
    φ = [φnode(i, j, 1, panel, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]
    (λ, φ)
end

# Latitude-longitude buckets of width B for the nearest-center search; the longitude search widens as 1 / cos φ,
# to the whole row near the poles (Yang polar caps, tripolar north fold).
struct CenterBuckets
    width :: Float64
    buckets :: Matrix{Vector{Int}}
end

function CenterBuckets(λ, φ, width)
    Nλ, Nφ = ceil(Int, 360 / width), ceil(Int, 180 / width)
    buckets = [Int[] for _ in 1:Nλ, _ in 1:Nφ]
    for n in eachindex(λ)
        push!(buckets[bucket_index(width, Nλ, Nφ, λ[n], φ[n])...], n)
    end
    return CenterBuckets(width, buckets)
end

bucket_index(width, Nλ, Nφ, λ, φ) = (mod(floor(Int, mod(λ + 180, 360) / width), Nλ) + 1,
                                     clamp(floor(Int, (φ + 90) / width) + 1, 1, Nφ))

function nearest_center(b::CenterBuckets, centers, λ, φ)
    Nλ, Nφ = size(b.buckets)
    q = unit_vector(λ, φ)
    bi, bj = bucket_index(b.width, Nλ, Nφ, λ, φ)
    φedge = min(abs(φ) + 2b.width, 90)
    half = φedge > 90 - 4b.width ? Nλ : ceil(Int, 1 / cosd(φedge)) + 1
    di = 2half + 1 ≥ Nλ ? (0:Nλ-1) : (-half:half)
    nearest, c★ = 0, -2.0
    for dj in -1:1, d in di
        1 ≤ bj + dj ≤ Nφ || continue
        for n in b.buckets[mod1(bi + d, Nλ), bj + dj]
            c = dot3(q, centers[n])
            c > c★ && ((nearest, c★) = (n, c))
        end
    end
    return nearest, acos(clamp(c★, -1, 1))
end

#####
##### Yin-Yang -> points: bilinear interpolation, 4 weights per point
#####

# Gnomonic projection of the unit vector P onto the plane tangent at q, with tangent basis (e₁, e₂)
gnomonic(P, q, e₁, e₂) = (dot3(P, e₁) / dot3(P, q), dot3(P, e₂) / dot3(P, q))

# (s, t) of the origin in the bilinear quad with corners c₁₁, c₂₁, c₁₂, c₂₂ (Newton)
function inverse_bilinear(c₁₁, c₂₁, c₁₂, c₂₂)
    s, t = 0.5, 0.5
    for _ in 1:20
        x = (1-s) * (1-t) * c₁₁[1] + s * (1-t) * c₂₁[1] + (1-s) * t * c₁₂[1] + s * t * c₂₂[1]
        y = (1-s) * (1-t) * c₁₁[2] + s * (1-t) * c₂₁[2] + (1-s) * t * c₁₂[2] + s * t * c₂₂[2]
        xs = (1-t) * (c₂₁[1] - c₁₁[1]) + t * (c₂₂[1] - c₁₂[1])
        ys = (1-t) * (c₂₁[2] - c₁₁[2]) + t * (c₂₂[2] - c₁₂[2])
        xt = (1-s) * (c₁₂[1] - c₁₁[1]) + s * (c₂₂[1] - c₂₁[1])
        yt = (1-s) * (c₁₂[2] - c₁₁[2]) + s * (c₂₂[2] - c₂₁[2])
        J = xs * yt - xt * ys
        δs, δt = (yt * x - xt * y) / J, (xs * y - ys * x) / J
        s, t = s - δs, t - δt
        abs(δs) + abs(δt) < 1e-13 && break
    end
    return s, t
end

"""
$(TYPEDSIGNATURES)

Bilinear interpolation weights from the cell centers of `yin_yang_grid` to the points `(λ, φ)` (degrees).
Each point is interpolated on the panel where its nearest center lies farthest from the panel edge, in the
quad of centers that contains it. Returns `(indices, weights)`, both `4 × length(λ)`; `indices` index
`vcat(vec(panel₁), vec(panel₂))`.
"""
function yin_yang_bilinear_weights(yin_yang_grid, λ, φ)
    panels = [panel_centers(yin_yang_grid, r) for r in 1:2]
    width = max(180 / size(panels[1][1], 2), 0.5)
    centers = [unit_vector.(λp, φp) for (λp, φp) in panels]
    buckets = [CenterBuckets(λp, φp, width) for (λp, φp) in panels]
    offsets = (0, length(panels[1][1]))
    indices = zeros(Int, 4, length(λ))
    weights = zeros(4, length(λ))

    Threads.@threads for m in eachindex(λ)
        q = unit_vector(λ[m], φ[m])
        e₁ = (-sind(λ[m]), cosd(λ[m]), 0.0)
        e₂ = (-sind(φ[m]) * cosd(λ[m]), -sind(φ[m]) * sind(λ[m]), cosd(φ[m]))

        # Distance (in cells) of each panel's nearest center from that panel's edge
        candidates = map(1:2) do r
            Nx, Ny = size(panels[r][1])
            n = first(nearest_center(buckets[r], centers[r], λ[m], φ[m]))
            n == 0 && return (r, 0, 0, -1)   # the point is outside this panel's search area
            i, j = Tuple(CartesianIndices((Nx, Ny))[n])
            (r, i, j, min(i - 1, Nx - i, j - 1, Ny - j))
        end
        r, i, j, _ = candidates[1][4] ≥ candidates[2][4] ? candidates[1] : candidates[2]
        Nx, Ny = size(panels[r][1])
        L = LinearIndices((Nx, Ny))

        for (i₀, j₀) in ((i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j))
            (1 ≤ i₀ < Nx && 1 ≤ j₀ < Ny) || continue
            corners = (L[i₀, j₀], L[i₀ + 1, j₀], L[i₀, j₀ + 1], L[i₀ + 1, j₀ + 1])
            s, t = inverse_bilinear(map(n -> gnomonic(centers[r][n], q, e₁, e₂), corners)...)
            if -1e-9 ≤ s ≤ 1 + 1e-9 && -1e-9 ≤ t ≤ 1 + 1e-9
                indices[:, m] .= corners .+ offsets[r]
                weights[:, m] .= ((1-s) * (1-t), s * (1-t), (1-s) * t, s * t)
                break
            end
        end
    end

    return indices, weights
end

#####
##### Points -> Yin-Yang: area-weighted average over the wet source cells under each panel cell
#####

"""
$(TYPEDSIGNATURES)

Area-weighted averaging of source cells with centers `(λ, φ)`, areas `area` and mask `wet` onto each panel of
`yin_yang_grid`. A source cell belongs to the panel cell with the nearest center, if that center is within
3/4 of a panel spacing (the cell's half-diagonal); each panel is treated separately, so cells in the panel overlap are averaged on both panels.
Returns, per panel, `(; rows, columns, weights, wet_fraction)`: `weights[n]` multiplies source cell `columns[n]`
for panel cell `rows[n]` (rows of panel cells with no wet source cell are absent), and `wet_fraction` is the
wet area fraction of each panel cell.
"""
function yin_yang_area_average_weights(yin_yang_grid, λ, φ, area, wet)
    return map(1:2) do r
        λp, φp = panel_centers(yin_yang_grid, r)
        spacing = 90 / size(λp, 2)
        centers = unit_vector.(λp, φp)
        buckets = CenterBuckets(λp, φp, max(2spacing, 0.5))

        target = zeros(Int, length(λ))
        Threads.@threads for m in eachindex(λ)
            n, distance = nearest_center(buckets, centers, λ[m], φ[m])
            target[m] = distance < deg2rad(3spacing / 4) ? n : 0   # within the cell's half-diagonal
        end

        wet_area = zeros(length(λp))
        total_area = zeros(length(λp))
        for m in eachindex(target)
            n = target[m]
            n == 0 && continue
            total_area[n] += area[m]
            wet[m] && (wet_area[n] += area[m])
        end

        columns = findall(m -> target[m] > 0 && wet[m], eachindex(target))
        rows = target[columns]
        weights = area[columns] ./ wet_area[rows]
        wet_fraction = reshape(ifelse.(total_area .> 0, wet_area ./ total_area, 0), size(λp))
        (; rows, columns, weights, wet_fraction)
    end
end
