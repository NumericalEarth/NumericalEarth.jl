#####
##### Background tracer diffusivity from observed 1° maps of log₁₀ κ
#####

using Oceananigans.Grids: znodes
using Statistics: median

# Layer midpoints (m) of the three maps: 300–500 m, 500–1000 m and 1000–2000 m.
const OBSERVED_DIFFUSIVITY_LAYERS = (("k_u.txt", 400), ("k_c.txt", 750), ("k_l.txt", 1500))
const OBSERVED_LOG10_DIFFUSIVITY_BOUNDS = (-7, -4)
# Values outside this range are interpolation artifacts of the maps and are treated as missing.
const OBSERVED_LOG10_DIFFUSIVITY_VALID = (-8, -4)
# Poleward of these latitudes the maps are mostly artifacts (outside the observational coverage); there each map takes
# the zonal median of its last covered row.
const OBSERVED_DIFFUSIVITY_LATITUDES = (-62, 70)

read_whitespace_table(path) = permutedims(reduce(hcat, [parse.(Float64, split(line)) for line in eachline(path) if !isempty(strip(line))]))

# Missing cells take the mean of their valid 4-neighbors, sweeping until every cell is filled.
function fill_missing!(a)
    Nx, Ny = size(a)
    while any(isnan, a)
        filled = copy(a)
        for j in 1:Ny, i in 1:Nx
            isnan(a[i, j]) || continue
            neighbors = (a[mod1(i - 1, Nx), j], a[mod1(i + 1, Nx), j], a[i, max(j - 1, 1)], a[i, min(j + 1, Ny)])
            valid = filter(!isnan, neighbors)
            isempty(valid) || (filled[i, j] = sum(valid) / length(valid))
        end
        a .= filled
    end
    return a
end

# Bilinear interpolation on the 1° source grid with cell centers at 0.5°, 1.5°, …: periodic in λ, clamped in φ.
function bilinear(a, λ, φ)
    Nx, Ny = size(a)
    x = mod(λ - 0.5, 360)
    y = clamp(φ + 89.5, 0, Ny - 1)
    i₀, j₀ = floor(Int, x), min(floor(Int, y), Ny - 2)
    ξ, η = x - i₀, y - j₀
    i₁, i₂, j₁, j₂ = mod(i₀, Nx) + 1, mod(i₀ + 1, Nx) + 1, j₀ + 1, j₀ + 2
    return (1 - ξ) * (1 - η) * a[i₁, j₁] + ξ * (1 - η) * a[i₂, j₁] + (1 - ξ) * η * a[i₁, j₂] + ξ * η * a[i₂, j₂]
end

# Smoothstep blend of log₁₀ κ between the layer midpoints, held constant above the first and below the last.
function vertical_profile(depth, depths, values)
    depth <= first(depths) && return first(values)
    depth >= last(depths) && return last(values)
    n = findlast(≤(depth), depths)
    ξ = (depth - depths[n]) / (depths[n + 1] - depths[n])
    return values[n] + (values[n + 1] - values[n]) * ξ^2 * (3 - 2ξ)
end

"""
    observed_background_diffusivity(grid, directory)

Return a `CenterField` of background tracer diffusivity built from the maps of log₁₀(κ / m² s⁻¹) in `directory`
(`k_u.txt`, `k_c.txt`, `k_l.txt` for 300–500, 500–1000 and 1000–2000 m on a 1° grid, rows from 89.5°S). Values outside
`OBSERVED_LOG10_DIFFUSIVITY_VALID` count as missing, poleward of `OBSERVED_DIFFUSIVITY_LATITUDES` each map takes the
zonal median of its last covered row; each map is gap-filled from its neighbors, clipped to
`OBSERVED_LOG10_DIFFUSIVITY_BOUNDS` and interpolated bilinearly to the grid; log₁₀ κ then varies smoothly in depth
between the layer midpoints.
"""
function observed_background_diffusivity(grid, directory)
    maps = map(OBSERVED_DIFFUSIVITY_LAYERS) do (file, _)
        a = permutedims(read_whitespace_table(joinpath(directory, file)))   # (λ, φ)
        φ = -89.5 .+ (0:size(a, 2) - 1)'
        covered = OBSERVED_DIFFUSIVITY_LATITUDES[1] .< φ .< OBSERVED_DIFFUSIVITY_LATITUDES[2]
        a = ifelse.(covered .& (OBSERVED_LOG10_DIFFUSIVITY_VALID[1] .<= a .<= OBSERVED_LOG10_DIFFUSIVITY_VALID[2]), a, NaN)
        rows = findall(vec(covered))
        south, north = first(rows), last(rows)
        a[:, 1:south-1] .= median(filter(!isnan, a[:, south]))
        a[:, north+1:end] .= median(filter(!isnan, a[:, north]))
        clamp.(fill_missing!(a), OBSERVED_LOG10_DIFFUSIVITY_BOUNDS...)
    end
    depths = Float64[d for (_, d) in OBSERVED_DIFFUSIVITY_LAYERS]

    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(cpu_grid)
    z = znodes(cpu_grid, Center())
    κ = zeros(Nx, Ny, Nz)
    for j in 1:Ny, i in 1:Nx
        λ = λnode(i, j, 1, cpu_grid, Center(), Center(), Center())
        φ = φnode(i, j, 1, cpu_grid, Center(), Center(), Center())
        values = Float64[bilinear(a, λ, φ) for a in maps]
        for k in 1:Nz
            κ[i, j, k] = 10^vertical_profile(-z[k], depths, values)
        end
    end

    field = CenterField(grid)
    set!(field, κ)
    fill_halo_regions!(field)
    return field
end
