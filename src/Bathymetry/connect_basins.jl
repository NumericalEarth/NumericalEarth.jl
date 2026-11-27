function nearest_cell(grid, (λ₀, φ₀))
    function squared_angular_distance((i, j))
        λ = λnode(i, j, 1, grid, Center(), Center(), Center())
        φ = φnode(i, j, 1, grid, Center(), Center(), Center())
        Δλ = mod(λ - λ₀ + 180, 360) - 180
        return (Δλ * cosd(φ₀))^2 + (φ - φ₀)^2
    end

    return argmin(squared_angular_distance, Iterators.product(1:size(grid, 1), 1:size(grid, 2)))
end

"""
$(TYPEDSIGNATURES)

Ensure that the ocean cells nearest `from = (λ, φ)` and `to` belong to the same basin: the same region of
wet cells (`bottom_height < 0`) connected through shared faces, since diagonal contact carries no flow on a
C grid. If they belong to different basins, the cells along the path between them that needs the least total
deepening are deepened to `depth`. The path is searched within the index box spanned by the two cells, padded
by its own size, so it stays local to the passage and does not depend on resolution.

Basins are identified in the grid's index space, so the connection between `from` and `to` must not cross a
periodic or folded boundary. Pass the bottom height of the global grid, not that of a distributed rank.

Returns the number of deepened cells.

```jldoctest
julia> using NumericalEarth, Oceananigans

julia> grid = LatitudeLongitudeGrid(size = (5, 3), longitude = (0, 5), latitude = (0, 3), topology = (Bounded, Bounded, Flat));

julia> bottom_height = Field{Center, Center, Nothing}(grid);

julia> set!(bottom_height, (λ, φ) -> 2 < λ < 3 ? 10 : -500);

julia> connect_basins!(bottom_height, (0.5, 1.5), (4.5, 1.5); depth = 300)
1

julia> Array(interior(bottom_height, :, 2, 1))
5-element Vector{Float64}:
 -500.0
 -500.0
 -300.0
 -500.0
 -500.0
```
"""
function connect_basins!(bottom_height, from, to; depth)
    bottom_height_cpu = on_architecture(CPU(), bottom_height)
    grid = bottom_height_cpu.grid
    z = view(interior(bottom_height_cpu), :, :, 1)

    i₁, j₁ = nearest_cell(grid, from)
    i₂, j₂ = nearest_cell(grid, to)

    basins = ImageMorphology.label_components(z .< 0)
    basins[i₁, j₁] == basins[i₂, j₂] != 0 && return 0

    pad = max(2, abs(i₂ - i₁), abs(j₂ - j₁))
    is = max(1, min(i₁, i₂) - pad):min(size(grid, 1), max(i₁, i₂) + pad)
    js = max(1, min(j₁, j₂) - pad):min(size(grid, 2), max(j₁, j₂) + pad)
    window = view(z, is, js)

    # Least-deepening path by Bellman-Ford relaxation: the cost of entering a cell is the deepening it needs.
    deepening = @. max(0, window + depth)
    start = CartesianIndex(i₁ - first(is) + 1, j₁ - first(js) + 1)
    stop  = CartesianIndex(i₂ - first(is) + 1, j₂ - first(js) + 1)
    cost = fill(Inf, size(window))
    previous = fill(start, size(window))
    cost[start] = deepening[start]
    faces = (CartesianIndex(1, 0), CartesianIndex(-1, 0), CartesianIndex(0, 1), CartesianIndex(0, -1))

    relaxed = true
    while relaxed
        relaxed = false
        for c in CartesianIndices(cost), d in faces
            n = c + d
            checkbounds(Bool, cost, n) || continue
            if cost[c] + deepening[n] < cost[n]
                cost[n] = cost[c] + deepening[n]
                previous[n] = c
                relaxed = true
            end
        end
    end

    deepened = 0
    c = stop
    while true
        if deepening[c] > 0
            window[c] = -depth
            deepened += 1
        end
        c == start && break
        c = previous[c]
    end

    set!(bottom_height, interior(bottom_height_cpu))

    return deepened
end
