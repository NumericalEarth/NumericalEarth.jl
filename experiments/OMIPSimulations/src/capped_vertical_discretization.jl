using Oceananigans.Grids: MutableVerticalDiscretization

"""
    capped_geometric_vertical_faces(number_of_levels, depth; surface_grid_size, maximum_grid_size)

Vertical cell interfaces spanning `[-depth, 0]` with `number_of_levels` cells whose spacing grows
geometrically from `surface_grid_size` at the free surface and is then held at `maximum_grid_size`.

The single-parameter `ExponentialDiscretization` used elsewhere pins the surface spacing and pays for
it in the abyss: `Nz = 70`, `Δz_top = 1.5 m` gives 138 m at 1500 m and 400 m at 4900 m, so a
200 m overflow plume occupies one to two cells over the whole depth range where Nordic Seas water has
to keep its density. Capping the spacing decouples the two ends.

The growth ratio is found by bisection so that the spacings sum to `depth` exactly. A cap is reachable
only if `number_of_levels * maximum_grid_size` exceeds `depth` by enough to also pay for the near-surface
cells that sit below the cap.

Returns interfaces in ascending order, from `-depth` to `0`.
"""
function capped_geometric_vertical_faces(number_of_levels, depth;
                                         surface_grid_size,
                                         maximum_grid_size,
                                         tolerance = 1e-10,
                                         maximum_iterations = 200)

    surface_grid_size < maximum_grid_size ||
        throw(ArgumentError("surface_grid_size = $surface_grid_size must be < maximum_grid_size = $maximum_grid_size"))
    number_of_levels * maximum_grid_size > depth ||
        throw(ArgumentError("$number_of_levels levels capped at $maximum_grid_size m cannot span $depth m"))

    spacings(ratio) = [min(maximum_grid_size, surface_grid_size * ratio^(n - 1)) for n in 1:number_of_levels]
    spanned(ratio)  = sum(spacings(ratio))

    spanned(1) < depth ||
        throw(ArgumentError("$number_of_levels levels of $surface_grid_size m already exceed $depth m"))

    lower = 1.0
    upper = 2.0
    while spanned(upper) < depth
        upper *= 2
        upper > 1e6 && throw(ArgumentError("no growth ratio spans $depth m with these constraints"))
    end

    ratio = upper
    for _ in 1:maximum_iterations
        ratio = (lower + upper) / 2
        total = spanned(ratio)
        abs(total - depth) <= tolerance * depth && break
        total < depth ? (lower = ratio) : (upper = ratio)
    end

    Δz = spacings(ratio)
    Δz .*= depth / sum(Δz)

    faces = zeros(number_of_levels + 1)
    for n in 1:number_of_levels
        faces[n + 1] = faces[n] - Δz[n]
    end

    return reverse(faces)
end

"""
    omip_vertical_discretization(number_of_levels, depth; surface_grid_size, maximum_grid_size)

The vertical discretization used by [`build_grid`](@ref): the single-parameter exponential when
`maximum_grid_size` is `nothing`, and [`capped_geometric_vertical_faces`](@ref) otherwise.
"""
function omip_vertical_discretization(number_of_levels, depth; surface_grid_size, maximum_grid_size)
    if isnothing(maximum_grid_size)
        scale = exponential_scale(number_of_levels, depth, surface_grid_size)
        return ExponentialDiscretization(number_of_levels, -depth, 0; scale, mutable = true)
    end

    Δz_top = isnothing(surface_grid_size) ? depth / number_of_levels / 10 : surface_grid_size
    faces = capped_geometric_vertical_faces(number_of_levels, depth;
                                            surface_grid_size = Δz_top,
                                            maximum_grid_size)

    return MutableVerticalDiscretization(faces)
end
