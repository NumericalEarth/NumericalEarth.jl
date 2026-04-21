using Oceananigans
using Oceananigans.Grids: znode

# Scalar-extraction functions used by both the simulation's `:averages` writer
# (via `omip_diagnostics.jl`) and the EN4 proxy driver. Each returns a lazy
# Oceananigans `Reduction`.

# `view(T, :, :, Nz)` + `Average` does not normalize correctly — Average
# assumes a 3-D volume even when a view has size=1 in z. Using a `condition`
# that selects k == Nz keeps all the metric logic intact.
@inline _top_layer(Nz) = (i, j, k, grid, args...) -> k == Nz
@inline _combine(c1, c2) = (i, j, k, grid, args...) ->
    c1(i, j, k, grid, args...) & c2(i, j, k, grid, args...)

global_volume_T(T; hemisphere = global_ocean) =
    Average(T; condition = hemisphere)

function global_surface_T(T; hemisphere = global_ocean)
    Nz = size(T, 3)
    return Average(T; condition = _combine(_top_layer(Nz), hemisphere))
end

function global_surface_S(S; hemisphere = global_ocean)
    Nz = size(S, 3)
    return Average(S; condition = _combine(_top_layer(Nz), hemisphere))
end

@inline in_top_300m(i, j, k, grid, args...) =
    znode(i, j, k, grid, Center(), Center(), Center()) > -300

function global_ohc_300(T, ρ, cp; hemisphere = global_ocean)
    cond = _combine(in_top_300m, hemisphere)
    return Integral(ρ * cp * T; condition = cond)
end
