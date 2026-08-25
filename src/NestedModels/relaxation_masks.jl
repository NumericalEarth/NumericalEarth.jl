#####
##### Relaxation-zone masks shared by nested children
#####
#
# Ramp shapes are isbits callables giving the nudging weight against the normalized distance from a
# wall, s ∈ [0, 1]. The contract is ramp(0) = 1, ramp(1) = 0, monotone in between.

struct CosineRamp end
struct SmoothStepRamp end

@inline (::CosineRamp)(s)     = (1 + cos(π * s)) / 2
@inline (::SmoothStepRamp)(s) = 1 - s^2 * (3 - 2s)

"""
$(TYPEDSIGNATURES)

Return a callable `(x, y, z) -> [0, 1]` that is 1 at the lateral walls of `grid` and ramps to 0 over
the outermost `width` cells, following `ramp`. This is the Davies relaxation zone of a nested child:
pass it as the `mask` of [`parent_forcings`](@ref) so the interior nudging toward the parent acts only
near the open boundaries.
"""
function davies_relaxation_mask(grid, width; ramp = CosineRamp())
    x₁, x₂ = extrema(xnodes(grid, Face(), Center(), Center()))
    y₁, y₂ = extrema(ynodes(grid, Center(), Face(), Center()))
    Nx, Ny, _ = size(grid)
    w = width * max((x₂ - x₁) / Nx, (y₂ - y₁) / Ny)

    return (x, y, z) -> begin
        d = min(x - x₁, x₂ - x, y - y₁, y₂ - y)
        s = clamp(d / w, zero(d), one(d))
        return oftype(d, ramp(s))
    end
end
