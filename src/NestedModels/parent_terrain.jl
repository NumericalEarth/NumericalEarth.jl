#####
##### blend_parent_terrain!: match the child's terrain to the parent's at the open boundaries
#####
#
# A nested child's lateral boundary conditions interpolate parent state that was produced over the
# parent's (smoother) orography. Blending the child elevation toward the parent's over the outermost
# cells makes the terrain at the open boundaries consistent with that state, avoiding spurious
# boundary-layer flows where the two orographies disagree.

using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!

@kernel function _blend_parent_terrain!(elevation, parent_elevation, width, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        w = clamp(min(i - 1, Nx - i, j - 1, Ny - j) / width, 0, 1)
        elevation[i, j, 1] = w * elevation[i, j, 1] + (1 - w) * parent_elevation[i, j, 1]
    end
end

"""
    blend_parent_terrain!(elevation, parent_elevation; width)

Blend the child `elevation` (a two-dimensional field) toward `parent_elevation` (the parent's
surface elevation on the same grid) over the outermost `width` cells: the blend weight ramps
linearly from the parent's elevation at the boundary to the child's `width` cells inward, so the
terrain at the open boundaries matches the orography the parent state was produced with.
"""
function blend_parent_terrain!(elevation, parent_elevation; width)
    grid = elevation.grid
    Nx, Ny, _ = size(grid)
    launch!(architecture(grid), grid, :xy, _blend_parent_terrain!,
            elevation, parent_elevation, convert(eltype(grid), width), Nx, Ny)
    return elevation
end
