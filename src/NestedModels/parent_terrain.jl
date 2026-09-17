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
using Oceananigans.DistributedComputations: DistributedGrid, concatenate_local_sizes
using Oceananigans.Utils: launch!

@kernel function _blend_parent_terrain!(elevation, parent_elevation, width, i₀, j₀, Nx, Ny)
    i, j = @index(Global, NTuple)
    I = i + i₀
    J = j + j₀
    @inbounds begin
        w = clamp(min(I - 1, Nx - I, J - 1, Ny - J) / width, 0, 1)
        elevation[i, j, 1] = w * elevation[i, j, 1] + (1 - w) * parent_elevation[i, j, 1]
    end
end

# Offset of this rank's first cell within the global domain, and the global horizontal size. The
# blend frame follows the *domain* edge, so both must be global: `size(grid)` is rank-local under a
# `Partition`, and using it would ring every rank's own subdomain instead.
blend_frame(grid) = (0, 0, size(grid, 1), size(grid, 2))

function blend_frame(grid::DistributedGrid)
    arch = architecture(grid)
    nx = concatenate_local_sizes(size(grid), arch, 1)
    ny = concatenate_local_sizes(size(grid), arch, 2)
    rx, ry = arch.local_index[1], arch.local_index[2]
    return sum(nx[1:rx-1]), sum(ny[1:ry-1]), sum(nx), sum(ny)
end

"""
    blend_parent_terrain!(elevation, parent_elevation; width)

Blend the child `elevation` (a two-dimensional field) toward `parent_elevation` (the parent's
surface elevation on the same grid) over the outermost `width` cells: the blend weight ramps
linearly from the parent's elevation at the boundary to the child's `width` cells inward, so the
terrain at the open boundaries matches the orography the parent state was produced with.

Distance to the boundary is measured in global indices, so under a `Partition` only the ranks
holding a true domain edge blend, and interior rank seams are left untouched.
"""
function blend_parent_terrain!(elevation, parent_elevation; width)
    grid = elevation.grid
    i₀, j₀, Nx, Ny = blend_frame(grid)
    launch!(architecture(grid), grid, :xy, _blend_parent_terrain!,
            elevation, parent_elevation, convert(eltype(grid), width), i₀, j₀, Nx, Ny)
    return elevation
end
