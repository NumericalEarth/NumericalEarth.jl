using Oceananigans.OrthogonalSphericalShellGrids: TripolarGridOfSomeKind
using Oceananigans.Fields: convert_to_0_360
using ..DataWrangling: BoundingBox

#####
##### Barrier application functions
#####

# A barrier is a `BoundingBox` whose horizontal extent defines a rectangular
# region to be temporarily marked as land during connected-component labeling.
# The `z` field of the `BoundingBox` is ignored: the flood-fill operates on
# the 2D bathymetry slice, and barriers act at every depth.

"""
    meridional_barrier(longitude, south, north; width=2)

Create a narrow meridional `BoundingBox` centered at `longitude` with meridional
extent `[south, north]` and zonal width `width` degrees. Useful for closing
straits like Cape Agulhas or Indonesian passages during basin labeling.
"""
meridional_barrier(longitude, south, north; width=2) = BoundingBox(longitude=(longitude - width/2, longitude + width/2), latitude=(south, north))

"""
    apply_barrier!(zb, grid, barrier::BoundingBox)

Mark all cells within the barrier's horizontal region as land (z = 0).
"""
apply_barrier!(zb, grid, barrier::BoundingBox) = launch!(architecture(grid), grid, :xy, _apply_barrier!, zb, grid, barrier)

apply_barrier!(zb, grid, barriers::Nothing) = zb

function apply_barrier!(zb, grid, barriers::AbstractVector)
    for barrier in barriers
        apply_barrier!(zb, grid, barrier)
    end
    return zb
end

@kernel function _apply_barrier!(zb, grid, barrier::BoundingBox)
    i, j = @index(Global, NTuple)

    in_lon = if isnothing(barrier.longitude) || (barrier.longitude[2] - barrier.longitude[1] >= 360)
        true
    else
        bw = convert_to_0_360(barrier.longitude[1])
        be = convert_to_0_360(barrier.longitude[2])
        λ = λnode(i, j, 1, grid, Center(), Center(), Center())
        λ = convert_to_0_360(λ)
        (bw <= λ <= be)
    end

    in_lat = if isnothing(barrier.latitude)
        true
    else
        φ = φnode(i, j, 1, grid, Center(), Center(), Center())
        barrier.latitude[1] <= φ <= barrier.latitude[2]
    end

    @inbounds zb[i, j, 1] = ifelse(in_lon & in_lat, zero(grid), zb[i, j, 1])
end

# Since the strel algorithm in `label_components` does not recognize periodic boundaries,
# before labeling connected regions we copy half the longitude extent on each side so that
# water cells near the boundary are correctly identified as connected.
function copy_periodic_longitude(zb_cpu::Field, ::Periodic)
    Nλ = size(zb_cpu, 1)
    zb_data   = zb_cpu.data[1:Nλ, :, 1]
    zb_parent = zb_data.parent

    # Concatenate a copy of the eastern half on the left and the western half on the right.
    # This is an O(Nλ × Nφ) CPU allocation, acceptable for the serial labeling step.
    half = Nλ ÷ 2
    zb_parent = vcat(zb_parent[half:Nλ, :], zb_parent, zb_parent[1:half, :])

    # Update offsets so that index 1 maps back to the original domain start
    yoffsets = zb_cpu.data.offsets[2]
    xoffsets = - half - 1

    return OffsetArray(zb_parent, xoffsets, yoffsets)
end

copy_periodic_longitude(zb_cpu::Field, tx) = interior(zb_cpu, :, :, 1)

"""
    label_ocean_basins(zb_field, TX, core_size)

Creates a matrix with a unique label for each connected basin. Useful for inpainting the bathymetry and
computing the masks for oceanic basins.

Handles periodic boundary extension internally and returns labels for the core region only.
"""
function label_ocean_basins(zb_field, TX, size)
    zb = copy_periodic_longitude(zb_field, TX()) # Outputs a 2D AbstractArray

    water = zb .< 0

    connectivity = ImageMorphology.strel(water)
    labels = ImageMorphology.label_components(connectivity)

    Nx, Ny = size[1], size[2]
    core_labels = labels[1:Nx, 1:Ny]

    # Enforce periodicity: merge labels that should be connected across
    # the periodic boundaries. This handles cases where a barrier (e.g., blocking
    # the Southern Ocean) prevents the extended domain from connecting basins
    # that are actually periodic neighbors.
    enforce_periodic_labels!(core_labels, TX())
    enforce_tripolar_labels!(core_labels, zb_field.grid)

    return core_labels
end

"""
    enforce_periodic_labels!(labels, ::Periodic)

Merge labels that should be connected due to periodic boundary conditions.
For each latitude, if the westernmost and easternmost cells are both water
(non-zero labels), they are periodic neighbors and must have the same label.
"""
function enforce_periodic_labels!(labels, ::Periodic)
    Nx, Ny = Base.size(labels)

    for j in 1:Ny
        label_west = labels[1, j]
        label_east = labels[Nx, j]

        # Both cells are water and have different labels: merge them
        if label_west != 0 && label_east != 0 && label_west != label_east
            # Replace all occurrences of label_east with label_west
            replace!(labels, label_east => label_west)
        end
    end

    return labels
end

# No-op for non-periodic domains
enforce_periodic_labels!(labels, tx) = labels

"""
    enforce_tripolar_labels!(labels, ::TripolarGridOfSomeKind)

Merge labels that should be connected due to zipper boundary conditions.
For each longitude, if cells reflected across the fold (Nx÷2) are water
(non-zero labels), they are periodic neighbors and must have the same label.
"""
function enforce_tripolar_labels!(labels, ::TripolarGridOfSomeKind)
    Nx, Ny = Base.size(labels)

    for i in 1:Nx÷2
        label_west = labels[i,      Ny]
        label_east = labels[Nx-i+1, Ny]

        # Both cells are water and have different labels: merge them
        if label_west != 0 && label_east != 0 && label_west != label_east
            # Replace all occurrences of label_east with label_west
            replace!(labels, label_east => label_west)
        end
    end

    return labels
end

# No-op for non-tripolar grids
enforce_tripolar_labels!(labels, grid) = labels

# Utilities to label ocean basins passing only the grid
function label_ocean_basins(grid::AbstractGrid; barriers=nothing)
    @warn "The grid is not immersed, there is only one ocean basin!"
    Nx, Ny, Nz = size(grid)
    return zeros(Int, Nx, Ny)
end

"""
    label_ocean_basins(grid::ImmersedBoundaryGrid; barriers=nothing)

Label connected ocean basins in an ImmersedBoundaryGrid.

Keyword Arguments
=================
- `barriers`: Collection of `BoundingBox`es applied before labeling. Each barrier
              temporarily marks its horizontal rectangle as land, allowing separation
              of connected ocean basins (e.g., separating Atlantic from Pacific via
              the Southern Ocean).
"""
function label_ocean_basins(grid::ImmersedBoundaryGrid; barriers=nothing)

    # The labelling algorithm works only on CPUs
    cpu_grid = on_architecture(CPU(), grid)

    TX = topology(cpu_grid, 1)
    zb = cpu_grid.immersed_boundary.bottom_height

    # If barriers are specified, apply them to a copy of the bathymetry
    if !isnothing(barriers)
        # Create a temporary field with the modified bathymetry
        zb_modified = Field{Center, Center, Nothing}(cpu_grid)
        parent(zb_modified) .= parent(zb)
        apply_barrier!(zb_modified, cpu_grid, barriers)
    else
        zb_modified = zb
    end

    return label_ocean_basins(zb_modified, TX, size(cpu_grid))
end
