using Oceananigans.OrthogonalSphericalShellGrids: TripolarGridOfSomeKind
using Oceananigans.ImmersedBoundaries: bottom_height_field
using Oceananigans.Fields: convert_to_0_360
using ..DataWrangling: BoundingBox

#####
##### Barriers
#####

"""
$(TYPEDSIGNATURES)

Return a narrow meridional `BoundingBox` centered at `longitude`, with zonal width `width` degrees and
meridional extent `[south, north]`, which closes straits such as Cape Agulhas or the Indonesian passages.
"""
meridional_barrier(longitude, south, north; width=2) = BoundingBox(longitude=(longitude - width/2, longitude + width/2), latitude=(south, north))

add_barrier(barriers, barrier) = isnothing(barriers) ? [barrier] : vcat(barriers, barrier)

# A barrier's vertical extent is ignored: the labeling operates on the two-dimensional bathymetry.
@kernel function _apply_barrier!(zb, grid, barrier::BoundingBox)
    i, j = @index(Global, NTuple)

    inside_longitude = if isnothing(barrier.longitude) || (barrier.longitude[2] - barrier.longitude[1] >= 360)
        true
    else
        λ = convert_to_0_360(λnode(i, j, 1, grid, Center(), Center(), Center()))
        convert_to_0_360(barrier.longitude[1]) <= λ <= convert_to_0_360(barrier.longitude[2])
    end

    inside_latitude = if isnothing(barrier.latitude)
        true
    else
        φ = φnode(i, j, 1, grid, Center(), Center(), Center())
        barrier.latitude[1] <= φ <= barrier.latitude[2]
    end

    @inbounds zb[i, j, 1] = ifelse(inside_longitude & inside_latitude, zero(grid), zb[i, j, 1])
end

#####
##### Connected component labeling
#####

# Cells (1, j) and (Nx, j) are neighbors across the periodic longitude seam.
function enforce_periodic_labels!(labels, ::Periodic)
    Nx, Ny = size(labels)

    for j in 1:Ny
        west = labels[1, j]
        east = labels[Nx, j]

        if west != 0 && east != 0 && west != east
            replace!(labels, east => west)
        end
    end

    return labels
end

enforce_periodic_labels!(labels, tx) = labels

# Cells (i, Ny) and (Nx-i+1, Ny) are neighbors across the tripolar fold.
function enforce_tripolar_labels!(labels, ::TripolarGridOfSomeKind)
    Nx, Ny = size(labels)

    for i in 1:Nx÷2
        label = labels[i, Ny]
        folded_label = labels[Nx-i+1, Ny]

        if label != 0 && folded_label != 0 && label != folded_label
            replace!(labels, folded_label => label)
        end
    end

    return labels
end

enforce_tripolar_labels!(labels, grid) = labels

"""
$(TYPEDSIGNATURES)

Label the connected water regions of the bottom height `zb`, returning a matrix carrying a unique integer per
basin and zero on land. Useful for inpainting the bathymetry and for computing the masks of oceanic basins.

Keyword Arguments
=================
- `barriers`: a vector of `BoundingBox`es marked as land before labeling, so that basins connected in the
              bathymetry are separated (closing Drake Passage separates the Atlantic from the Pacific).
"""
function label_ocean_basins(zb::Field; barriers=nothing)

    # The labeling is two-dimensional and serial, so it is performed on the CPU
    zb = on_architecture(CPU(), zb)
    grid = zb.grid

    if !isnothing(barriers)
        zb = set!(similar(zb), zb) # barriers are applied to a copy, `set!` returns its destination

        for barrier in barriers
            launch!(CPU(), grid, :xy, _apply_barrier!, zb, grid, barrier)
        end
    end

    water = interior(zb, :, :, 1) .< 0
    labels = ImageMorphology.label_components(ImageMorphology.strel(water))

    # `label_components` connects cells that share a face within the array, so cells that are neighbors across
    # the periodic longitude seam or across the tripolar fold are merged afterwards
    enforce_periodic_labels!(labels, topology(grid, 1)())
    enforce_tripolar_labels!(labels, grid)

    return labels
end

label_ocean_basins(grid::ImmersedBoundaryGrid; barriers=nothing) = label_ocean_basins(bottom_height_field(grid); barriers)
