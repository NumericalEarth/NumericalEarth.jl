#####
##### Geometry of the cells of one region: a Yin-Yang panel or a latitude-longitude grid.
##### Unit vectors are stored as Cartesian components in the geographic frame, whose z-axis is the
##### rotation axis and whose x-axis crosses the equator at longitude 0.
#####

struct RegionGeometry{A, V, F, B}
    area :: A         # owned area of each cell (m²): the cell area times its owned fraction
    radial :: V       # (x, y, z) components of the radial unit vector at (Center, Center)
    east :: V         # (x, y, z) components of the region's x-direction unit vector at (Center, Center)
    north :: V        # (x, y, z) components of the region's y-direction unit vector at (Center, Center)
    x_face_east :: F  # the x-direction unit vector at (Face, Center), or `nothing`
    x_face_north :: F # the y-direction unit vector at (Face, Center), or `nothing`
    y_face_east :: F  # the x-direction unit vector at (Center, Face), or `nothing`
    y_face_north :: F # the y-direction unit vector at (Center, Face), or `nothing`
    x_face_area :: B  # owned area at (Face, Center): the mean of the two adjacent cells', or `nothing`
    y_face_area :: B  # owned area at (Center, Face): the mean of the two adjacent cells', or `nothing`
    workspace :: A    # integrands, summed pairwise
end

geometry_fields(geometry::RegionGeometry) = map(name -> getfield(geometry, name), fieldnames(RegionGeometry))

Adapt.adapt_structure(to, geometry::RegionGeometry) = RegionGeometry(map(field -> adapt(to, field), geometry_fields(geometry))...)

move_to_architecture(arch, geometry::RegionGeometry) =
    RegionGeometry(map(field -> move_to_architecture(arch, field), geometry_fields(geometry))...)

move_to_architecture(arch, ::Nothing) = nothing
move_to_architecture(arch, array::AbstractArray) = on_architecture(arch, array)
move_to_architecture(arch, arrays::Tuple) = map(array -> move_to_architecture(arch, array), arrays)

@inline cartesian_unit_vector(λ, φ) = SVector(cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ))

@inline cross_product(a, b) = SVector(a[2] * b[3] - a[3] * b[2],
                                      a[3] * b[1] - a[1] * b[3],
                                      a[1] * b[2] - a[2] * b[1])

# The geographic coordinates (degrees) of the north pole of the frame in which a grid's x and y directions
# are east and north. A Yin-Yang panel is a `RotatedLatitudeLongitudeGrid`: Yin's pole is the geographic
# pole and Yang's lies on the equator.
frame_north_pole(::LatitudeLongitudeGrid) = (0, 90)
frame_north_pole(grid::RotatedLatitudeLongitudeGrid) = grid.conformal_mapping.north_pole

"""
$(SIGNATURES)

Return the radial unit vector at the geographic point `(λ, φ)` (degrees) and the east and north unit
vectors there of the frame whose north pole is at `pole = (λ₀, φ₀)`, as Cartesian 3-vectors in the
geographic frame. The basis is right-handed, `east × north = radial`, and is the basis in which a
`RotatedLatitudeLongitudeGrid` with that pole stores its horizontal vector components, so it is consistent
with the rotation that `fill_halo_regions!((u, v))` applies between Yin-Yang panels. Unlike the geographic
east and north, it is smooth where a panel crosses a geographic pole.
"""
function frame_unit_vectors(pole, λ, φ)
    radial = cartesian_unit_vector(λ, φ)
    eastward = cross_product(cartesian_unit_vector(pole...), radial)
    east = eastward / sqrt(sum(abs2, eastward))
    north = cross_product(radial, east)
    return radial, east, north
end

component_arrays(FT, vectors) = ntuple(d -> reshape(FT[v[d] for v in vectors], size(vectors)..., 1), 3)

# The geometry of `grid`, a single region on the CPU, with owned fractions `owned_fraction(i, j)`.
# `face_vectors` adds the unit vectors that project onto C-grid components.
function RegionGeometry(grid, owned_fraction; face_vectors=false)
    FT = eltype(grid)
    Nx, Ny, _ = size(grid)
    pole = frame_north_pole(grid)
    c, f = Center(), Face()

    vectors(ℓx, ℓy) = [frame_unit_vectors(pole, Float64(λnode(i, j, 1, grid, ℓx, ℓy, c)),
                                                Float64(φnode(i, j, 1, grid, ℓx, ℓy, c))) for i in 1:Nx, j in 1:Ny]

    centers = vectors(c, c)
    area = reshape(FT[owned_fraction(i, j) * Azᶜᶜᵃ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny], Nx, Ny, 1)
    radial, east, north = ntuple(d -> component_arrays(FT, getindex.(centers, d)), 3)

    if face_vectors
        x_faces, y_faces = vectors(f, c), vectors(c, f)
        x_face_east, x_face_north = component_arrays(FT, getindex.(x_faces, 2)), component_arrays(FT, getindex.(x_faces, 3))
        y_face_east, y_face_north = component_arrays(FT, getindex.(y_faces, 2)), component_arrays(FT, getindex.(y_faces, 3))
        x_face_area, y_face_area = zero(area), zero(area)
    else
        x_face_east = x_face_north = y_face_east = y_face_north = x_face_area = y_face_area = nothing
    end

    geometry = RegionGeometry(area, radial, east, north, x_face_east, x_face_north, y_face_east, y_face_north,
                              x_face_area, y_face_area, zero(area))
    update_face_areas!(geometry)

    return geometry
end

# The owned areas at the faces are those of Oceananigans' reductions on a Yin-Yang grid, which lump the owned
# areas of the two adjacent cells; cells beyond the edge of a panel own nothing.
# TODO: exact owned areas at face locations belong in `Oceananigans.MultiRegion` (`owned_weight` is lumped).
update_face_areas!(geometry::RegionGeometry{<:Any, <:Any, <:Any, Nothing}) = nothing

function update_face_areas!(geometry::RegionGeometry)
    area = geometry.area
    Nx, Ny, _ = size(area)
    for j in 1:Ny, i in 1:Nx
        geometry.x_face_area[i, j, 1] = (area[i, j, 1] + (i > 1 ? area[i - 1, j, 1] : 0)) / 2
        geometry.y_face_area[i, j, 1] = (area[i, j, 1] + (j > 1 ? area[i, j - 1, 1] : 0)) / 2
    end
    return nothing
end
