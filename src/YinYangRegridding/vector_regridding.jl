#####
##### Regridding of horizontal vectors as Cartesian 3-vectors
#####
##### Components in a panel's basis cannot be averaged across panels, and geographic components cannot be
##### averaged near the geographic poles, where the basis turns by a finite angle between neighboring
##### cells. The Cartesian components of a vector are scalars, so they can be regridded, and have their
##### Yin-Yang halos filled, like any other scalar.
#####

const centered_pair = ((Center(), Center()), (Center(), Center()))
const staggered_pair = ((Face(), Center()), (Center(), Face()))

@inline centered(i, j, grid, ::Center, ::Center, c) = @inbounds c[i, j, 1]
@inline centered(i, j, grid, ::Face, ::Center, u) = ℑxᶜᵃᵃ(i, j, 1, grid, u)
@inline centered(i, j, grid, ::Center, ::Face, v) = ℑyᵃᶜᵃ(i, j, 1, grid, v)

@kernel function _horizontal_to_cartesian!(X, Y, Z, u, v, grid, locations, geometry)
    i, j = @index(Global, NTuple)
    ℓu, ℓv = locations

    # Both components at the cell center, so that they form one vector at one point
    uᶜ = centered(i, j, grid, ℓu..., u)
    vᶜ = centered(i, j, grid, ℓv..., v)

    @inbounds begin
        X[i, j, 1] = uᶜ * geometry.east[1][i, j, 1] + vᶜ * geometry.north[1][i, j, 1]
        Y[i, j, 1] = uᶜ * geometry.east[2][i, j, 1] + vᶜ * geometry.north[2][i, j, 1]
        Z[i, j, 1] = uᶜ * geometry.east[3][i, j, 1] + vᶜ * geometry.north[3][i, j, 1]
    end
end

# Averages of Cartesian components pick up a radial part of second order in the cell size
@kernel function _remove_radial_component!(X, Y, Z, geometry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        x, y, z = geometry.radial[1][i, j, 1], geometry.radial[2][i, j, 1], geometry.radial[3][i, j, 1]
        radial_part = X[i, j, 1] * x + Y[i, j, 1] * y + Z[i, j, 1] * z
        X[i, j, 1] -= radial_part * x
        Y[i, j, 1] -= radial_part * y
        Z[i, j, 1] -= radial_part * z
    end
end

@inline projection(i, j, X, Y, Z, e) = @inbounds X[i, j, 1] * e[1][i, j, 1] + Y[i, j, 1] * e[2][i, j, 1] + Z[i, j, 1] * e[3][i, j, 1]

@kernel function _cartesian_to_centered_components!(u, v, X, Y, Z, geometry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        u[i, j, 1] = projection(i, j, X, Y, Z, geometry.east)
        v[i, j, 1] = projection(i, j, X, Y, Z, geometry.north)
    end
end

@kernel function _cartesian_to_staggered_components!(u, v, X, Y, Z, grid, geometry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        e, n = geometry.x_face_east, geometry.y_face_north

        u[i, j, 1] = ℑxᶠᵃᵃ(i, j, 1, grid, X) * e[1][i, j, 1] +
                     ℑxᶠᵃᵃ(i, j, 1, grid, Y) * e[2][i, j, 1] +
                     ℑxᶠᵃᵃ(i, j, 1, grid, Z) * e[3][i, j, 1]

        v[i, j, 1] = ℑyᵃᶠᵃ(i, j, 1, grid, X) * n[1][i, j, 1] +
                     ℑyᵃᶠᵃ(i, j, 1, grid, Y) * n[2][i, j, 1] +
                     ℑyᵃᶠᵃ(i, j, 1, grid, Z) * n[3][i, j, 1]
    end
end

regional_data(fields::Tuple, r) = map(field -> getregion(field.data, r), fields)

function validate_vector_locations(regridder, components, allowed)
    locations = map(horizontal_location, components)
    locations in allowed ||
        throw(ArgumentError("A YinYangRegridder regrids vector components at $allowed; got $locations."))
    foreach(component -> validate_grid(regridder, component), components)
    return locations
end

vector_locations(regridder, components::Tuple{YinYangField, YinYangField}) =
    validate_vector_locations(regridder, components, (centered_pair, staggered_pair))

vector_locations(regridder, components::Tuple{LatitudeLongitudeField, LatitudeLongitudeField}) =
    validate_vector_locations(regridder, components, (centered_pair,))

# Write the Cartesian components at cell centers of the horizontal vector `components` into `cartesian`
function horizontal_to_cartesian!(cartesian, regridder, components)
    u, v = components
    locations = vector_locations(regridder, components)
    geometries = region_geometries(regridder, u)

    for r in eachindex(geometries)
        region = getregion(u.grid, r)
        launch!(architecture(region), region, :xy, _horizontal_to_cartesian!,
                regional_data(cartesian, r)..., getregion(u.data, r), getregion(v.data, r),
                region, locations, geometries[r])
    end

    return nothing
end

function remove_radial_component!(cartesian, grid, geometries)
    for r in eachindex(geometries)
        region = getregion(grid, r)
        launch!(architecture(region), region, :xy, _remove_radial_component!, regional_data(cartesian, r)..., geometries[r])
    end
    return nothing
end

# Project the tangent vector `cartesian` onto the basis of `components` and fill the halos that can be filled
function cartesian_to_horizontal!(components, regridder, cartesian)
    u, v = components
    locations = vector_locations(regridder, components)
    geometries = region_geometries(regridder, u)

    # The face values average the cell centers on both sides, including those across the panel edges
    locations == staggered_pair && foreach(fill_halo_regions!, cartesian)

    for r in eachindex(geometries)
        region = getregion(u.grid, r)
        arch = architecture(region)
        uʳ, vʳ = getregion(u.data, r), getregion(v.data, r)

        if locations == staggered_pair
            launch!(arch, region, :xy, _cartesian_to_staggered_components!,
                    uʳ, vʳ, regional_data(cartesian, r)..., region, geometries[r])
        else
            launch!(arch, region, :xy, _cartesian_to_centered_components!,
                    uʳ, vʳ, regional_data(cartesian, r)..., geometries[r])
        end
    end

    fill_vector_halos!(components, locations)

    return nothing
end

fill_vector_halos!(components::Tuple{LatitudeLongitudeField, LatitudeLongitudeField}, locations) = foreach(fill_halo_regions!, components)
fill_vector_halos!(components::Tuple{YinYangField, YinYangField}, locations) = locations == staggered_pair ? fill_halo_regions!(components) : nothing

# The force and torque of `components`, measured where they live, and their Cartesian form at cell centers
# in the scratch of their grid
function measure_force_and_torque!(cartesian, regridder, components)
    geometries = region_geometries(regridder, first(components))
    locations = vector_locations(regridder, components)
    horizontal_to_cartesian!(cartesian, regridder, components)
    return locations == staggered_pair ? staggered_force_and_torque(geometries, components) :
                                         force_and_torque(geometries, cartesian)
end

"""
$(SIGNATURES)

Return the net force `Σ A τ` and the net torque per unit radius `Σ A r̂ × τ` of the horizontal vector
`components = (u, v)` on either grid of `regridder`, as Cartesian 3-vectors in the geographic frame.

The sums are taken where the components live. For components at (Center, Center), `A` are the owned
cell areas. For C-grid components, each is summed at its own faces, with the owned area there: the
mean of the owned areas of the two adjacent cells, which is the lumped face weight of Oceananigans'
reductions on a Yin-Yang grid, and an approximation of the area owned at a face.
"""
function surface_force_and_torque(regridder::YinYangRegridder, components::Tuple)
    cartesian = cartesian_scratch(regridder, first(components))
    return measure_force_and_torque!(cartesian, regridder, components)
end

validate_vector_correction(correction) = nothing

validate_vector_correction(::GlobalRescaling) =
    throw(ArgumentError("GlobalRescaling is not defined for vectors: use UniformRedistribution."))

# Regrid the vector `source` onto `destination` as a Cartesian 3-vector with `operator`, closing the force and
# torque with `correction` in the representation of `destination`
function regrid_vector!(destination, regridder, source, operator, correction)
    locations = vector_locations(regridder, destination) # validate before anything is written
    validate_vector_correction(correction)
    source_cartesian = cartesian_scratch(regridder, first(source))
    cartesian = cartesian_scratch(regridder, first(destination))
    grid = first(destination).grid
    geometries = region_geometries(regridder, first(destination))

    target = measure_force_and_torque!(source_cartesian, regridder, source)

    for d in 1:3
        regrid_interior!(cartesian[d], regridder, source_cartesian[d], operator)
    end

    remove_radial_component!(cartesian, grid, geometries)

    if locations == centered_pair
        shift_operator = rigid_shift_operator(regridder, first(destination))
        correct_force_and_torque!(cartesian, grid, geometries, shift_operator, target, correction)
        cartesian_to_horizontal!(destination, regridder, cartesian)
    else
        # The correction acts on the components that are applied: at the faces, after the conversion
        cartesian_to_horizontal!(destination, regridder, cartesian)
        shift_operator = regridder.rigid_shift_operators.yin_yang_staggered
        correct_staggered_force_and_torque!(destination, geometries, shift_operator, target, correction)
        isnothing(correction) || fill_vector_halos!(destination, locations)
    end

    return destination
end

const FieldPair = Tuple{Field, Field}

interpolate_to_yin_yang!(destination::Tuple{YinYangField, YinYangField}, regridder::YinYangRegridder,
                         source::Tuple{LatitudeLongitudeField, LatitudeLongitudeField}) =
    regrid_vector!(destination, regridder, source, BilinearInterpolation(), nothing)

transfer_flux!(destination::FieldPair, regridder::YinYangRegridder, source::FieldPair; correction = nothing) =
    regrid_vector!(destination, regridder, source, AreaWeightedAverage(), correction)
