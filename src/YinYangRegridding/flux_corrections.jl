#####
##### Global integrals in the ownership quadrature, and the corrections that close them
#####

"""
    GlobalRescaling()

Correction that multiplies a transferred scalar flux by one factor so that its global integral equals
the source's. It is defined for finite, non-negative fields only, such as precipitation and downwelling
radiation, whose sign it keeps. A source with negative, NaN or infinite values is an error, raised before
the destination is written; so is a transferred field with zero integral when the source's is not. Signed fluxes, such as net radiation and turbulent heat fluxes,
and vectors take `UniformRedistribution()`.

Like `UniformRedistribution`, it closes the budget integrated over the sphere with the ownership weights
and nothing else: it does not make the transfer locally conservative.
"""
struct GlobalRescaling end

"""
    UniformRedistribution()

Correction that spreads the difference between the global integrals of the source and of the transferred
flux uniformly over the sphere: additive, weighted by the owned areas, and defined for fields of either
sign, including fields whose integral vanishes. A scalar is shifted by a constant. A horizontal vector is
shifted by `P a + b × r̂`, the tangential part of a uniform vector `a` plus a rigid rotation `b`, chosen so
that both the net force `∫ τ dA` and the net torque `∫ r × τ dA` equal the source's, measured where the
components live: at cell centers, or at the C-grid faces with the lumped owned areas there.

It closes the budget integrated over the sphere with the ownership weights and nothing else: it does not
make the transfer locally conservative, nor conserve the flux through the seam between the panels. A
shifted field is not bounded by the bounds of its source.
"""
struct UniformRedistribution end

# Σ f(arrays...) over the cells of a region. The integrand is stored and then summed, so that the sum is
# pairwise: a sequential sum of 1e5 terms is accurate to 1e-13 only, which is the level of closure asked of
# the corrections.
function region_sum(f, geometry, arrays...)
    geometry.workspace .= f.(arrays...)
    return sum(geometry.workspace)
end

regional_interior(field, r) = interior(getregion(field, r))

total_area(geometries) = sum(geometry -> sum(geometry.area), geometries)

surface_integral(geometries::Tuple, field) =
    sum(r -> region_sum(*, geometries[r], geometries[r].area, regional_interior(field, r)), eachindex(geometries))

"""
$(SIGNATURES)

Return the integral over the sphere of `field`, a field at cell centers on either grid of `regridder`.
On the Yin-Yang grid this is the ownership quadrature `Σ w Az c` over both panels, in which every point
of the sphere counts once; on the latitude-longitude grid it is `Σ Az c`. The owned areas `w Az` are scaled
by one factor, within 1e-13 of one, so that they sum to the area of the latitude-longitude grid.
"""
surface_integral(regridder::YinYangRegridder, field) = surface_integral(region_geometries(regridder, field), field)

# What a correction requires of the source, checked before the destination is written
validate_correctable(regridder, source, correction) = nothing

function validate_correctable(regridder, source, ::GlobalRescaling)
    regions = eachindex(region_geometries(regridder, source))

    all(r -> all(isfinite, regional_interior(source, r)), regions) ||
        throw(ArgumentError("GlobalRescaling requires a finite flux; the source has NaN or infinite values."))

    lowest = minimum(r -> minimum(regional_interior(source, r)), regions)

    # A source with negative values can average to a non-negative field with a positive integral
    lowest ≥ 0 ||
        throw(ArgumentError("GlobalRescaling is defined for non-negative fluxes only; the source has minimum " *
                            "$lowest. Use UniformRedistribution for signed fluxes."))

    return nothing
end

correct_integral!(destination, regridder, source, ::Nothing) = nothing

function correct_integral!(destination, regridder, source, ::GlobalRescaling)
    geometries = region_geometries(regridder, destination)
    target = surface_integral(regridder, source)
    actual = surface_integral(geometries, destination)
    lowest = minimum(r -> minimum(regional_interior(destination, r)), eachindex(geometries))

    lowest ≥ 0 && target ≥ 0 ||
        throw(ArgumentError("GlobalRescaling is defined for non-negative fluxes only; the transferred field has " *
                            "minimum $lowest and the source integrates to $target. Use UniformRedistribution " *
                            "for signed fluxes."))

    # An identically zero field needs no correction
    actual == 0 && target == 0 && return nothing

    actual > 0 ||
        throw(ArgumentError("GlobalRescaling cannot close a source integral of $target on a transferred field " *
                            "that integrates to zero. Use UniformRedistribution."))

    factor = target / actual
    for r in eachindex(geometries)
        regional_interior(destination, r) .*= factor
    end
    return nothing
end

function correct_integral!(destination, regridder, source, ::UniformRedistribution)
    geometries = region_geometries(regridder, destination)
    target = surface_integral(regridder, source)
    actual = surface_integral(geometries, destination)
    shift = (target - actual) / total_area(geometries)
    for r in eachindex(geometries)
        regional_interior(destination, r) .+= shift
    end
    return nothing
end

#####
##### Force and torque of a tangent vector field held as Cartesian components (X, Y, Z)
#####

@inline torque_x(A, y, z, Y, Z) = A * (y * Z - z * Y)

# The net force Σ A τ and the net torque per unit radius Σ A r̂ × τ
function force_and_torque(geometries, cartesian)
    force = zero(SVector{3, Float64})
    torque = zero(SVector{3, Float64})

    for r in eachindex(geometries)
        geometry = geometries[r]
        A = geometry.area
        x, y, z = geometry.radial
        X, Y, Z = map(component -> regional_interior(component, r), cartesian)
        force += SVector(region_sum(*, geometry, A, X), region_sum(*, geometry, A, Y), region_sum(*, geometry, A, Z))
        torque += SVector(region_sum(torque_x, geometry, A, y, z, Y, Z),
                          region_sum(torque_x, geometry, A, z, x, Z, X),
                          region_sum(torque_x, geometry, A, x, y, X, Y))
    end

    return force, torque
end

# The 6 × 6 matrix that maps the parameters (a, b) of the rigid shift δ = P a + b × r̂, with P = I - r̂ r̂ᵀ the
# tangential projection, to its net force and net torque per unit radius:
#
#     Σ A δ      = M a - C b,
#     Σ A r̂ × δ = C a + M b,     M = Σ A P,   C = Σ A [r̂]ₓ.
#
# On the sphere M = (8π/3) R² I and C = 0, so the matrix is well conditioned.
function rigid_shift_operator(geometries)
    M = zero(SMatrix{3, 3, Float64})
    C = zero(SMatrix{3, 3, Float64})
    identity_matrix = SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1)

    for geometry in geometries
        A = Array(geometry.area)
        x, y, z = map(Array, geometry.radial)
        for n in eachindex(A)
            r̂ = SVector{3, Float64}(x[n], y[n], z[n])
            M += A[n] * (identity_matrix - r̂ * r̂')
            C += A[n] * SMatrix{3, 3, Float64}(0, r̂[3], -r̂[2], -r̂[3], 0, r̂[1], r̂[2], -r̂[1], 0)
        end
    end

    return SMatrix{6, 6, Float64}([M -C; C M])
end

@kernel function _add_rigid_shift!(X, Y, Z, a, b, geometry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        r̂ = SVector(geometry.radial[1][i, j, 1], geometry.radial[2][i, j, 1], geometry.radial[3][i, j, 1])
        radial_part = a[1] * r̂[1] + a[2] * r̂[2] + a[3] * r̂[3]
        δ = a - radial_part * r̂ + cross_product(b, r̂)
        X[i, j, 1] += δ[1]
        Y[i, j, 1] += δ[2]
        Z[i, j, 1] += δ[3]
    end
end

correct_force_and_torque!(cartesian, grid, geometries, operator, target, ::Nothing) = nothing

correct_force_and_torque!(cartesian, grid, geometries, operator, target, ::GlobalRescaling) =
    throw(ArgumentError("GlobalRescaling is not defined for vectors: use UniformRedistribution."))

# Shift the tangent vector `cartesian` on `grid` rigidly so that its force and torque equal `target`
function correct_force_and_torque!(cartesian, grid, geometries, operator, target, ::UniformRedistribution)
    force, torque = force_and_torque(geometries, cartesian)
    target_force, target_torque = target
    parameters = operator \ SVector{6, Float64}((target_force - force)..., (target_torque - torque)...)

    FT = eltype(grid)
    a = SVector{3, FT}(parameters[1], parameters[2], parameters[3])
    b = SVector{3, FT}(parameters[4], parameters[5], parameters[6])

    for r in eachindex(geometries)
        region = getregion(grid, r)
        X, Y, Z = map(component -> getregion(component.data, r), cartesian)
        launch!(architecture(region), region, :xy, _add_rigid_shift!, X, Y, Z, a, b, geometries[r])
    end

    return nothing
end

#####
##### Force and torque of C-grid components, measured at the faces where they live
#####
##### With q = (e; r̂ × e) the 6-vector of a unit component along e at a face, the force and torque of the
##### components (u, v) with owned face areas (Aᵘ, Aᵛ) are Σ Aᵘ u qᵘ + Σ Aᵛ v qᵛ, and the rigid shift with
##### parameters p = (a; b) changes the components by δu = qᵘ ⋅ p and δv = qᵛ ⋅ p. At an x-face r̂ × e is the
##### y-direction unit vector there, and at a y-face r̂ × n is minus the x-direction unit vector.
#####

@inline triple_product(a, b, c) = a * b * c

function staggered_force_and_torque(geometries, (u, v))
    force = zero(SVector{3, Float64})
    torque = zero(SVector{3, Float64})

    for r in eachindex(geometries)
        g = geometries[r]
        uʳ, vʳ = regional_interior(u, r), regional_interior(v, r)
        component_sum(area, component, vector) = SVector(ntuple(d -> region_sum(triple_product, g, area, component, vector[d]), 3))
        force += component_sum(g.x_face_area, uʳ, g.x_face_east) + component_sum(g.y_face_area, vʳ, g.y_face_north)
        torque += component_sum(g.x_face_area, uʳ, g.x_face_north) - component_sum(g.y_face_area, vʳ, g.y_face_east)
    end

    return force, torque
end

# The symmetric 6 × 6 matrix Σ Aᵘ qᵘ qᵘᵀ + Σ Aᵛ qᵛ qᵛᵀ that maps the parameters of a rigid shift of C-grid
# components to its force and torque
function staggered_rigid_shift_operator(geometries)
    operator = zero(SMatrix{6, 6, Float64})

    for geometry in geometries
        Aᵘ, Aᵛ = Array(geometry.x_face_area), Array(geometry.y_face_area)
        eᵘ, nᵘ = map(Array, geometry.x_face_east), map(Array, geometry.x_face_north)
        eᵛ, nᵛ = map(Array, geometry.y_face_east), map(Array, geometry.y_face_north)
        for m in eachindex(Aᵘ)
            qᵘ = SVector{6, Float64}(eᵘ[1][m], eᵘ[2][m], eᵘ[3][m], nᵘ[1][m], nᵘ[2][m], nᵘ[3][m])
            qᵛ = SVector{6, Float64}(nᵛ[1][m], nᵛ[2][m], nᵛ[3][m], -eᵛ[1][m], -eᵛ[2][m], -eᵛ[3][m])
            operator += Aᵘ[m] * qᵘ * qᵘ' + Aᵛ[m] * qᵛ * qᵛ'
        end
    end

    return operator
end

@kernel function _add_staggered_rigid_shift!(u, v, a, b, geometry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        eᵘ, nᵘ, eᵛ, nᵛ = geometry.x_face_east, geometry.x_face_north, geometry.y_face_east, geometry.y_face_north
        u[i, j, 1] += a[1] * eᵘ[1][i, j, 1] + a[2] * eᵘ[2][i, j, 1] + a[3] * eᵘ[3][i, j, 1] +
                      b[1] * nᵘ[1][i, j, 1] + b[2] * nᵘ[2][i, j, 1] + b[3] * nᵘ[3][i, j, 1]
        v[i, j, 1] += a[1] * nᵛ[1][i, j, 1] + a[2] * nᵛ[2][i, j, 1] + a[3] * nᵛ[3][i, j, 1] -
                      b[1] * eᵛ[1][i, j, 1] - b[2] * eᵛ[2][i, j, 1] - b[3] * eᵛ[3][i, j, 1]
    end
end

correct_staggered_force_and_torque!(components, geometries, operator, target, ::Nothing) = nothing

correct_staggered_force_and_torque!(components, geometries, operator, target, ::GlobalRescaling) =
    throw(ArgumentError("GlobalRescaling is not defined for vectors: use UniformRedistribution."))

# Shift the C-grid `components` rigidly so that their force and torque, measured at the faces, equal `target`
function correct_staggered_force_and_torque!(components, geometries, operator, target, ::UniformRedistribution)
    u, v = components
    force, torque = staggered_force_and_torque(geometries, components)
    target_force, target_torque = target
    parameters = operator \ SVector{6, Float64}((target_force - force)..., (target_torque - torque)...)

    FT = eltype(u.grid)
    a = SVector{3, FT}(parameters[1], parameters[2], parameters[3])
    b = SVector{3, FT}(parameters[4], parameters[5], parameters[6])

    for r in eachindex(geometries)
        region = getregion(u.grid, r)
        launch!(architecture(region), region, :xy, _add_staggered_rigid_shift!,
                getregion(u.data, r), getregion(v.data, r), a, b, geometries[r])
    end

    return nothing
end
