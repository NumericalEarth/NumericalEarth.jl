include("runtests_setup.jl")

using LibGEOS

using Oceananigans.AbstractOperations: RegriddedOperation
using Oceananigans.Fields: Field, interior, set!
using Oceananigans.Grids: RightCenterFolded, RightFaceFolded
using Oceananigans.Operators: Δxᶜᶠᵃ, Δyᶠᶜᵃ, Δzᶜᶠᶜ, Δzᶠᶜᶜ
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

function manually_integrated_transports(source_u, source_v)
    grid = on_architecture(CPU(), source_u.grid)
    u_data = Array(interior(source_u))
    v_data = Array(interior(source_v))
    Nxᵤ, Nyᵤ, Nz = size(source_u)
    Nxᵥ, Nyᵥ, _ = size(source_v)
    integrated_u = zeros(Float64, Nxᵤ, Nyᵤ)
    integrated_v = zeros(Float64, Nxᵥ, Nyᵥ)

    for k in 1:Nz, j in 1:Nyᵤ, i in 1:Nxᵤ
        integrated_u[i, j] += u_data[i, j, k] *
                              Δyᶠᶜᵃ(i, j, k, grid) *
                              Δzᶠᶜᶜ(i, j, k, grid)
    end

    for k in 1:Nz, j in 1:Nyᵥ, i in 1:Nxᵥ
        integrated_v[i, j] += v_data[i, j, k] *
                              Δxᶜᶠᵃ(i, j, k, grid) *
                              Δzᶜᶠᶜ(i, j, k, grid)
    end

    return integrated_u, integrated_v
end

function east_north_transport(integrated_u, integrated_v, grid, extension)
    Nxᵤ, Nyᵤ = size(integrated_u)
    Nxᵥ, Nyᵥ = size(integrated_v)
    east_transport = 0.0
    north_transport = 0.0

    for j in 1:Nyᵤ, i in 1:Nxᵤ
        east, north = extension.face_basis_u(grid, i, j)
        transport = integrated_u[i, j]
        east_transport += east * transport
        north_transport += north * transport
    end

    for j in 1:Nyᵥ, i in 1:Nxᵥ
        east, north = extension.face_basis_v(grid, i, j)
        transport = integrated_v[i, j]
        east_transport += east * transport
        north_transport += north * transport
    end

    return east_transport, north_transport
end

function valid_source_u_face(grid, i, j, extension)
    λ₁, φ₁, λ₂, φ₂ = extension.source_u_segment_endpoints(grid, i, j)
    segment_length = hypot(λ₂ - λ₁, φ₂ - φ₁)
    return all(isfinite, (λ₁, φ₁, λ₂, φ₂, segment_length)) && segment_length > eps(Float64)
end

function valid_source_v_face(grid, i, j, extension)
    λ₁, φ₁, λ₂, φ₂ = extension.source_v_segment_endpoints(grid, i, j)
    segment_length = hypot(λ₂ - λ₁, φ₂ - φ₁)
    return all(isfinite, (λ₁, φ₁, λ₂, φ₂, segment_length)) && segment_length > eps(Float64)
end

function remove_global_mean_transport!(source_u, source_v, extension)
    grid = on_architecture(CPU(), source_u.grid)
    Nxᵤ, Nyᵤ, Nz = size(source_u)
    Nxᵥ, Nyᵥ, _ = size(source_v)
    u_data = Array(interior(source_u))
    v_data = Array(interior(source_v))

    for j in 1:Nyᵤ, i in 1:Nxᵤ
        valid_source_u_face(grid, i, j, extension) && continue
        u_data[i, j, :] .= 0
    end

    for j in 1:Nyᵥ, i in 1:Nxᵥ
        valid_source_v_face(grid, i, j, extension) && continue
        v_data[i, j, :] .= 0
    end

    set!(source_u, u_data)
    set!(source_v, v_data)

    integrated_u, integrated_v = manually_integrated_transports(source_u, source_v)
    source_east, source_north = east_north_transport(integrated_u, integrated_v, grid, extension)
    east_east = 0.0
    east_north = 0.0
    north_north = 0.0

    # These coefficients describe the two global geographic velocity modes in
    # the native C-grid basis. Removing them makes the test flow globally closed.
    for j in 1:Nyᵤ, i in 1:Nxᵤ
        valid_source_u_face(grid, i, j, extension) || continue
        column_area = sum(Δyᶠᶜᵃ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid) for k in 1:Nz)
        east, north = extension.face_basis_u(grid, i, j)
        east_east += column_area * east^2
        east_north += column_area * east * north
        north_north += column_area * north^2
    end

    for j in 1:Nyᵥ, i in 1:Nxᵥ
        valid_source_v_face(grid, i, j, extension) || continue
        column_area = sum(Δxᶜᶠᵃ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid) for k in 1:Nz)
        east, north = extension.face_basis_v(grid, i, j)
        east_east += column_area * east^2
        east_north += column_area * east * north
        north_north += column_area * north^2
    end

    determinant = east_east * north_north - east_north^2
    east_correction = (source_east * north_north - source_north * east_north) / determinant
    north_correction = (source_north * east_east - source_east * east_north) / determinant

    for k in 1:Nz, j in 1:Nyᵤ, i in 1:Nxᵤ
        valid_source_u_face(grid, i, j, extension) || continue
        east, north = extension.face_basis_u(grid, i, j)
        u_data[i, j, k] -= east_correction * east + north_correction * north
    end

    for k in 1:Nz, j in 1:Nyᵥ, i in 1:Nxᵥ
        valid_source_v_face(grid, i, j, extension) || continue
        east, north = extension.face_basis_v(grid, i, j)
        v_data[i, j, k] -= east_correction * east + north_correction * north
    end

    set!(source_u, u_data)
    set!(source_v, v_data)
    return nothing
end

tripolar_size(::Type{RightFaceFolded}) = (16, 9, 3)
tripolar_size(::Type{RightCenterFolded}) = (16, 8, 3)

@testset "Global transport conservation [$arch]" for arch in test_architectures
    destination_grid = LatitudeLongitudeGrid(CPU();
                                             size = (20, 10, 1),
                                             longitude = (0, 360),
                                             latitude = (-90, 90),
                                             z = (-1, 0))

    for fold_topology in (RightFaceFolded, RightCenterFolded)
        source_grid = TripolarGrid(arch;
                                   size = tripolar_size(fold_topology),
                                   z = (-300, 0),
                                   southernmost_latitude = -75,
                                   fold_topology)
        source_u = XFaceField(source_grid)
        source_v = YFaceField(source_grid)

        u_flux(λ, φ, z) = isfinite(λ) && isfinite(φ) ? 1 + cosd(φ)^2 + z / 1000 : 0.0
        v_flux(λ, φ, z) = isfinite(λ) && isfinite(φ) ? 2 + sind(λ)^2 - z / 2000 : 0.0
        set!(source_u, u_flux)
        set!(source_v, v_flux)

        extension = Base.get_extension(NumericalEarth, :NumericalEarthVelocityTransportExt)
        @test !isnothing(extension)
        remove_global_mean_transport!(source_u, source_v, extension)

        manually_integrated_u, manually_integrated_v = manually_integrated_transports(source_u, source_v)
        cpu_source_grid = on_architecture(CPU(), source_grid)
        source_east, source_north = east_north_transport(manually_integrated_u,
                                                         manually_integrated_v,
                                                         cpu_source_grid,
                                                         extension)
        transport_scale = sum(abs, manually_integrated_u) + sum(abs, manually_integrated_v)
        @test max(abs(source_east), abs(source_north)) / transport_scale < 1e-12

        operations = RegriddedOperation((; u=source_u, v=source_v), destination_grid)
        transport = Field(operations)

        cache = operations.u.regridder
        @test cache === operations.v.regridder && cache.status.computations == 1

        destination_u = Array(interior(transport.u))
        destination_v = Array(interior(transport.v))
        destination_east = sum(destination_u)
        destination_north = sum(destination_v)
        destination_scale = sum(abs, destination_u) + sum(abs, destination_v)
        @test destination_scale > sqrt(eps(Float64)) * transport_scale
        @test max(abs(destination_east), abs(destination_north)) / transport_scale < 1e-12
    end
end
