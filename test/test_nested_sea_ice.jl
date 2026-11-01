include("runtests_setup.jl")

using NumericalEarth.NestedModels: SeaIceBoundarySeries, SeaIceBoundaryVelocity, boundary_ice_velocity
using Oceananigans.BoundaryConditions: fill_halo_regions!, getbc, NormalFlow, TracerReservoir, Value
using Oceananigans.Grids: halo_size
using Oceananigans.Units: days, minutes

function nested_sea_ice_test_ocean(arch)
    parent_grid = LatitudeLongitudeGrid(arch; size = (12, 12, 6),
                                        longitude = (-2, 2), latitude = (28, 32), z = (-1000, 0),
                                        topology = (Bounded, Bounded, Bounded))

    parent_ocean = PrescribedOcean(parent_grid, [0.0, 10days])
    set!(parent_ocean; T = 15, S = 35, u = 0.1, v = -0.05, η = 0)

    grid = LatitudeLongitudeGrid(arch; size = (20, 20, 6), halo = (7, 7, 7),
                                 longitude = (-1, 1), latitude = (29, 31), z = (-1000, 0),
                                 topology = (Bounded, Bounded, Bounded))

    return Simulation(nested_ocean_model(parent_ocean, grid); Δt = 10minutes)
end

# The parent ice varies along both directions, so that a boundary read at the wrong index shows.
function nested_sea_ice_test_parent(grid)
    times = [0.0, 10days]
    series(LX, LY) = FieldTimeSeries{LX, LY, Nothing}(grid, times)

    h, ℵ, u, v = series(Center, Center), series(Center, Center), series(Center, Center), series(Center, Center)

    for n in eachindex(times)
        set!(h[n], (λ, φ) -> 1 + (λ + 1) / 4 + (φ - 29) / 8 + n / 10)
        set!(ℵ[n], (λ, φ) -> φ > 30.5 ? NaN : 0.9)
        set!(u[n], 0.2)
        set!(v[n], -0.1)
    end

    return (; h, ℵ, u, v)
end

@testset "boundary_ice_velocity: ice leaves, is not pulled in, and follows the interior without parent ice" begin
    # inward = 1: a west or south side
    @test boundary_ice_velocity(0.2, 0.9, 0.0, 0.05, 1) == 0.05    # parent inflow: copy the interior
    @test boundary_ice_velocity(-0.2, 0.9, 0.0, 0.05, 1) == 0.0    # parent outflow, interior inward: clamp
    @test boundary_ice_velocity(-0.2, 0.9, 0.0, -0.05, 1) == -0.05 # parent outflow, interior outward: pass through
    @test boundary_ice_velocity(0.0, 0.9, -0.3, 0.05, 1) == 0.0    # free drift: the ocean decides the direction
    @test boundary_ice_velocity(0.2, NaN, 0.0, -0.05, 1) == -0.05  # no parent ice
    @test boundary_ice_velocity(0.2, 0.0, 0.0, -0.05, 1) == -0.05

    # inward = -1: an east or north side
    @test boundary_ice_velocity(-0.2, 0.9, 0.0, -0.05, -1) == -0.05
    @test boundary_ice_velocity(0.2, 0.9, 0.0, -0.05, -1) == 0.0
    @test boundary_ice_velocity(0.2, 0.9, 0.0, 0.05, -1) == 0.05
end

@testset "nested_sea_ice_boundary_conditions: the ice opens where the ocean does on $(arch)" for arch in test_architectures
    ocean = nested_sea_ice_test_ocean(arch)
    grid = ocean.model.child.grid
    parent_ice = nested_sea_ice_test_parent(grid)

    bcs = nested_sea_ice_boundary_conditions(ocean, parent_ice)

    for side in (:west, :east, :south, :north)
        for name in (:h, :ℵ)
            bc = getproperty(bcs[name], side)
            @test bc.classification isa Value
            @test bc.classification.scheme isa TracerReservoir
            @test bc.condition isa SeaIceBoundarySeries
        end

        hs_bc = getproperty(bcs.hs, side)
        @test hs_bc.classification isa Value
        @test hs_bc.classification.scheme isa TracerReservoir
        @test hs_bc.condition == 0
    end

    @test bcs.u.west.classification isa NormalFlow
    @test bcs.u.east.condition isa SeaIceBoundaryVelocity{1}
    @test bcs.v.south.condition isa SeaIceBoundaryVelocity{2}
    @test bcs.v.north.classification isa NormalFlow

    # Each boundary reads the parent at its own row or column; open water is no ice
    cpu_grid = on_architecture(CPU(), grid)
    h = on_architecture(CPU(), parent_ice.h)
    ℵ = on_architecture(CPU(), parent_ice.ℵ)
    Nx, Ny, _ = size(grid)
    west_h  = on_architecture(CPU(), bcs.h.west.condition)
    north_h = on_architecture(CPU(), bcs.h.north.condition)
    east_ℵ  = on_architecture(CPU(), bcs.ℵ.east.condition)

    @test [getbc(west_h, j, 1, cpu_grid) for j in 1:Ny] ≈ [h[1, j, 1, 1] for j in 1:Ny]
    @test [getbc(north_h, i, 1, cpu_grid) for i in 1:Nx] ≈ [h[i, Ny, 1, 1] for i in 1:Nx]
    @test [getbc(east_ℵ, j, 1, cpu_grid) for j in 1:Ny] == [isnan(ℵ[Nx, j, 1, 1]) ? 0.0 : 0.9 for j in 1:Ny]
    @test getbc(west_h, Ny + 5, 1, cpu_grid) == getbc(west_h, Ny, 1, cpu_grid)
end

@testset "nested sea ice: boundary halos and a coupled step on $(arch)" for arch in test_architectures
    ocean = nested_sea_ice_test_ocean(arch)
    grid = ocean.model.child.grid
    parent_ice = nested_sea_ice_test_parent(grid)

    # A plain value condition puts the parent on the boundary face
    bcs = nested_sea_ice_boundary_conditions(ocean, parent_ice; tracer_scheme = nothing)
    sea_ice = sea_ice_simulation(grid, ocean; boundary_conditions = bcs)
    set_parent_sea_ice!(sea_ice, parent_ice)

    h = sea_ice.model.ice_thickness
    ℵ = sea_ice.model.ice_concentration
    fill_halo_regions!(h, sea_ice.model.clock, fields(sea_ice.model))

    Nx, Ny, _ = size(grid)
    hᶜ = Array(parent(h))
    hᵖ = Array(parent(on_architecture(CPU(), parent_ice.h[1])))
    Hx, Hy, _ = halo_size(grid)
    face_value(i₀, i₁, j) = (hᶜ[i₀ + Hx, j + Hy, 1] + hᶜ[i₁ + Hx, j + Hy, 1]) / 2

    @test [face_value(0, 1, j) for j in 1:Ny] ≈ [hᵖ[1 + Hx, j + Hy, 1] for j in 1:Ny]
    @test [face_value(Nx, Nx + 1, j) for j in 1:Ny] ≈ [hᵖ[Nx + Hx, j + Hy, 1] for j in 1:Ny]

    @test all(isfinite, Array(interior(ℵ)))
    @test maximum(Array(interior(ℵ))) ≤ 0.9

    # A coupled step with the default reservoir scheme
    bcs = nested_sea_ice_boundary_conditions(ocean, parent_ice)
    sea_ice = sea_ice_simulation(grid, ocean; boundary_conditions = bcs, Δt = 10minutes)
    set_parent_sea_ice!(sea_ice, parent_ice)

    coupled = OceanSeaIceModel(ocean, sea_ice)
    for _ in 1:3
        time_step!(coupled, 10minutes)
    end

    @test all(isfinite, Array(interior(sea_ice.model.ice_thickness)))
    @test all(isfinite, Array(interior(sea_ice.model.velocities.u)))
    @test all(isfinite, Array(interior(ocean.model.child.velocities.u)))
end

@testset "nested_sea_ice_simulation starts from the parent and steps with the ocean on $(arch)" for arch in test_architectures
    ocean = nested_sea_ice_test_ocean(arch)
    grid = ocean.model.child.grid
    parent_ice = nested_sea_ice_test_parent(grid)

    sea_ice = nested_sea_ice_simulation(ocean, parent_ice)
    manual = sea_ice_simulation(grid, ocean; boundary_conditions = nested_sea_ice_boundary_conditions(ocean, parent_ice))
    set_parent_sea_ice!(manual, parent_ice)

    @test Array(interior(sea_ice.model.ice_thickness)) == Array(interior(manual.model.ice_thickness))
    @test Array(interior(sea_ice.model.ice_concentration)) == Array(interior(manual.model.ice_concentration))
    @test typeof(sea_ice.model.ice_thickness.boundary_conditions) == typeof(manual.model.ice_thickness.boundary_conditions)
    @test typeof(sea_ice.model.velocities.u.boundary_conditions) == typeof(manual.model.velocities.u.boundary_conditions)

    coupled = OceanSeaIceModel(ocean, sea_ice)
    for _ in 1:3
        time_step!(coupled, 10minutes)
    end

    @test all(isfinite, Array(interior(sea_ice.model.ice_thickness)))
    @test all(isfinite, Array(interior(sea_ice.model.velocities.u)))
end
