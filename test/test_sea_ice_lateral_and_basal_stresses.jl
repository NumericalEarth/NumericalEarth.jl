include("runtests_setup.jl")

using NumericalEarth.SeaIces: LandfastBasalStress, sea_ice_dynamics, sea_ice_simulation,
                              sea_ice_velocity_boundary_conditions

using ClimaSeaIce.Rheologies: FreeSlip, NoSlip, u_lateral_boundary_condition, v_lateral_boundary_condition
using ClimaSeaIce.SeaIceDynamics: SemiImplicitStress, StressBalanceFreeDrift

using Oceananigans.BoundaryConditions: Value, fill_halo_regions!

# A shallow shelf (20 m) next to a deep basin (40 m), vertically resolved so the shelf column depth
# does not collapse onto a single cell.
function shelf_grid(arch = CPU(); Nx = 8, Ny = 8, Nz = 8)
    underlying = RectilinearGrid(arch; size = (Nx, Ny, Nz), x = (0, 8e4), y = (0, 8e4), z = (-40, 0),
                                 topology = (Bounded, Bounded, Bounded))
    bottom = [i <= Nx ÷ 2 ? -20.0 : -40.0 for i in 1:Nx, j in 1:Ny]
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))
end

@testset "Sea-ice lateral boundary conditions" begin
    grid = shelf_grid()

    free_slip = sea_ice_velocity_boundary_conditions(grid, :free_slip)
    no_slip = sea_ice_velocity_boundary_conditions(grid, :no_slip)

    @test u_lateral_boundary_condition(free_slip.u.immersed) isa FreeSlip
    @test v_lateral_boundary_condition(free_slip.v.immersed) isa FreeSlip

    @test u_lateral_boundary_condition(no_slip.u.immersed) isa NoSlip
    @test v_lateral_boundary_condition(no_slip.v.immersed) isa NoSlip

    # Oceananigans regularizes a scalar immersed condition onto the sides where the component is
    # tangential: south/north for u, west/east for v.
    @test no_slip.u.immersed.south.classification isa Value
    @test no_slip.u.immersed.north.classification isa Value
    @test no_slip.v.immersed.west.classification isa Value
    @test no_slip.v.immersed.east.classification isa Value
end

@testset "Sea-ice basal stress wiring" begin
    grid = shelf_grid()

    dynamics = sea_ice_dynamics(grid)

    @test dynamics.basal_stress isa LandfastBasalStress

    # The sea floor carries the basal stress, so what the coupler hands the ocean is the bare drag.
    @test dynamics.external_momentum_stresses.bottom isa SemiImplicitStress
    @test dynamics.free_drift isa StressBalanceFreeDrift
    @test dynamics.free_drift.bottom_momentum_stress === dynamics.external_momentum_stresses.bottom

    @test sea_ice_dynamics(grid; basal_stress = nothing).basal_stress isa Nothing
end

@testset "Sea-ice simulation carries the no-slip condition" begin
    grid = shelf_grid()
    sea_ice = sea_ice_simulation(grid)

    u = sea_ice.model.velocities.u
    v = sea_ice.model.velocities.v

    @test u_lateral_boundary_condition(u.boundary_conditions.immersed) isa NoSlip
    @test v_lateral_boundary_condition(v.boundary_conditions.immersed) isa NoSlip
end

@testset "Sea-ice simulation boundary conditions" begin
    grid = shelf_grid()
    h = FieldBoundaryConditions(grid, (Center(), Center(), nothing); east = ValueBoundaryCondition(1.5))
    ℵ = FieldBoundaryConditions(grid, (Center(), Center(), nothing); east = ValueBoundaryCondition(0.8))
    sea_ice = sea_ice_simulation(grid; boundary_conditions = (; h, ℵ))

    h_east = sea_ice.model.ice_thickness.boundary_conditions.east
    ℵ_east = sea_ice.model.ice_concentration.boundary_conditions.east
    @test h_east.classification isa Value
    @test h_east.condition == 1.5
    @test ℵ_east.classification isa Value
    @test ℵ_east.condition == 0.8

    # Fields not passed keep the defaults, including the lateral condition on the velocities
    u = sea_ice.model.velocities.u
    @test u_lateral_boundary_condition(u.boundary_conditions.immersed) isa NoSlip

    v_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing); east = ValueBoundaryCondition(0.1))
    sea_ice = sea_ice_simulation(grid; boundary_conditions = (; v = v_bcs))
    v_east = sea_ice.model.velocities.v.boundary_conditions.east
    @test v_east.classification isa Value
    @test v_east.condition == 0.1
end

@testset "Sea-ice simulation boundary conditions from a time series" begin
    Nx, Ny = 4, 3
    grid = RectilinearGrid(size = (Nx, Ny), x = (0, 4e4), y = (0, 3e4), topology = (Bounded, Bounded, Flat))

    times = [0.0, 1.0, 2.0]
    h_boundary = FieldTimeSeries{Nothing, Center, Nothing}(grid, times)
    for n in eachindex(times)
        interior(h_boundary, 1, :, 1, n) .= (1:Ny) .+ times[n]
    end

    h_bcs = FieldBoundaryConditions(grid, (Center(), Center(), nothing); east = ValueBoundaryCondition(h_boundary))
    sea_ice = sea_ice_simulation(grid; boundary_conditions = (; h = h_bcs))

    model = sea_ice.model
    h = model.ice_thickness
    model.clock.time = 1.5
    fill_halo_regions!(h, model.clock)

    # The halo mirrors the interior about the boundary value, interpolated in time
    for j in 1:Ny
        @test h[Nx+1, j, 1] ≈ 2 * (j + 1.5) - h[Nx, j, 1]
    end
end
