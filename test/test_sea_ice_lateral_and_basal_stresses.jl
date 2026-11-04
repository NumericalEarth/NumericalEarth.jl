include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: surface_layer_velocities
using NumericalEarth.SeaIces: LandfastBasalStress, sea_ice_dynamics, sea_ice_simulation,
                              sea_ice_velocity_boundary_conditions

using ClimaSeaIce.Rheologies: FreeSlip, NoSlip, u_lateral_boundary_condition, v_lateral_boundary_condition
using ClimaSeaIce.SeaIceDynamics: SemiImplicitStress, StressBalanceFreeDrift

using Oceananigans.BoundaryConditions: Value

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

@testset "Ice-ocean drag reads the velocity averaged over the reference depth" begin
    grid = RectilinearGrid(size = (8, 8, 8), halo = (7, 7, 7), x = (0, 8e4), y = (0, 8e4), z = (-40, 0),
                           topology = (Bounded, Bounded, Bounded))   # 5 m cells
    ocean = ocean_simulation(grid; coriolis = nothing)
    set!(ocean.model, u = (x, y, z) -> -z)
    uˢˡ, _ = surface_layer_velocities(ocean, 6)

    # top 6 m: 5 m of the cell centred at 2.5 m and 1 m of the cell centred at 7.5 m
    @test all(interior(uˢˡ, 2:grid.Nx, :, 1) .≈ (5 * 2.5 + 1 * 7.5) / 6)
end
