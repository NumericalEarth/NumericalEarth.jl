include("runtests_setup.jl")

using NumericalEarth.SeaIces: LandfastBasalStress, SeaIceBottomStress,
                              lateral_boundary_conditions, sea_ice_dynamics,
                              basal_τx_coefficient, basal_τy_coefficient,
                              materialize_basal_stress
using NumericalEarth.EarthSystemModels.InterfaceComputations: ice_ocean_momentum_stress

using ClimaSeaIce.Rheologies: ElastoViscoPlasticRheology
using ClimaSeaIce.SeaIceDynamics: SemiImplicitStress, implicit_τx_coefficient, explicit_τx

using Oceananigans.BoundaryConditions: ImmersedBoundaryCondition
using Oceananigans.Fields: interior

# A shallow shelf (20 m) next to a deep basin (2000 m), so exactly half the domain can ground ice.
function shelf_grid(arch = CPU(); Nx = 8, Ny = 8, Nz = 4)
    underlying = RectilinearGrid(arch; size = (Nx, Ny, Nz), x = (0, 8e4), y = (0, 8e4), z = (-2000, 0),
                                 topology = (Bounded, Bounded, Bounded))
    bottom = [i <= Nx ÷ 2 ? -20.0 : -2000.0 for i in 1:Nx, j in 1:Ny]
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))
end

@testset "Landfast basal stress" begin
    grid = shelf_grid()

    b = LandfastBasalStress()
    @test b.water_depth isa Nothing
    @test b.critical_thickness_parameter == 8
    @test b.stress_parameter == 15
    @test b.maximum_water_depth == 30

    bm = materialize_basal_stress(b, grid)
    H = interior(bm.water_depth)[:, :, 1]
    @test all(H[1:4, :] .≈ 20)
    @test all(H[5:8, :] .≈ 2000)

    # Thick, compact ice: grounds on the shelf (hᶜ = 20/8 = 2.5 m) and never in the basin.
    h = Field{Center, Center, Nothing}(grid); set!(h, 3.0)
    ℵ = Field{Center, Center, Nothing}(grid); set!(ℵ, 1.0)
    ρ = Field{Center, Center, Nothing}(grid); set!(ρ, 900.0)
    u = Field{Face,   Center, Nothing}(grid); set!(u, 0.1)
    v = Field{Center, Face,   Nothing}(grid); set!(v, 0.0)
    fields = (; h, ℵ, ρ, u, v)

    kᴺ = size(grid, 3)
    shelf = basal_τx_coefficient(2, 4, kᴺ, grid, bm, fields)
    basin = basal_τx_coefficient(7, 4, kᴺ, grid, bm, fields)

    @test shelf > 0
    @test basin == 0

    # τᵇ = k₂ (h - H ℵ / k₁) exp(-C(1-ℵ)) / (|u| + u₀), with ℵ = 1 so the exponential is unity.
    @test shelf ≈ 15 * (3.0 - 20/8) / (0.1 + 5e-5)

    # Thin ice cannot ground even on the shelf.
    set!(h, 1.0)
    @test basal_τx_coefficient(2, 4, kᴺ, grid, bm, fields) == 0

    # Unconsolidated ice is released by the concentration hardening.
    set!(h, 3.0); set!(ℵ, 0.5)
    loose = basal_τx_coefficient(2, 4, kᴺ, grid, bm, fields)
    @test 0 < loose < shelf

    @test basal_τx_coefficient(2, 4, kᴺ, grid, nothing, fields) == 0
    @test basal_τy_coefficient(2, 4, kᴺ, grid, nothing, fields) == 0
end

@testset "SeaIceBottomStress composition" begin
    grid = shelf_grid()

    τo = SemiImplicitStress(Float64; Cᴰ = 5.5e-3, ρₑ = 1026.0)
    τb = SeaIceBottomStress(τo, materialize_basal_stress(LandfastBasalStress(), grid))

    # Only the ocean part is handed to the ocean; the seabed carries the rest.
    @test ice_ocean_momentum_stress(τb) === τo
    @test ice_ocean_momentum_stress(τo) === τo

    h = Field{Center, Center, Nothing}(grid); set!(h, 3.0)
    ℵ = Field{Center, Center, Nothing}(grid); set!(ℵ, 1.0)
    ρ = Field{Center, Center, Nothing}(grid); set!(ρ, 900.0)
    u = Field{Face,   Center, Nothing}(grid); set!(u, 0.1)
    v = Field{Center, Face,   Nothing}(grid); set!(v, 0.0)
    fields = (; h, ℵ, ρ, u, v)

    clock = Clock(grid)
    kᴺ = size(grid, 3)

    # The composite adds the two drags, and the explicit part is the ocean's alone.
    total = implicit_τx_coefficient(2, 4, kᴺ, grid, τb, clock, fields)
    ocean = implicit_τx_coefficient(2, 4, kᴺ, grid, τo, clock, fields)
    basal = basal_τx_coefficient(2, 4, kᴺ, grid, τb.basal, fields)

    @test total ≈ ocean + basal
    @test basal > ocean                      # grounded keels dominate the water drag
    @test explicit_τx(2, 4, kᴺ, grid, τb, clock, fields) ≈ explicit_τx(2, 4, kᴺ, grid, τo, clock, fields)

    # In the deep basin the composite reduces to the ocean drag.
    @test implicit_τx_coefficient(7, 4, kᴺ, grid, τb, clock, fields) ≈
          implicit_τx_coefficient(7, 4, kᴺ, grid, τo, clock, fields)
end

@testset "Lateral boundary conditions" begin
    grid = shelf_grid()
    dynamics = sea_ice_dynamics(grid)

    free_slip = lateral_boundary_conditions(grid, Val(:free_slip), dynamics, 1800)
    @test isnothing(free_slip.u.immersed) || !(free_slip.u.immersed isa ImmersedBoundaryCondition)

    no_slip = lateral_boundary_conditions(grid, Val(:no_slip), dynamics, 1800)
    @test no_slip.u.immersed isa ImmersedBoundaryCondition
    @test no_slip.v.immersed isa ImmersedBoundaryCondition

    # u feels the walls normal to y, v the walls normal to x.
    @test !isnothing(no_slip.u.immersed.south) && !isnothing(no_slip.u.immersed.north)
    @test !isnothing(no_slip.v.immersed.west)  && !isnothing(no_slip.v.immersed.east)

    # Without a momentum equation there are no rheology auxiliaries to read, so we fall back.
    @test lateral_boundary_conditions(grid, Val(:no_slip), nothing, 1800).u.immersed ===
          lateral_boundary_conditions(grid, Val(:free_slip), nothing, 1800).u.immersed
end

@testset "sea_ice_dynamics defaults" begin
    grid = shelf_grid()
    dynamics = sea_ice_dynamics(grid)

    τ = dynamics.external_momentum_stresses.bottom
    @test τ isa SeaIceBottomStress
    @test τ.ocean.Cᴰ == 5.5e-3
    @test τ.basal isa LandfastBasalStress
    @test τ.basal.water_depth isa Field          # materialized on the velocity grid

    # Free drift balances against the ocean drag alone, and still dispatches.
    @test dynamics.free_drift.bottom_momentum_stress isa SemiImplicitStress

    @test sea_ice_dynamics(grid; basal_stress = nothing).external_momentum_stresses.bottom.basal isa Nothing
end
