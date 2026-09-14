include("runtests_setup.jl")

using NumericalEarth.Oceans: net_flux, net_flux_coefficient
using NumericalEarth.EarthSystemModels.InterfaceComputations: SeaIceOceanFluxes

using ClimaSeaIce.SeaIceDynamics: SemiImplicitStress, x_momentum_stress, y_momentum_stress,
                                  explicit_τx, explicit_τy,
                                  implicit_τx_coefficient, implicit_τy_coefficient

using Oceananigans.BoundaryConditions: IMEXFlux
using Oceananigans.Fields: interior

# The ice-ocean drag is handed to the ocean as an affine flux J = Fₑ + λ uᵒ. The split is only
# correct if it reproduces the total stress that the un-split formulation would have applied, so
# these tests pin that identity rather than the individual terms — a sign error in either part
# cancels in isolation but not in the sum.

@testset "Ice-ocean drag split reproduces the total stress" begin
    grid = RectilinearGrid(size = (4, 4, 4), x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Periodic, Periodic, Bounded))
    kᴺ = size(grid, 3)

    uᵒ = Field{Face, Center, Center}(grid);   set!(uᵒ, 0.30)
    vᵒ = Field{Center, Face, Center}(grid);   set!(vᵒ, -0.10)
    uⁱ = Field{Face, Center, Nothing}(grid);  set!(uⁱ, -0.20)
    vⁱ = Field{Center, Face, Nothing}(grid);  set!(vⁱ, 0.05)
    h  = Field{Center, Center, Nothing}(grid); set!(h, 1.0)
    ℵ  = Field{Center, Center, Nothing}(grid); set!(ℵ, 1.0)

    τ = SemiImplicitStress(uₑ=uᵒ, vₑ=vᵒ, Cᴰ=5.5e-3, ρₑ=1026.0)
    fields = (; u=uⁱ, v=vⁱ, h, ℵ)
    clock = Clock(grid)

    i = j = 2
    λˣ = implicit_τx_coefficient(i, j, kᴺ, grid, τ, clock, fields)
    λʸ = implicit_τy_coefficient(i, j, kᴺ, grid, τ, clock, fields)
    Fˣ = x_momentum_stress(i, j, kᴺ, grid, τ, clock, fields) - explicit_τx(i, j, kᴺ, grid, τ, clock, fields)
    Fʸ = y_momentum_stress(i, j, kᴺ, grid, τ, clock, fields) - explicit_τy(i, j, kᴺ, grid, τ, clock, fields)

    # Reassembling with the ocean velocity must recover the total stress exactly.
    @test Fˣ + λˣ * uᵒ[i, j, kᴺ] ≈ x_momentum_stress(i, j, kᴺ, grid, τ, clock, fields)
    @test Fʸ + λʸ * vᵒ[i, j, kᴺ] ≈ y_momentum_stress(i, j, kᴺ, grid, τ, clock, fields)

    # λ is a drag, so it must be positive-definite: it can only remove ocean momentum.
    @test λˣ > 0
    @test λʸ > 0

    # The explicit part carries only the ice velocity, so it flips sign with the ice.
    set!(uⁱ, 0.20)
    Fˣ⁺ = x_momentum_stress(i, j, kᴺ, grid, τ, clock, fields) - explicit_τx(i, j, kᴺ, grid, τ, clock, fields)
    @test sign(Fˣ⁺) == -sign(Fˣ)
end

@testset "SeaIceOceanFluxes carries the coefficient fields" begin
    grid = RectilinearGrid(size = (4, 4, 4), x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Periodic, Periodic, Bounded))
    fluxes = SeaIceOceanFluxes(grid)

    @test size(parent(fluxes.x_momentum_coefficient)) == size(parent(fluxes.x_momentum))
    @test size(parent(fluxes.y_momentum_coefficient)) == size(parent(fluxes.y_momentum))
    @test all(interior(fluxes.x_momentum_coefficient) .== 0)
    @test all(interior(fluxes.y_momentum_coefficient) .== 0)
end

@testset "Ocean surface momentum uses a semi-implicit flux" begin
    # `ocean_simulation` defaults to WENO advection, which needs a halo of at least 6.
    grid = RectilinearGrid(size = (8, 8, 8), x = (0, 1), y = (0, 1), z = (-1, 0),
                           halo = (7, 7, 7), topology = (Periodic, Periodic, Bounded))
    ocean = ocean_simulation(grid)

    u_top = ocean.model.velocities.u.boundary_conditions.top.condition
    v_top = ocean.model.velocities.v.boundary_conditions.top.condition

    @test u_top isa IMEXFlux
    @test v_top isa IMEXFlux

    # The coupler reaches the two halves through these accessors.
    @test net_flux(u_top) isa Field
    @test net_flux_coefficient(u_top) isa Field
    @test net_flux_coefficient(v_top) isa Field

    # Tracers stay fully explicit — only momentum carries the stiff ice drag.
    T_top = ocean.model.tracers.T.boundary_conditions.top.condition
    @test !(T_top isa IMEXFlux)
    @test net_flux_coefficient(T_top) isa Nothing
end
