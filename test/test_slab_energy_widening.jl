using Test
using Oceananigans
using Oceananigans.Fields: interior, CenterField
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth
using NumericalEarth.Lands: SlabLand, SlabEnergy

slab_widen_grid() = RectilinearGrid(CPU();
                                    size = (2, 1), halo = (1, 1),
                                    x = (0, 2), y = (0, 1),
                                    topology = (Bounded, Bounded, Flat))

scalar2(field) = Array(interior(field))[:, 1, 1]

@testset "SlabEnergy widening" begin
    @testset "scalar heat_capacity preserves existing behavior" begin
        grid = slab_widen_grid()
        C = 1.0e6
        land = SlabLand(grid; energy = SlabEnergy(Float64; heat_capacity = C))
        fill!(land.state.T, 300.0)
        fill!(land.fluxes.net_energy_flux, 100.0)   # W m⁻²
        time_step!(land, 10.0)
        @test all(scalar2(land.state.T) .≈ 300.0 + 100.0 * 10.0 / C)
    end

    @testset "per-cell Field heat_capacity gives per-cell tendency" begin
        grid = slab_widen_grid()
        C_field = CenterField(grid)
        C_field[1, 1, 1] = 1.0e6
        C_field[2, 1, 1] = 2.0e6
        land = SlabLand(grid; energy = SlabEnergy(Float64; heat_capacity = C_field))
        fill!(land.state.T, 300.0)
        fill!(land.fluxes.net_energy_flux, 100.0)
        time_step!(land, 10.0)
        T = scalar2(land.state.T)
        @test T[1] ≈ 300.0 + 100.0 * 10.0 / 1.0e6
        @test T[2] ≈ 300.0 + 100.0 * 10.0 / 2.0e6
    end
end
