include("runtests_setup.jl")

using NumericalEarth
using Oceananigans
using Oceananigans.Grids: znode
using Test

@testset "hydrostatic_pressure_from_surface" begin
    Rᵈ, Rᵛ, g = 287.0, 461.0, 9.81
    T₀, p₀ = 250.0, 1.0e5

    grid = RectilinearGrid(size = (2, 2, 8), x = (0, 1), y = (0, 1), z = (0, 2000),
                           topology = (Periodic, Periodic, Bounded))
    zc(k) = znode(1, 1, k, grid, Center(), Center(), Center())

    T = CenterField(grid)
    set!(T, T₀)

    # Dry isothermal: p(z) = p₀ exp(−g z / (Rᵈ T₀)) exactly (constant T ⇒ the level-by-level
    # integration telescopes to the analytic profile).
    p = hydrostatic_pressure_from_surface(T, fill(p₀, 2, 2), fill(0.0, 2, 2);
                                          dry_gas_constant = Rᵈ,
                                          vapor_gas_constant = Rᵛ,
                                          gravitational_acceleration = g)
    for k in 1:8
        @test interior(p)[1, 1, k] ≈ p₀ * exp(-g * zc(k) / (Rᵈ * T₀)) rtol = 1e-6
    end
    @test all(diff(interior(p)[1, 1, :]) .< 0)   # monotonically decreasing with height

    # Moist: Rᵐ = qᵈRᵈ + qᵛRᵛ > Rᵈ ⇒ slower falloff ⇒ higher pressure aloft.
    qᵛ = CenterField(grid)
    set!(qᵛ, 0.02)
    pᵐ = hydrostatic_pressure_from_surface(T, fill(p₀, 2, 2), fill(0.0, 2, 2);
                                           qᵛ = qᵛ,
                                           dry_gas_constant = Rᵈ,
                                           vapor_gas_constant = Rᵛ,
                                           gravitational_acceleration = g)
    @test interior(pᵐ)[1, 1, end] > interior(p)[1, 1, end]

    # Orography offset: surface pressure given at z = 500 m, reduced down to the z = 0 bottom
    # face ⇒ p(z) = p₀ exp(−g (z − 500) / (Rᵈ T₀)), so near-surface pressure exceeds p₀.
    orog = 500.0
    pᵒ = hydrostatic_pressure_from_surface(T, fill(p₀, 2, 2), fill(orog, 2, 2);
                                           dry_gas_constant = Rᵈ,
                                           vapor_gas_constant = Rᵛ,
                                           gravitational_acceleration = g)
    for k in 1:8
        @test interior(pᵒ)[1, 1, k] ≈ p₀ * exp(-g * (zc(k) - orog) / (Rᵈ * T₀)) rtol = 1e-6
    end
    @test interior(pᵒ)[1, 1, 1] > p₀
end
