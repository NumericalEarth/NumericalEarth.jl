include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: reference_density, heat_capacity

for arch in test_architectures
    A = typeof(arch)

    @testset "Section transports [$A]" begin
        Lz = 100
        underlying_grid = LatitudeLongitudeGrid(arch; size = (20, 20, 4), z = (-Lz, 0),
                                                longitude = (-10, 10), latitude = (60, 80))

        # Land east of 0°E, so the zonal section below is open only west of 0°E
        bottom(λ, φ) = ifelse(λ > 0, 0, -Lz)
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

        ocean = ocean_simulation(grid; momentum_advection = nothing, tracer_advection = nothing,
                                 closure = nothing, coriolis = nothing)

        u₀, v₀, T₀, S₀ = 0.1, 0.2, 2, 35
        set!(ocean.model, u = u₀, v = v₀, T = T₀, S = S₀)

        transport(operation) = only(Array(interior(compute!(operation))))

        ρᵒᶜ = reference_density(ocean)
        cᵒᶜ = heat_capacity(ocean)
        R = grid.underlying_grid.radius

        # Walking west along 70°N, northward flow is to the right; the open part spans 5°W to 0°E
        zonal = TransportSection(grid, [5, -5], [70, 70])
        V = v₀ * R * deg2rad(5) * cosd(70) * Lz

        @test transport(volume_transport(ocean, zonal)) ≈ V
        @test transport(heat_transport(ocean, zonal; reference_temperature = 1)) ≈ ρᵒᶜ * cᵒᶜ * (T₀ - 1) * V
        @test transport(salt_transport(ocean, zonal)) ≈ ρᵒᶜ * S₀ / 1000 * V
        @test transport(freshwater_transport(ocean, zonal)) ≈ (34.8 - S₀) / 34.8 * V

        reversed = TransportSection(grid, [-5, 5], [70, 70])
        @test transport(volume_transport(ocean, reversed)) ≈ -V

        # Walking north along 5°W, eastward flow is to the right
        meridional = TransportSection(grid, [-5, -5], [65, 75])
        @test transport(volume_transport(ocean, meridional)) ≈ u₀ * R * deg2rad(10) * Lz
    end
end
