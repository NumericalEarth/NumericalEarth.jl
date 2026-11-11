include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: reference_density, heat_capacity
using Oceananigans.BoundaryConditions: West, East, South, North

for arch in test_architectures
    A = typeof(arch)

    @testset "Section transports [$A]" begin
        Lz = 100
        underlying_grid = LatitudeLongitudeGrid(arch; size = (20, 20, 4), z = (-Lz, 0),
                                                longitude = (-10, 10), latitude = (60, 80))

        # Land east of 0°E, and west of it a bottom at depth H that cuts the third of four layers
        H = 60
        bottom(λ, φ) = ifelse(λ > 0, 0, -H)
        grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

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
        V = v₀ * R * deg2rad(5) * cosd(70) * H

        @test transport(volume_transport(ocean, zonal)) ≈ V
        @test transport(heat_transport(ocean, zonal; reference_temperature = 1)) ≈ ρᵒᶜ * cᵒᶜ * (T₀ - 1) * V
        @test transport(salt_transport(ocean, zonal)) ≈ ρᵒᶜ * S₀ / 1000 * V
        @test transport(freshwater_transport(ocean, zonal)) ≈ (34.8 - S₀) / 34.8 * V

        reversed = TransportSection(grid, [-5, 5], [70, 70])
        @test transport(volume_transport(ocean, reversed)) ≈ -V

        # Walking north along 5°W, eastward flow is to the right
        meridional = TransportSection(grid, [-5, -5], [65, 75])
        @test transport(volume_transport(ocean, meridional)) ≈ u₀ * R * deg2rad(10) * H

        # Flow into the domain through its edges, which set! leaves closed; the east edge is land
        (; u, v) = ocean.model.velocities
        view(interior(u), 1, :, :) .= u₀
        view(interior(u), size(u, 1), :, :) .= u₀
        view(interior(v), :, 1, :) .= v₀
        view(interior(v), :, size(v, 2), :) .= v₀

        @test transport(volume_transport(ocean, TransportSection(grid, West()))) ≈ u₀ * R * deg2rad(20) * H
        @test transport(volume_transport(ocean, TransportSection(grid, East()))) == 0
        @test transport(volume_transport(ocean, TransportSection(grid, South()))) ≈ v₀ * R * deg2rad(10) * cosd(60) * H
        @test transport(volume_transport(ocean, TransportSection(grid, North()))) ≈ -v₀ * R * deg2rad(10) * cosd(80) * H

        # A section along a whole row of v faces gives the meridional heat transport at that row
        set!(ocean.model, v = (λ, φ, z) -> cosd(9λ) * (1 + z / Lz), T = (λ, φ, z) -> φ - 60 + z / Lz)
        j = size(grid, 2) ÷ 2
        v_signs = Field{Center, Face, Nothing}(grid)
        view(interior(v_signs), :, j, 1) .= 1
        row = TransportSection(Field{Face, Center, Nothing}(grid), v_signs)

        mht = Field(meridional_heat_transport(OceanOnlyModel(ocean); reference_temperature = 1))
        @test transport(heat_transport(ocean, row; reference_temperature = 1)) ≈ Array(interior(compute!(mht)))[1, j, 1]
    end
end
