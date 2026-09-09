include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: exchange_grid, reference_density, heat_capacity

@testset "PrescribedOcean exchange grid and properties" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch,
                                     size = (8, 8, 1),
                                     z = (-100, 0),
                                     latitude = (-80, 80),
                                     longitude = (0, 360),
                                     halo = (6, 6, 3))

        ocean = PrescribedOcean(grid; density = 1025, heat_capacity = 3990)

        # The exchange grid is the ocean grid. The 4-argument (with-land) method is the
        # path taken when a PrescribedOcean is coupled alongside a land component, where
        # component_interfaces.jl calls exchange_grid(atmosphere, ocean, sea_ice, land).
        @test exchange_grid(nothing, ocean, nothing) === grid
        @test exchange_grid(nothing, ocean, nothing, nothing) === grid

        @test reference_density(ocean) == 1025
        @test heat_capacity(ocean) == 3990
    end
end
