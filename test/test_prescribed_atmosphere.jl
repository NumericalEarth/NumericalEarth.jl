include("runtests_setup.jl")

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: parent
using Oceananigans.Grids: halo_size

function assert_tripolar_velocity_zipper(field, grid)
    set!(field, 1)
    fill_halo_regions!(field)

    Hx, Hy, Hz = halo_size(grid)
    data = parent(field)

    i = Hx + 1:size(data, 1) - Hx
    j_interior = Hy + size(grid, 2)
    j_halo = j_interior + 1
    k = 1

    north_halo = Array(@view data[i, j_halo, k])
    @test all(north_halo .== -1)
end

@testset "PrescribedAtmosphere tripolar velocity zipper sign" begin
    grid = TripolarGrid(CPU(); size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))
    atmosphere = PrescribedAtmosphere(grid, [0.0])

    for field in (atmosphere.velocities.u[1], atmosphere.velocities.v[1])
        assert_tripolar_velocity_zipper(field, grid)
    end
end

@testset "Regridded prescribed atmosphere tripolar velocity zipper sign" begin
    # The exchanger builds its velocity state on the (tripolar) exchange grid with
    # north-fold BCs, independent of the atmosphere's source grid or data, so a plain
    # PrescribedAtmosphere on a lat-lon grid exercises the same path without any download.
    # Dataset-backed atmosphere constructors are exercised by the download tests.
    atmosphere_grid = LatitudeLongitudeGrid(CPU(); size = (36, 18, 1),
                                            longitude = (0, 360), latitude = (-80, 80),
                                            z = (-1, 0), halo = (3, 3, 3))
    atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0])

    exchange_grid = TripolarGrid(CPU(); size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))
    exchanger = NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere, exchange_grid)

    for field in (exchanger.state.u, exchanger.state.v)
        assert_tripolar_velocity_zipper(field, exchange_grid)
    end
end
