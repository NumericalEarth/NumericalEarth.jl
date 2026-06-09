include("runtests_setup.jl")

using CDSAPI

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: parent
using Oceananigans.Grids: halo_size

@testset "PrescribedAtmosphere tripolar velocity zipper sign" begin
    grid = TripolarGrid(CPU(); size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))
    atmosphere = PrescribedAtmosphere(grid, [0.0])

    for field in (atmosphere.velocities.u[1], atmosphere.velocities.v[1])
        set!(field, 1)
        fill_halo_regions!(field)

        Hx, Hy, Hz = halo_size(grid)
        data = parent(field)

        i = Hx + 1:size(data, 1) - Hx
        j_interior = Hy + size(grid, 2)
        j_halo = j_interior + 1
        k = 1

        seam = Array(@view data[i, j_interior, k])
        north_halo = Array(@view data[i, j_halo, k])

        @test all(north_halo .== -1)
    end
end
