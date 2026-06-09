include("runtests_setup.jl")

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: parent
using Oceananigans.Grids: halo_size

@testset "JRA55PrescribedAtmosphere tripolar velocity zipper sign" begin
    ocean_grid = TripolarGrid(CPU(); size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))
    ocean = ocean_simulation(ocean_grid; closure = nothing)
    atmosphere = JRA55PrescribedAtmosphere(CPU(); time_indices_in_memory = 2)
    radiation = JRA55PrescribedRadiation(CPU(); time_indices_in_memory = 2)

    coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

    for field in (coupled_model.interfaces.exchanger.atmosphere.state.u,
                  coupled_model.interfaces.exchanger.atmosphere.state.v)
        set!(field, 1)
        fill_halo_regions!(field)

        Hx, Hy, Hz = halo_size(ocean_grid)
        data = parent(field)

        i = Hx + 1:size(data, 1) - Hx
        j_interior = Hy + size(ocean_grid, 2)
        j_halo = j_interior + 1
        k = 1

        seam = Array(@view data[i, j_interior, k])
        north_halo = Array(@view data[i, j_halo, k])

        @test all(north_halo .== -1)
    end
end
