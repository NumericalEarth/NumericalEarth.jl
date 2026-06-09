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
    exchange_grid = TripolarGrid(CPU(); size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))

    era5_dataset = ERA5HourlySingleLevel()
    era5_start = first_date(era5_dataset, :temperature)
    era5_end = era5_start + Hour(1)

    ospapa_dataset = OSPapaHourly()
    ospapa_start = first_date(ospapa_dataset, :air_temperature)
    ospapa_end = ospapa_start + Hour(1)

    atmospheres = (
        (name = "JRA55", atmosphere = JRA55PrescribedAtmosphere(CPU(); time_indices_in_memory = 2)),
        (name = "ECCO", atmosphere = ECCOPrescribedAtmosphere(CPU(); dataset = ECCO4Monthly(),
                                                                     start_date = start_date,
                                                                     end_date = start_date + Month(1),
                                                                     time_indices_in_memory = 2)),
        (name = "ERA5", atmosphere = ERA5PrescribedAtmosphere(CPU(); dataset = era5_dataset,
                                                                     start_date = era5_start,
                                                                     end_date = era5_end,
                                                                     time_indices_in_memory = 2)),
        (name = "OSPapa", atmosphere = OSPapaPrescribedAtmosphere(CPU(); start_date = ospapa_start,
                                                                         end_date = ospapa_end))
    )

    for (; name, atmosphere) in atmospheres
        @testset "$name" begin
            exchanger = NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere, exchange_grid)
            for field in (exchanger.state.u, exchanger.state.v)
                assert_tripolar_velocity_zipper(field, exchange_grid)
            end
        end
    end
end
