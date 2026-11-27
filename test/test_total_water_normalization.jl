include("runtests_setup.jl")

using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization
using NumericalEarth.EarthSystemModels: compute_total_water!

@testset "Total water normalization" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (12, 10, 4), halo = (7, 7, 7), longitude = (0, 360), latitude = (-60, 60),
                                 z = MutableVerticalDiscretization(collect(range(-400, 0, length = 5))))

    ocean = ocean_simulation(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 20), warn = false)
    set!(ocean.model, T = 10, S = 35)
    sea_ice = sea_ice_simulation(grid, ocean)

    atmosphere_grid = LatitudeLongitudeGrid(CPU(); size = (36, 18), longitude = (0, 360), latitude = (-90, 90),
                                            topology = (Periodic, Bounded, Flat))
    atmosphere = PrescribedAtmosphere(atmosphere_grid, [0, 1days])
    parent(atmosphere.precipitation_flux.rain) .= 1e-4
    radiation = PrescribedRadiation(atmosphere_grid, [0, 1days])

    coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
    simulation = Simulation(coupled_model; Δt = 10minutes, stop_iteration = 12)

    normalization = TotalWaterNormalization(coupled_model)
    add_callback!(simulation, normalization, IterationInterval(1))

    run!(simulation)

    # Without the normalization the rain would add about 10¹¹ m³ of water over the run
    @test only(Array(compute_total_water!(normalization, coupled_model))) ≈
          only(Array(normalization.reference_water)) atol = 1
end
