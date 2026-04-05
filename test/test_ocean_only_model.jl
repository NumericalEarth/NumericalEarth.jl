include("runtests_setup.jl")

using CUDA
using Oceananigans.OrthogonalSphericalShellGrids

@testset "OceanOnly Time stepping test" begin
    for arch in test_architectures

        A = typeof(arch)

        λ★, φ★ = 35.1, 50.1

        grid = RectilinearGrid(arch, size = 200, x = λ★, y = φ★,
                                z = (-400, 0), topology = (Flat, Flat, Bounded))

        ocean = ocean_simulation(grid)
        data = Int[]
        pushdata(sim) = push!(data, iteration(sim))
        add_callback!(ocean, pushdata)
        backend = JRA55NetCDFBackend(4)
        atmosphere = JRA55PrescribedAtmosphere(arch; backend)
        radiation = Radiation(arch)
        coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
        Δt = 60
        for n = 1:3
            time_step!(coupled_model, Δt)
        end
        @test data == [0, 1, 2, 3]

        # TODO: do the same for a SeaIceSimulation, and eventually prognostic Atmos

        #####
        ##### Ocean and prescribed atmosphere
        #####

        grid = TripolarGrid(arch;
                            size = (50, 50, 10),
                            halo = (7, 7, 7),
                            z = (-5000, 0))

        bottom_height = regrid_bathymetry(grid;
                                        minimum_depth = 10,
                                        interpolation_passes = 5,
                                        major_basins = 1)

        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

        free_surface = SplitExplicitFreeSurface(grid; substeps=20)
        ocean = ocean_simulation(grid; free_surface)

        backend = JRA55NetCDFBackend(4)
        atmosphere = JRA55PrescribedAtmosphere(arch; backend)
        radiation = Radiation(arch)

        # Fluxes are computed when the model is constructed, so we just test that this works.
        @test begin
            coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
            time_step!(coupled_model, 1)
            true
        end

        #####
        ##### Ocean with prescribed atmosphere and land
        #####

        @info "Testing OceanOnlyModel with JRA55PrescribedLand on $A..."
        land = JRA55PrescribedLand(arch; backend)

        @test begin
            ocean_with_land = ocean_simulation(grid; free_surface)
            coupled_model = OceanOnlyModel(ocean_with_land; atmosphere, land, radiation)

            # Verify land exchanger is present
            @test !isnothing(coupled_model.interfaces.exchanger.land)
            @test coupled_model.land === land

            time_step!(coupled_model, 1)
            true
        end
    end
end
