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
        atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=4)
        radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory=4)
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

        atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=4)
        radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory=4)

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
        land_dates = all_dates(RepeatYearJRA55(), :river_freshwater_flux)
        land = JRA55PrescribedLand(arch; end_date=land_dates[2])

        @test begin
            ocean_with_land = ocean_simulation(grid; free_surface)
            coupled_model = OceanOnlyModel(ocean_with_land; atmosphere, land, radiation)

            # Verify land exchanger is present
            @test !isnothing(coupled_model.interfaces.exchanger.land)
            @test coupled_model.land.freshwater_flux === land.freshwater_flux

            time_step!(coupled_model, 1)
            true
        end
    end
end

@testset "ocean_simulation merges partial boundary conditions with defaults" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing ocean_simulation partial boundary conditions on $A..."

        grid = LatitudeLongitudeGrid(arch;
                                     size = (20, 16, 4),
                                     longitude = (5, 25),
                                     latitude = (67, 75),
                                     z = (-1000, 0),
                                     halo = (7, 7, 7))

        T_lateral_bcs = FieldBoundaryConditions(east = ValueBoundaryCondition(2),
                                                west = ValueBoundaryCondition(4))
        u_lateral_bcs = FieldBoundaryConditions(east = OpenBoundaryCondition(0.1))

        reference = ocean_simulation(grid)
        ocean = ocean_simulation(grid; boundary_conditions = (u = u_lateral_bcs, T = T_lateral_bcs))

        T_bcs = ocean.model.tracers.T.boundary_conditions
        u_bcs = ocean.model.velocities.u.boundary_conditions

        # The user-prescribed lateral sides survive...
        @test T_bcs.east.condition == 2
        @test T_bcs.west.condition == 4
        @test u_bcs.east.condition == 0.1

        # ... and the default top fluxes (read by the coupling), bottom drag, and
        # immersed drag survive alongside them.
        reference_T_bcs = reference.model.tracers.T.boundary_conditions
        reference_u_bcs = reference.model.velocities.u.boundary_conditions

        @test typeof(T_bcs.top) == typeof(reference_T_bcs.top)
        @test !isnothing(T_bcs.top.condition)
        @test typeof(u_bcs.top) == typeof(reference_u_bcs.top)
        @test typeof(u_bcs.bottom) == typeof(reference_u_bcs.bottom)
        @test typeof(u_bcs.immersed) == typeof(reference_u_bcs.immersed)
    end
end
