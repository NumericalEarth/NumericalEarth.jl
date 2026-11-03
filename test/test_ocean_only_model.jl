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
        atmosphere = synthetic_prescribed_atmosphere(arch)
        radiation = synthetic_prescribed_radiation(arch)
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

        bottom_height = synthetic_bottom_height(grid;
                                                minimum_depth = 10,
                                                interpolation_passes = 5,
                                                major_basins = 1)

        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

        free_surface = SplitExplicitFreeSurface(grid; substeps=20)
        ocean = ocean_simulation(grid; free_surface)

        @test NumericalEarth.Oceans.get_radiative_forcing(ocean) isa NumericalEarth.Oceans.TwoColorRadiation

        atmosphere = synthetic_prescribed_atmosphere(arch)
        radiation = synthetic_prescribed_radiation(arch)

        # Fluxes are computed when the model is constructed, so we just test that this works.
        @test begin
            coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
            time_step!(coupled_model, 1)
            true
        end

        #####
        ##### Ocean with prescribed atmosphere and land
        #####

        @info "Testing OceanOnlyModel with PrescribedLand on $A..."
        land = synthetic_prescribed_land(arch)

        @test begin
            ocean_with_land = ocean_simulation(grid; free_surface)
            coupled_model = OceanOnlyModel(ocean_with_land; atmosphere, land, radiation)

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
        u_lateral_bcs = FieldBoundaryConditions(east = NormalFlowBoundaryCondition(0.1))

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
        @test typeof(u_bcs.top) == typeof(reference_u_bcs.top)
        @test typeof(u_bcs.bottom) == typeof(reference_u_bcs.bottom)
        @test typeof(u_bcs.immersed) == typeof(reference_u_bcs.immersed)
    end
end

@testset "Ocean exchanger state covers the halo columns the flux kernels visit" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "on $A" begin
            # Bounded, so nothing but the conversion kernel itself can put the ocean temperature in
            # the exchanger's halo columns (interface_kernel_parameters spans 0:N+1).
            grid = RectilinearGrid(arch, size = (8, 6, 4), halo = (5, 5, 5),
                                   x = (0, 80e3), y = (0, 60e3), z = (-100, 0),
                                   topology = (Bounded, Bounded, Bounded))

            ocean = ocean_simulation(grid; momentum_advection = nothing, tracer_advection = nothing,
                                     closure = nothing, bottom_drag_coefficient = 0, warn = false)
            set!(ocean.model, T = 15, S = 35)

            model = OceanOnlyModel(ocean)
            Oceananigans.TimeSteppers.update_state!(model)

            Nx, Ny, _ = size(grid)
            Hx, Hy, _ = Oceananigans.Grids.halo_size(grid)
            T = Array(parent(model.interfaces.exchanger.ocean.state.T))
            T₀ = T[Hx+1, Hy+1, 1]

            # Halo columns and rows, corners excluded (the ocean's own tracer corners are not filled)
            edges = vcat(T[Hx, Hy+1:Hy+Ny, 1], T[Hx+Nx+1, Hy+1:Hy+Ny, 1],
                         T[Hx+1:Hx+Nx, Hy, 1], T[Hx+1:Hx+Nx, Hy+Ny+1, 1])

            @test all(isapprox.(edges, T₀; rtol = 1e-4))
        end
    end
end
