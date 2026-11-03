include("runtests_setup.jl")

using Glob
using MPI

function run_mpi_script(filename, script; nranks=4)
    script_path = joinpath(@__DIR__, filename)
    write(script_path, script)

    try
        run(`$(mpiexec()) -n $(nranks) $(Base.julia_cmd()) -O0 $(script_path)`)
    finally
        rm(script_path, force=true)
    end

    return nothing
end

function distributed_ecco_download_script()
    return raw"""
    include("runtests_setup.jl")

    using MPI
    MPI.Init()

    using CFTime
    using Dates
    using NumericalEarth.DataWrangling: metadata_path
    using Oceananigans.DistributedComputations

    @testset "Distributed ECCO download" begin
        dates = DateTimeProlepticGregorian(1992, 1, 1) : Month(1) : DateTimeProlepticGregorian(1994, 4, 1)
        metadata = Metadata(:u_velocity; dataset=ECCO4Monthly(), dates)
        download(metadata)

        @root for metadatum in metadata
            @test isfile(metadata_path(metadatum))
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
    """
end

function distributed_bathymetry_interpolation_script()
    return raw"""
    include("runtests_setup.jl")

    using MPI
    MPI.Init()

    using Oceananigans.DistributedComputations

    @testset "Distributed Bathymetry interpolation" begin
        metadata = Metadatum(:bottom_height; dataset = SyntheticBathymetry())

        global_grid = LatitudeLongitudeGrid(CPU();
                                            size = (40, 40, 1),
                                            longitude = (0, 100),
                                            latitude = (0, 20),
                                            z = (0, 1))

        interpolation_passes = 4
        global_height = regrid_bathymetry(global_grid, metadata; interpolation_passes, cache = false)

        arch_x  = Distributed(CPU(), partition=Partition(4, 1))
        arch_y  = Distributed(CPU(), partition=Partition(1, 4))
        arch_xy = Distributed(CPU(), partition=Partition(2, 2))

        for arch in (arch_x, arch_y, arch_xy)
            local_grid = LatitudeLongitudeGrid(arch;
                                               size = (40, 40, 1),
                                               longitude = (0, 100),
                                               latitude = (0, 20),
                                               z = (0, 1))

            local_height = regrid_bathymetry(local_grid, metadata; interpolation_passes, cache = false)

            Nx, Ny, _ = size(local_grid)
            rx, ry, _ = arch.local_index
            irange = (rx - 1) * Nx + 1 : rx * Nx
            jrange = (ry - 1) * Ny + 1 : ry * Ny

            begin
                @test interior(global_height, irange, jrange, 1) == interior(local_height, :, :, 1)
            end
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
    """
end

function distributed_grids_script()
    return raw"""
    include("runtests_setup.jl")

    using MPI
    MPI.Init()

    using NumericalEarth.DataWrangling.ORCA: ORCAOne
    using Oceananigans.DistributedComputations
    using Oceananigans.DistributedComputations: all_reduce, concatenate_local_sizes, local_size
    using Oceananigans.Grids: Sides, Closed
    using Oceananigans.Operators: Azᶜᶜᶜ
    using NumericalEarth.Lands: build_flux_routing
    using NumericalEarth.Oceans: river_mouth_vertical_diffusivity
    using Oceananigans.BoundaryConditions: regularize_boundary_condition, LeftBoundary
    using Oceananigans.Grids: topology

    # The window of the serial grid held by this rank
    function rank_window(arch, global_size, local_size_)
        sizes = local_size(arch, global_size)
        rx, ry, _ = arch.local_index
        istart = 1 + sum(concatenate_local_sizes(sizes, arch, 1)[1:rx-1])
        jstart = 1 + sum(concatenate_local_sizes(sizes, arch, 2)[1:ry-1])
        return istart:istart+local_size_[1]-1, jstart:jstart+local_size_[2]-1
    end

    # Each rank's metrics are its window of the serial grid
    @testset "Distributed ORCAGrid" begin
        orca(arch) = ORCAGrid(arch; dataset = ORCAOne(), Nz = 5, z = (-5000, 0), with_bathymetry = false)

        global_grid = orca(CPU())

        for partition in (Partition(1, 4), Partition(2, 2))
            arch = Distributed(CPU(); partition)
            local_grid = orca(arch)
            irange, jrange = rank_window(arch, size(global_grid), size(local_grid))

            for metric in (:λᶜᶜᵃ, :φᶜᶜᵃ, :Δxᶜᶜᵃ, :Δyᶜᶜᵃ, :Azᶜᶜᵃ)
                local_metric  = getproperty(local_grid, metric)[1:length(irange), 1:length(jrange)]
                global_metric = getproperty(global_grid, metric)[irange, jrange]
                @test local_metric == global_metric
            end
        end
    end

    # Each rank's bottom height, basins removed, is its window of the serial one
    @testset "Distributed ORCAGrid bathymetry" begin
        orca(arch) = ORCAGrid(arch; dataset = ORCAOne(), Nz = 5, z = (-5000, 0), major_basins = 1)

        global_grid = orca(CPU())

        for partition in (Partition(1, 4), Partition(2, 2))
            arch = Distributed(CPU(); partition)
            local_grid = orca(arch)
            irange, jrange = rank_window(arch, size(global_grid), size(local_grid))

            local_bottom  = local_grid.immersed_boundary.bottom_height[1:length(irange), 1:length(jrange), 1]
            global_bottom = global_grid.immersed_boundary.bottom_height[irange, jrange, 1]
            @test local_bottom == global_bottom
        end
    end

    # Every rank regularizes the west boundary condition, but only the westmost ranks keep it, so a source
    # spanning only the child's west edge must not be checked against the other ranks' local edges.
    @testset "Distributed interpolated boundary source bracket" begin
        loc = (Center(), Center(), Center())

        for (longitude, valid) in (((-101, -99), true), ((-99, -97), false))
            source_grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 4), longitude, latitude = (29, 41), z = (0, 1),
                                                topology = (Bounded, Bounded, Bounded))
            source = NumericalEarth.NestedModels.Interpolated(CenterField(source_grid))

            for partition in (Partition(4, 1), Partition(2, 2))
                grid = LatitudeLongitudeGrid(Distributed(CPU(); partition); size = (20, 20, 4), longitude = (-100, -90),
                                             latitude = (30, 40), z = (0, 1), topology = (Bounded, Bounded, Bounded))
                west_rank = topology(grid, 1) <: Sides{Closed}

                if valid || !west_rank
                    @test regularize_boundary_condition(source, grid, loc, 1, LeftBoundary) isa NumericalEarth.NestedModels.Interpolated
                else
                    @test_throws ArgumentError regularize_boundary_condition(source, grid, loc, 1, LeftBoundary)
                end
            end
        end
    end

    # Every mouth is routed by exactly one rank, so the ranks together deposit the serial total
    @testset "Distributed river routing" begin
        function coastal_grid(arch)
            underlying = LatitudeLongitudeGrid(arch; size = (20, 20, 1), longitude = (-10, 10), latitude = (-10, 10),
                                               z = (-10, 0), halo = (4, 4, 4))
            return ImmersedBoundaryGrid(underlying, GridFittedBottom((λ, φ) -> ifelse(λ < 0, -10, 10)))
        end

        # A JRA55-like per-area flux on the coastal land column of a finer source grid
        source_grid = LatitudeLongitudeGrid(CPU(); size = (40, 40), longitude = (-10, 10), latitude = (-10, 10),
                                            topology = (Bounded, Bounded, Flat))
        flux_time_series = FieldTimeSeries{Center, Center, Nothing}(source_grid, [0.0])
        coast_i = findfirst(>(0), Array(λnodes(source_grid, Center(), Center(), Center())))
        interior(flux_time_series[1])[coast_i, :, 1] .= 1:40
        source_flux = Array(interior(flux_time_series[1]))[:, :, 1]

        function deposited_mass_rate(routing, grid)
            ti, tj, offsets = Array(routing.target_i), Array(routing.target_j), Array(routing.offsets)
            oi, oj, weight = Array(routing.contribution_outlet_i), Array(routing.contribution_outlet_j), Array(routing.contribution_weight)
            return sum((weight[k] * source_flux[oi[k], oj[k]] * Azᶜᶜᶜ(ti[c], tj[c], 1, grid)
                        for c in eachindex(ti) for k in offsets[c]:offsets[c+1]-1); init = 0.0)
        end

        serial_grid = coastal_grid(CPU())
        serial_mass_rate = deposited_mass_rate(build_flux_routing(serial_grid, flux_time_series), serial_grid)
        @test serial_mass_rate ≈ sum(source_flux[coast_i, j] * Azᶜᶜᶜ(coast_i, j, 1, source_grid) for j in 1:40)

        for partition in (Partition(1, 4), Partition(2, 2))
            arch = Distributed(CPU(); partition)
            local_grid = coastal_grid(arch)
            routing = build_flux_routing(local_grid, flux_time_series)
            @test all_reduce(+, deposited_mass_rate(routing, local_grid), arch) ≈ serial_mass_rate

            river_mixing = river_mouth_vertical_diffusivity(local_grid, (; rivers = routing))
            @test count(>(0), interior(river_mixing.κ.parameters)) == length(routing.target_i)
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
    """
end

function distributed_earth_system_checkpointing_script()
    return raw"""
    include("runtests_setup.jl")

    using Glob
    using MPI
    MPI.Init()

    using Oceananigans.OutputWriters: Checkpointer

    @testset "Distributed EarthSystemModel checkpointing" begin
        mpi_ranks = MPI.Comm_size(MPI.COMM_WORLD)

        # Use an x-only partition that matches the active MPI communicator.
        arch = Distributed(CPU(), partition=Partition(mpi_ranks, 1))

        # Choose a size divisible by mpi_ranks for robust domain decomposition.
        grid = LatitudeLongitudeGrid(arch;
                                     size = (12 * mpi_ranks, 24, 4),
                                     z = (-100, 0),
                                     latitude = (-10, 10),
                                     longitude = (0, 24),
                                     halo = (6, 6, 6))

        function make_coupled_model(grid, arch)
            @inline hi(λ, φ) = φ > 70 || φ < -70

            ocean = ocean_simulation(grid, free_surface=SplitExplicitFreeSurface(grid; substeps=30), closure=nothing)
            set!(ocean.model, T=20, S=35, u=0.01, v=-0.005)

            sea_ice = sea_ice_simulation(grid, ocean)
            set!(sea_ice.model, h=hi, ℵ=hi)

            backend = JRA55NetCDFBackend(4)
            atmosphere = JRA55PrescribedAtmosphere(arch; backend)

            return OceanSeaIceModel(ocean, sea_ice; atmosphere)
        end

        # Reference run: run to 3, then continue to 6.
        model = make_coupled_model(grid, arch)
        simulation = Simulation(model, Δt=60, stop_iteration=3)
        run!(simulation)

        simulation = Simulation(model, Δt=60, stop_iteration=6)
        run!(simulation)

        ref_T  = Array(interior(model.ocean.model.tracers.T))
        ref_S  = Array(interior(model.ocean.model.tracers.S))
        ref_u  = Array(interior(model.ocean.model.velocities.u))
        ref_v  = Array(interior(model.ocean.model.velocities.v))
        ref_h  = Array(interior(model.sea_ice.model.ice_thickness))
        ref_ui = Array(interior(model.sea_ice.model.velocities.u))
        ref_vi = Array(interior(model.sea_ice.model.velocities.v))
        ref_time = model.clock.time
        ref_iteration = model.clock.iteration

        # Checkpointed run: run to 3 and checkpoint.
        model = make_coupled_model(grid, arch)
        simulation = Simulation(model, Δt=60, stop_iteration=3)

        prefix = "distributed_osm_checkpointer_test_rank$(MPI.Comm_rank(MPI.COMM_WORLD))"
        simulation.output_writers[:checkpointer] = Checkpointer(simulation.model;
                                                                schedule = IterationInterval(3),
                                                                prefix = prefix)
        run!(simulation)

        @test !isempty(glob("$(prefix)_iteration3*.jld2"))

        # Recreate and restore from latest checkpoint.
        model = make_coupled_model(grid, arch)
        simulation = Simulation(model, Δt=60, stop_iteration=6)
        simulation.output_writers[:checkpointer] = Checkpointer(model;
                                                                schedule = IterationInterval(3),
                                                                prefix = prefix)

        set!(simulation; checkpoint=:latest)
        @test simulation.model.clock.iteration == 3

        set!(simulation; iteration=3)
        @test simulation.model.clock.iteration == 3

        run!(simulation)

        T  = Array(interior(model.ocean.model.tracers.T))
        S  = Array(interior(model.ocean.model.tracers.S))
        u  = Array(interior(model.ocean.model.velocities.u))
        v  = Array(interior(model.ocean.model.velocities.v))
        h  = Array(interior(model.sea_ice.model.ice_thickness))
        ui = Array(interior(model.sea_ice.model.velocities.u))
        vi = Array(interior(model.sea_ice.model.velocities.v))

        # Match serial checkpointer tolerances.
        @test T ≈ ref_T rtol=1e-13
        @test S ≈ ref_S rtol=1e-13
        @test h ≈ ref_h rtol=1e-13
        @test u ≈ ref_u rtol=1e-10
        @test v ≈ ref_v rtol=1e-10
        @test ui ≈ ref_ui rtol=1e-10
        @test vi ≈ ref_vi rtol=1e-10
        @test model.clock.time == ref_time
        @test model.clock.iteration == ref_iteration

        rm.(glob("$(prefix)_iteration*.jld2"), force=true)
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
    """
end

@testset "Distributed ECCO download" begin
    run_mpi_script("distributed_ecco_download.jl", distributed_ecco_download_script())
end

@testset "Distributed Bathymetry interpolation" begin
    run_mpi_script("distributed_bathymetry_interpolation.jl", distributed_bathymetry_interpolation_script())
end

@testset "Distributed grids and river routing" begin
    run_mpi_script("distributed_grids.jl", distributed_grids_script())
end

@testset "Distributed EarthSystemModel checkpointing" begin
    run_mpi_script("distributed_earth_system_checkpointing.jl", distributed_earth_system_checkpointing_script())
end
