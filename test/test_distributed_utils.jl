include("runtests_setup.jl")

using Glob
using MPI
MPI.Init()

using CFTime
using Dates
using NCDatasets
using NumericalEarth.DataWrangling: metadata_path
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: reconstruct_global_grid
using Oceananigans.OutputWriters: Checkpointer

# We start by building a fake bathymetry on rank 0 and save it to file
rm("./trivial_bathymetry.nc", force=true)

res = 0.5 # degrees
λ = -180+res/2:res:180-res/2
φ = 0:res:50

Nλ = length(λ)
Nφ = length(φ)

@root begin
    ds = NCDataset("./trivial_bathymetry.nc", "c")

    # Define the dimension "lon" and "lat" with the size Nλ and Nφ respectively
    defDim(ds, "lon", Nλ)
    defDim(ds, "lat", Nφ)
    defVar(ds, "lat", Float32, ("lat", ))
    defVar(ds, "lon", Float32, ("lon", ))

    # Define the variables z
    z = defVar(ds, "z", Float32, ("lon", "lat"))

    # Generate some example data
    data = [Float32(-i) for i = 1:Nλ, j = 1:Nφ]

    # write a the complete data set
    ds["lon"][:] = λ
    ds["lat"][:] = φ
    z[:, :] = data

    close(ds)
end

struct TrivalBathymetry end

import NumericalEarth.DataWrangling: download_dataset, z_interfaces, longitude_interfaces, latitude_interfaces, metadata_filename

download_dataset(::Metadatum{<:TrivalBathymetry}) = nothing
Base.size(::TrivalBathymetry) = (Nλ, Nφ, 1)
Base.size(::TrivalBathymetry, variable) = (Nλ, Nφ, 1)
z_interfaces(::TrivalBathymetry) = (0, 1)
longitude_interfaces(::TrivalBathymetry) = (-180, 180)
latitude_interfaces(::TrivalBathymetry) = (0, 50)
metadata_filename(::TrivalBathymetry, name, date, bounding_box) = "trivial_bathymetry.nc"

@testset "Distributed ECCO download" begin
    dates = DateTimeProlepticGregorian(1992, 1, 1) : Month(1) : DateTimeProlepticGregorian(1994, 4, 1)
    metadata = Metadata(:u_velocity; dataset=ECCO4Monthly(), dates)
    download_dataset(metadata)

    @root for metadatum in metadata
        @test isfile(metadata_path(metadatum))
    end
end

@testset "Distributed Bathymetry interpolation" begin
    TrivialBathymetry_metadata = Metadata(:z, TrivalBathymetry(), nothing, nothing, ".")

    global_grid = LatitudeLongitudeGrid(CPU();
                                        size = (40, 40, 1),
                                        longitude = (0, 100),
                                        latitude = (0, 20),
                                        z = (0, 1))

    interpolation_passes = 4
    global_height = regrid_bathymetry(global_grid, TrivialBathymetry_metadata;
                                      interpolation_passes)

    arch_x  = Distributed(CPU(), partition=Partition(4, 1))
    arch_y  = Distributed(CPU(), partition=Partition(1, 4))
    arch_xy = Distributed(CPU(), partition=Partition(2, 2))

    for arch in (arch_x, arch_y, arch_xy)
        local_grid = LatitudeLongitudeGrid(arch;
                                           size = (40, 40, 1),
                                           longitude = (0, 100),
                                           latitude = (0, 20),
                                           z = (0, 1))

        local_height = regrid_bathymetry(local_grid, TrivialBathymetry_metadata;
                                         interpolation_passes)

        Nx, Ny, _ = size(local_grid)
        rx, ry, _ = arch.local_index
        irange = (rx - 1) * Nx + 1 : rx * Nx
        jrange = (ry - 1) * Ny + 1 : ry * Ny

        begin
            @test interior(global_height, irange, jrange, 1) == interior(local_height, :, :, 1)
        end
    end
end

@testset "Distributed EarthSystemModel checkpointing" begin
    mpi_ranks = MPI.Comm_size(MPI.COMM_WORLD)

    if mpi_ranks == 1
        @warn "Skipping distributed checkpointing test because only one MPI rank is active."
    else
        # Use an x-only partition that matches the active MPI communicator.
        arch = Distributed(CPU(), partition=Partition(mpi_ranks, 1))

        # Choose a size divisible by mpi_ranks for robust domain decomposition.
        grid = LatitudeLongitudeGrid(arch;
                                     size = (12 * mpi_ranks, 24, 4),
                                     z = (-100, 0),
                                     latitude = (-80, 80),
                                     longitude = (0, 360),
                                     halo = (6, 6, 6))

        function make_coupled_model(grid, arch)
            @inline hi(λ, φ) = φ > 70 || φ < -70

            ocean = ocean_simulation(grid, closure=nothing)
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
end

MPI.Finalize()
