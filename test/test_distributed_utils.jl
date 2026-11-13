include("runtests_setup.jl")

using MPI
MPI.Init()

using CFTime
using Dates
using NumericalEarth.DataWrangling: NearestNeighborInpainting, all_dates, inpainted_metadata_path, metadata_path, BoundingBox
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: update_field_time_series!
using Oceananigans.Grids: halo_size, node, MutableVerticalDiscretization
using NumericalEarth.DataWrangling.ORCA: ORCAOne
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: all_reduce, concatenate_local_sizes, local_size, reconstruct_global_grid
using Oceananigans.Grids: Sides, Closed
using Oceananigans.Operators: Azᶜᶜᶜ
using NumericalEarth.Lands: build_flux_routing
using NumericalEarth.Oceans: river_mouth_vertical_diffusivity
using Oceananigans.BoundaryConditions: regularize_boundary_condition, LeftBoundary
using Oceananigans.Grids: topology

using CopernicusMarine

@testset "Distributed ECCO download" begin
    dates = DateTimeProlepticGregorian(1992, 1, 1) : Month(1) : DateTimeProlepticGregorian(1994, 4, 1)
    metadata = Metadata(:u_velocity; dataset=ECCO4Monthly(), dates)
    download(metadata)

    @root for metadatum in metadata
        @test isfile(metadata_path(metadatum))
    end
end

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

@testset "Distributed inpainted-data cache" begin
    # one data directory for all ranks
    dir = MPI.bcast(mktempdir(), MPI.COMM_WORLD)
    metadatum  = Metadatum(:temperature; dataset = SyntheticOcean(), dir)
    inpainting = NearestNeighborInpainting(10)
    @root download(metadatum)

    # Only rank 0 needs this field: it inpaints it and caches it without waiting for the other ranks
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && Field(metadatum; inpainting)
    @test MPI.Allreduce(1, +, MPI.COMM_WORLD) == MPI.Comm_size(MPI.COMM_WORLD)
    @test isfile(inpainted_metadata_path(metadatum))

    cached = Field(metadatum; inpainting)
    fresh  = Field(metadatum; inpainting, cache_inpainted_data = false)
    @test interior(cached) == interior(fresh)
    MPI.Barrier(MPI.COMM_WORLD)    # the directory goes when rank 0 exits
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

@testset "Distributed GLORYS set! matches serial" begin
    # A bbox-restricted GLORYS field, set! onto a distributed target grid, must
    # reproduce the serial result rank-by-rank: each rank's local interior equals
    # the matching window of the field assembled on a single process.
    region = BoundingBox(longitude=(200, 210), latitude=(30, 40))
    md = Metadatum(:temperature; dataset=GLORYSDaily(), region)

    # The CopernicusMarine extension already guards the subset call with `@root`.
    download(md)
    MPI.Barrier(MPI.COMM_WORLD)

    target_grid(arch) = LatitudeLongitudeGrid(arch;
                                              size = (20, 20, 10),
                                              longitude = (201, 209),
                                              latitude = (31, 39),
                                              z = (-1000, 0))

    serial = CenterField(target_grid(CPU()))
    set!(serial, md; inpainting=nothing)
    serial_interior = Array(interior(serial))

    for partition in (Partition(4, 1), Partition(1, 4), Partition(2, 2))
        arch = Distributed(CPU(); partition)
        local_field = CenterField(target_grid(arch))
        set!(local_field, md; inpainting=nothing)

        Nx, Ny, _ = size(local_field.grid)
        rx, ry, _ = arch.local_index
        irange = (rx - 1) * Nx + 1 : rx * Nx
        jrange = (ry - 1) * Ny + 1 : ry * Ny

        @test Array(interior(local_field)) ≈ serial_interior[irange, jrange, :]
    end
end

@testset "Dataset FieldTimeSeries advanced on one rank" begin
    arch = Distributed(CPU(), partition = Partition(4, 1))
    grid = LatitudeLongitudeGrid(arch; size = (16, 8, 4), longitude = (0, 80), latitude = (-20, 20), z = (-1000, 0))
    dates = all_dates(SyntheticOcean(), :temperature)[1:4]
    metadata = Metadata(:temperature; dataset = SyntheticOcean(), dates)
    kw = (; time_indices_in_memory = 2, cache_inpainted_data = false)

    # Mid-run, only some ranks may advance a series (eg open-boundary data on boundary ranks)
    alone = FieldTimeSeries(metadata, grid; kw...)
    arch.local_rank == 0 && update_field_time_series!(alone, 3)
    @test all_reduce(min, arch.field_count[], arch) == all_reduce(max, arch.field_count[], arch)

    # The interpolated halo facing a neighbour matches what a halo exchange delivers
    together = FieldTimeSeries(metadata, grid; kw...)
    exchanged = CenterField(grid)
    interior(exchanged) .= interior(together[3])
    fill_halo_regions!(exchanged)
    Nx, Ny, Nz = size(grid)
    east_halo = (Nx+1:Nx+halo_size(grid)[1], 1:Ny, 1:Nz)

    if arch.local_rank == 0
        @test interior(alone[3]) ≈ interior(exchanged)
        @test alone[3].data[east_halo...] ≈ exchanged.data[east_halo...]
    end
end

@testset "Distributed interpolation onto a mutable grid" begin
    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = LatitudeLongitudeGrid(arch; size = (16, 8, 4), halo = (3, 3, 3), longitude = (0, 80), latitude = (-20, 20),
                                 z = MutableVerticalDiscretization((-1000, 0)))
    source_grid = LatitudeLongitudeGrid(CPU(); size = (60, 40, 4), longitude = (-30, 110), latitude = (-40, 40), z = (-1000, 0))
    source = CenterField(source_grid)
    set!(source, (λ, φ, z) -> λ + 2φ)
    fill_halo_regions!(source)

    for loc in ((Face(), Center(), Center()), (Center(), Face(), Center()), (Face(), Face(), Center()))
        field = Field(loc, grid)
        NumericalEarth.DataWrangling.interpolate_physical!(field, source)

        Nx, Ny, Nz = size(grid)
        Hx, Hy, _ = halo_size(grid)
        expected(is, js) = [node(i, j, 1, grid, loc...)[1] + 2 * node(i, j, 1, grid, loc...)[2] for i in is, j in js]

        @test field.data[1:Nx, 1:Ny, 1] ≈ expected(1:Nx, 1:Ny)

        # Halos facing a neighbour are interpolated, not exchanged
        arch.local_index[1] == 1 && @test field.data[Nx+1:Nx+Hx, 1:Ny, 1] ≈ expected(Nx+1:Nx+Hx, 1:Ny)
        arch.local_index[2] == 1 && @test field.data[1:Nx, Ny+1:Ny+Hy, 1] ≈ expected(1:Nx, Ny+1:Ny+Hy)
    end
end

MPI.Finalize()
