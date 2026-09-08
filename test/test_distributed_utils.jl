include("runtests_setup.jl")

using MPI
MPI.Init()

using CFTime
using Dates
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.DataWrangling.ORCA: ORCAOne
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: concatenate_local_sizes, local_size, reconstruct_global_grid

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

# Each rank holds a contiguous window of the serial grid, and the last rank in a direction absorbs the
# remainder, so the window comes from the actual local sizes rather than an even split.
function rank_window(arch, global_size, local_size_)
    sizes = local_size(arch, global_size)
    rx, ry, _ = arch.local_index
    istart = 1 + sum(concatenate_local_sizes(sizes, arch, 1)[1:rx-1])
    jstart = 1 + sum(concatenate_local_sizes(sizes, arch, 2)[1:ry-1])
    return istart:istart+local_size_[1]-1, jstart:jstart+local_size_[2]-1
end

# The eORCA mesh is assembled globally and then cut to each rank's slice, so a rank's metrics must be
# the matching window of the serial grid. An x partition must be 1 or even, and x-only is rejected,
# because the northern fold pairs each rank with the one mirrored across the pole.
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

# The basin flood fill runs on rank 0 and is shared, so every rank's bottom height must agree with the
# serial one over its own window.
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

MPI.Finalize()
