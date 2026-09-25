include("runtests_setup.jl")

using MPI
MPI.Init()

using CFTime
using Dates
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.NestedModels: blend_parent_terrain!
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: reconstruct_global_grid, reconstruct_global_field

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

@testset "Distributed terrain blending" begin
    Nx, Ny = 150, 90 # over 4 ranks these split 37/37/37/39 and 22/22/22/24: uneven in both directions
    width = 6

    child_terrain(x, y) = 1000 + 2x + 3y
    parent_terrain(x, y) = 500 - x + y / 2

    function blended_elevation(arch)
        grid = RectilinearGrid(arch; size = (Nx, Ny, 1), x = (0, Nx), y = (0, Ny), z = (0, 1),
                               topology = (Bounded, Bounded, Bounded))
        elevation = Field{Center, Center, Nothing}(grid)
        parent_surface = Field{Center, Center, Nothing}(grid)
        set!(elevation, child_terrain)
        set!(parent_surface, parent_terrain)
        blend_parent_terrain!(elevation, parent_surface; width)
        return reconstruct_global_field(elevation)
    end

    reference = Array(interior(blended_elevation(CPU()), :, :, 1))

    # Rank-local sizes would blend a frame around each rank's own subdomain, banding parent orography
    # along every interior seam: bands in x under Partition(4, 1), in y under Partition(1, 4), and a
    # cross under Partition(2, 2), which is the only case with an even split in both directions.
    for partition in (Partition(4, 1), Partition(1, 4), Partition(2, 2))
        blended = blended_elevation(Distributed(CPU(); partition))
        @test Array(interior(blended, :, :, 1)) == reference
    end
end

MPI.Finalize()
