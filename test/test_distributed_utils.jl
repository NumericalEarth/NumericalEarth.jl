include("runtests_setup.jl")

using MPI
MPI.Init()

using CFTime
using Dates
using NCDatasets
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.NestedModels: blend_parent_terrain!
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: reconstruct_global_grid, reconstruct_global_field

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

using Downloads: Downloads, download
import NumericalEarth.DataWrangling: z_interfaces, longitude_interfaces, latitude_interfaces, metadata_filename

Downloads.download(::Metadatum{<:TrivalBathymetry}) = nothing
Base.size(::TrivalBathymetry) = (Nλ, Nφ, 1)
Base.size(::TrivalBathymetry, variable) = (Nλ, Nφ, 1)
z_interfaces(::TrivalBathymetry) = (0, 1)
longitude_interfaces(::TrivalBathymetry) = (-180, 180)
latitude_interfaces(::TrivalBathymetry) = (0, 50)
metadata_filename(::TrivalBathymetry, name, date, region) = "trivial_bathymetry.nc"

@testset "Distributed ECCO download" begin
    dates = DateTimeProlepticGregorian(1992, 1, 1) : Month(1) : DateTimeProlepticGregorian(1994, 4, 1)
    metadata = Metadata(:u_velocity; dataset=ECCO4Monthly(), dates)
    download(metadata)

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
