include("runtests_setup.jl")

using MPI
MPI.Init()

using CFTime
using Dates
using NCDatasets
using NumericalEarth.DataWrangling: metadata_path, BoundingBox
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: reconstruct_global_grid


using CopernicusMarine

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

MPI.Finalize()
