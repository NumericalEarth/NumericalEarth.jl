include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, BoundingBoxOffset, region_info, restrict,
    file_cell_index
using Oceananigans.Grids: Bounded, Flat

# Build the native grid of `bbox` on a global lattice, the way `construct_native_grid` does,
# and the file coordinates of a window around it: `before` native cells to the west/south of
# the grid and `after` cells to the east/north. The grid's first cell is then file cell
# `before + 1`, so `region_info` must report an offset of exactly `before`.
function lattice_offset(bbox, longitude_interfaces, latitude_interfaces, Nx_global, Ny_global,
                        before, after; FT = Float32)

    longitude, Nx = restrict(bbox.longitude, longitude_interfaces, Nx_global)
    latitude,  Ny = restrict(bbox.latitude,  latitude_interfaces,  Ny_global)

    Δλ = (last(longitude_interfaces) - first(longitude_interfaces)) / Nx_global
    Δφ = (last(latitude_interfaces)  - first(latitude_interfaces))  / Ny_global

    grid = LatitudeLongitudeGrid(CPU(), FT; size = (Nx, Ny), longitude, latitude,
                                 topology = (Bounded, Bounded, Flat))
    field = Field{Center, Center, Nothing}(grid)

    λc = [first(longitude) + (i - 1/2 - before) * Δλ for i in 1:(Nx + before + after)]
    φc = [first(latitude)  + (j - 1/2 - before) * Δφ for j in 1:(Ny + before + after)]

    return region_info(bbox, field, λc, φc)
end

@testset "Region offset on a fine grid" begin
    # A 300 m land product: 1/336°, interfaces on whole degrees.
    Δ = 1/336
    longitude_interfaces = (-180, 180)
    latitude_interfaces = (-60, 80)
    Nx_global = round(Int, 360 / Δ)
    Ny_global = round(Int, 140 / Δ)

    bbox = BoundingBox(longitude = (-91.6, -91.2), latitude = (37.2, 37.6))

    # The grid's nodes are Float32 and land a few units in the last place (ULPs) off the file's
    # Float64 coordinates.
    # The offset must still come out exactly, whatever margin the file carries.
    for before in 0:3, after in 0:3
        @test lattice_offset(bbox, longitude_interfaces, latitude_interfaces,
                             Nx_global, Ny_global, before, after) ==
              BoundingBoxOffset(before, before, 0)
    end

    # The shape a server-side subset actually delivers: no margin on the west/south side,
    # spare cells only on the east/north. The offset is zero and must not drift into the
    # slack the trailing cells create.
    @test lattice_offset(bbox, longitude_interfaces, latitude_interfaces,
                         Nx_global, Ny_global, 0, 10) == BoundingBoxOffset(0, 0, 0)

    # A Float64 grid must land on the same answer.
    @test lattice_offset(bbox, longitude_interfaces, latitude_interfaces,
                         Nx_global, Ny_global, 2, 2; FT = Float64) ==
          BoundingBoxOffset(2, 2, 0)

    # Away from the prime meridian the longitude wrap has to map [0, 360] onto the file's
    # convention without losing the cell.
    east = BoundingBox(longitude = (268.4, 268.8), latitude = (37.2, 37.6))
    @test lattice_offset(east, longitude_interfaces, latitude_interfaces,
                         Nx_global, Ny_global, 2, 2) == BoundingBoxOffset(2, 2, 0)
end

@testset "Region offset on a coarse grid" begin
    # ERA5's 0.25° lattice.
    longitude_interfaces = (-0.125, 359.875)
    latitude_interfaces = (-90, 90)
    Nx_global = 1440
    Ny_global = 721

    bbox = BoundingBox(longitude = (-62, -58), latitude = (16, 20))

    for before in 0:2, after in 0:2
        @test lattice_offset(bbox, longitude_interfaces, latitude_interfaces,
                             Nx_global, Ny_global, before, after) ==
              BoundingBoxOffset(before, before, 0)
    end
end

@testset "File cell index tolerance" begin
    # A Float32 node promoted to Float64 lands a few ulps off the label it should match; the
    # index must not slip a cell. Half a cell past a label belongs to the next one.
    Δ = 1/336
    hc = [37.2 + (j - 1) * Δ for j in 1:64]

    @test file_cell_index(hc[10], hc) == 10
    @test file_cell_index(Float64(Float32(hc[10])), hc) == 10
    @test file_cell_index(prevfloat(hc[10], 4), hc) == 10
    @test file_cell_index(hc[10] + Δ/2, hc) == 11

    # A single coordinate has no spacing to scale by, and must not error.
    @test file_cell_index(37.2, [37.2]) == 1

    # A node before the first label reports an index below 1: the file starts east of the read.
    @test file_cell_index(hc[1] - 2Δ, hc) == -1
end
