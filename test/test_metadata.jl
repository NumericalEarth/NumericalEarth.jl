include("runtests_setup.jl")

using NumericalEarth.DataWrangling: Column, Linear, Nearest,
                                    BoundingBox, dataset_location,
                                    restrict_location, native_grid

using Oceananigans: RectilinearGrid, LatitudeLongitudeGrid, location
using Oceananigans.Grids: topology, Flat, Bounded, Periodic

@testset "Column construction" begin
    col = Column(35.1, 50.1)
    @test col.longitude == 35.1
    @test col.latitude == 50.1
    @test col.z === nothing
    @test col.interpolation isa Linear

    col_nearest = Column(35.1, 50.1; interpolation=Nearest())
    @test col_nearest.interpolation isa Nearest

    col_z = Column(35.1, 50.1; z=(-400, 0))
    @test col_z.z == (-400, 0)
end

@testset "Column isa checks" begin
    @test Column(0, 0) isa Column
    @test !(BoundingBox(longitude=(0, 10), latitude=(0, 10)) isa Column)
    @test !(nothing isa Column)
end

@testset "restrict_location" begin
    # Column reduces horizontal locations to Nothing
    @test restrict_location((Center, Center, Center), Column(0, 0)) == (Nothing, Nothing, Center)
    @test restrict_location((Face, Center, Center), Column(0, 0)) == (Nothing, Nothing, Center)
    @test restrict_location((Center, Face, Center), Column(0, 0)) == (Nothing, Nothing, Center)
    @test restrict_location((Center, Center, Nothing), Column(0, 0)) == (Nothing, Nothing, Nothing)

    # BoundingBox and nothing leave location unchanged
    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    @test restrict_location((Face, Center, Center), bbox) == (Face, Center, Center)
    @test restrict_location((Center, Center, Center), nothing) == (Center, Center, Center)
end

@testset "dataset_location fallback" begin
    # Default fallback returns (Center, Center, Center)
    @test dataset_location(ECCO2Monthly(), :temperature) == (Center, Center, Center)
    @test dataset_location(ECCO4Monthly(), :temperature) == (Center, Center, Center)

    # ECCO staggered velocities
    @test dataset_location(ECCO4Monthly(), :u_velocity) == (Face, Center, Center)
    @test dataset_location(ECCO4Monthly(), :v_velocity) == (Center, Face, Center)

    # ECCO 2D fields
    @test dataset_location(ECCO4Monthly(), :free_surface) == (Center, Center, Nothing)

    # Non-ECCO datasets use the generic fallback
    @test dataset_location(JRA55.RepeatYearJRA55(), :temperature) == (Center, Center, Center)
end

@testset "location(metadata) with Column region" begin
    # Column metadata: location is restricted
    col = Column(35.1, 50.1)
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
    @test location(md) == (Nothing, Nothing, Center)

    # ECCO velocity + Column: horizontal locations dropped
    md_u = Metadatum(:u_velocity; dataset=ECCO4Monthly(), region=col)
    @test location(md_u) == (Nothing, Nothing, Center)

    # ECCO 2D field + Column
    md_fs = Metadatum(:free_surface; dataset=ECCO4Monthly(), region=col)
    @test location(md_fs) == (Nothing, Nothing, Nothing)

    # No region: full dataset location
    md_full = Metadatum(:u_velocity; dataset=ECCO4Monthly())
    @test location(md_full) == (Face, Center, Center)

    # BoundingBox: full dataset location
    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    md_bbox = Metadatum(:u_velocity; dataset=ECCO4Monthly(), region=bbox)
    @test location(md_bbox) == (Face, Center, Center)
end

@testset "native_grid with Column region" begin
    col = Column(35.1, 50.1)
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
    grid = native_grid(md)

    @test grid isa RectilinearGrid
    @test topology(grid) == (Flat, Flat, Bounded)
    _, _, Nz, _ = size(md)
    @test size(grid) == (1, 1, Nz)
end

@testset "native_grid without region" begin
    md = Metadatum(:temperature; dataset=ECCO4Monthly())
    grid = native_grid(md)

    @test grid isa LatitudeLongitudeGrid
    Nx, Ny, Nz, _ = size(md)
    @test size(grid) == (Nx, Ny, Nz)
end

@testset "native_grid with BoundingBox region" begin
    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=bbox)
    grid = native_grid(md)

    @test grid isa LatitudeLongitudeGrid
    # Grid should be smaller than the full global grid
    Nx_full, Ny_full, _, _ = size(md)
    Nx, Ny, Nz = size(grid)
    @test Nx < Nx_full
    @test Ny < Ny_full
end

@testset "Metadata region keyword" begin
    # region keyword replaces BoundingBox
    col = Column(35.1, 50.1)
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
    @test md.region isa Column
    @test md.region.longitude == 35.1

    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    md2 = Metadatum(:temperature; dataset=ECCO4Monthly(), region=bbox)
    @test md2.region isa BoundingBox

    md3 = Metadatum(:temperature; dataset=ECCO4Monthly())
    @test md3.region === nothing
end

@testset "Metadata iteration propagates region" begin
    col = Column(35.1, 50.1)
    md = Metadata(:temperature; dataset=ECCO4Monthly(), region=col,
                  start_date=DateTime(1992, 1, 1), end_date=DateTime(1992, 3, 1))

    for sub_md in md
        @test sub_md.region === col
    end

    @test first(md).region === col
    @test last(md).region === col
    @test md[1].region === col
end
