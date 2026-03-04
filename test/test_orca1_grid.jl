using NumericalEarth
using NumericalEarth.DataWrangling: download_dataset, metadata_path
using NumericalEarth.DataWrangling.ORCA: default_south_rows_to_remove
using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using NCDatasets
using Test

@testset "ORCA1 Metadatum construction" begin
    bathy_meta = Metadatum(:bottom_height; dataset=ORCA1())
    @test bathy_meta.name == :bottom_height
    @test bathy_meta.dataset isa ORCA1

    mesh_meta = Metadatum(:mesh_mask; dataset=ORCA1())
    @test mesh_meta.name == :mesh_mask
    @test mesh_meta.dataset isa ORCA1
end

@testset "ORCAGrid with ORCA1 dataset" begin
    south_rows_to_remove = 43
    grid = ORCAGrid(CPU(); dataset=ORCA1(), Nz=5, z=(-5000, 0), halo=(4, 4, 4), south_rows_to_remove)
    @test grid.underlying_grid.Ny == 332 - south_rows_to_remove

    grid = ORCAGrid(CPU(); dataset=ORCA1(), Nz=5, z=(-5000, 0), halo=(4, 4, 4), south_rows_to_remove=0)

    # Default returns ImmersedBoundaryGrid with bathymetry
    @test grid isa ImmersedBoundaryGrid
    underlying = grid.underlying_grid
    @test underlying isa Oceananigans.Grids.OrthogonalSphericalShellGrid
    @test underlying isa TripolarGrid
    @test underlying.Nx == 362
    @test underlying.Ny == 332
    @test underlying.Nz == 5

    # Coordinates span near-global domain
    @test minimum(underlying.λᶜᶜᵃ) < -179
    @test maximum(underlying.λᶜᶜᵃ) > 179
    @test minimum(underlying.φᶜᶜᵃ) < -80
    @test maximum(underlying.φᶜᶜᵃ) > 80
end

@testset "ORCAGrid without bathymetry" begin
    grid = ORCAGrid(CPU(); dataset=ORCA1(), Nz=5, z=(-5000, 0), halo=(4, 4, 4),
                    with_bathymetry=false)

    @test grid isa Oceananigans.Grids.OrthogonalSphericalShellGrid
    @test grid isa TripolarGrid
    @test !(grid isa ImmersedBoundaryGrid)
    @test grid.Nx == 362
    @test grid.Ny == 332 - default_south_rows_to_remove(ORCA1())
    @test grid.Nz == 5
end

@testset "ORCAGrid with south_rows_to_remove" begin
    Nremove = 40
    grid = ORCAGrid(CPU(); dataset=ORCA1(), Nz=5, z=(-5000, 0), halo=(4, 4, 4),
                    south_rows_to_remove=Nremove)

    @test grid isa ImmersedBoundaryGrid
    underlying = grid.underlying_grid
    @test underlying.Nx == 362
    @test underlying.Ny == 332 - Nremove
    @test underlying.Nz == 5
end

@testset "ORCA1 bathymetry retrieval" begin
    bathy_md = Metadatum(:bottom_height; dataset=ORCA1())
    download_dataset(bathy_md)
    path = metadata_path(bathy_md)
    @test isfile(path)

    ds = Dataset(path)
    bathy = ds["Bathymetry"][:, :]
    close(ds)

    @test size(bathy) == (362, 332)
    @test maximum(bathy) > 5000  # Deep ocean
end
