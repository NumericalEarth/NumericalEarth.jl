using NumericalEarth
using NumericalEarth.DataWrangling: download_dataset, metadata_path
using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using NCDatasets
using Test

@testset "ORCA1 Metadatum construction" begin
    bathy_md = Metadatum(:bottom_height; dataset=ORCA1())
    @test bathy_md.name == :bottom_height
    @test bathy_md.dataset isa ORCA1

    mesh_md = Metadatum(:mesh_mask; dataset=ORCA1())
    @test mesh_md.name == :mesh_mask
    @test mesh_md.dataset isa ORCA1
end

@testset "ORCA1Grid construction" begin
    grid = ORCA1Grid(CPU(); Nz=5, z=(-5000, 0), halo=(4, 4, 4))

    @test grid isa Oceananigans.Grids.OrthogonalSphericalShellGrid
    @test grid isa TripolarGrid
    @test grid.Nx == 362
    @test grid.Ny == 332
    @test grid.Nz == 5

    # Coordinates span near-global domain
    @test minimum(grid.λᶜᶜᵃ) < -179
    @test maximum(grid.λᶜᶜᵃ) > 179
    @test minimum(grid.φᶜᶜᵃ) < -80
    @test maximum(grid.φᶜᶜᵃ) > 80
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
