using NumericalEarth
using NumericalEarth.DataWrangling: download_dataset, metadata_path
using NumericalEarth.DataWrangling.ORCA: default_south_rows_to_remove
using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using NCDatasets
using Statistics
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
    @test grid.underlying_grid.Ny == 333 - south_rows_to_remove

    grid = ORCAGrid(CPU(); dataset=ORCA1(), Nz=5, z=(-5000, 0), halo=(4, 4, 4), south_rows_to_remove=0)

    # Default returns ImmersedBoundaryGrid with bathymetry
    @test grid isa ImmersedBoundaryGrid
    underlying = grid.underlying_grid
    @test underlying isa Oceananigans.Grids.OrthogonalSphericalShellGrid
    @test underlying isa TripolarGrid
    @test underlying.Nx == 362
    @test underlying.Ny == 333
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
    @test grid.Ny == 333 - default_south_rows_to_remove(ORCA1())
    @test grid.Nz == 5
end

@testset "ORCAGrid with south_rows_to_remove" begin
    Nremove = 40
    grid = ORCAGrid(CPU(); dataset=ORCA1(), Nz=5, z=(-5000, 0), halo=(4, 4, 4),
                    south_rows_to_remove=Nremove)

    @test grid isa ImmersedBoundaryGrid
    underlying = grid.underlying_grid
    @test underlying.Nx == 362
    @test underlying.Ny == 333 - Nremove
    @test underlying.Nz == 5
end

@testset "ORCAGrid metric consistency" begin
    grid = ORCAGrid(CPU(); dataset=ORCA1(), Nz=5, z=(-5000, 0), halo=(4, 4, 4), with_bathymetry=false)

    Nx, Ny = grid.Nx, grid.Ny
    Hx, Hy = grid.Hx, grid.Hy

    # No NaNs or Infs in any grid data
    for name in (:λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ,
                 :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ,
                 :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ,
                 :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
                 :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ)
        data = getproperty(grid, name)
        @test all(isfinite, data) == true
    end

    # All interior metrics (Δx, Δy, Az) are strictly positive
    # Check only interior points to avoid halo issues
    for name in (:Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ,
                 :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
                 :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ)
        data = getproperty(grid, name)
        interior = data[1:Nx, 1:Ny]
        @test all(x -> x > 0, interior) == true
    end

    # Face-x longitude is west of Center-x longitude (stagger check)
    # At mid-latitudes (away from poles), Face[i] should be ≤ Center[i] in longitude.
    # Check a mid-latitude row (away from fold and south boundary).
    jmid = Ny ÷ 2
    λF   = grid.λᶠᶜᵃ[1:Nx, jmid]
    λC   = grid.λᶜᶜᵃ[1:Nx, jmid]
    # Most Face longitudes should be less than the corresponding Center longitude
    # (allowing for wraparound near ±180°). Count how many satisfy Face < Center.
    n_west = count(i -> begin
        δ = λC[i] - λF[i]
        # Handle wraparound: if δ < -180, add 360
        δ < -180 && (δ += 360)
        δ > 180 && (δ -= 360)
        δ > 0
    end, 1:Nx)

    @test n_west / Nx > 0.95  # vast majority should satisfy this

    # Face-y latitude is south of Center-y latitude (stagger check)
    # At interior points, Face[j] should be < Center[j] in latitude.
    imid = Nx ÷ 2
    φF   = grid.φᶜᶠᵃ[imid, 1:Ny]
    φC   = grid.φᶜᶜᵃ[imid, 1:Ny-1]  # Center has Ny-1 interior points
    nsouth = count(j -> φF[j] < φC[j], 1:length(φC))
    @test nsouth / length(φC) > 0.95

    # Periodic overlap: first and last unique columns should be consistent
    # After filling halos, the periodic halo should smoothly wrap.
    # Check that Δx at the periodic boundary has no discontinuity.
    jmid = Ny ÷ 2
    Δx = grid.Δxᶜᶜᵃ[:, jmid]
    # The relative jump from column Nx to column 1 (via periodic halo)
    # should be similar to the jump between adjacent interior columns
    interior_variation = maximum(abs, diff(Array(Δx[1:Nx]))) / mean(Δx[1:Nx])
    boundary_jump = abs(Δx[Nx] - Δx[1]) / mean(Δx[1:Nx])
    @test boundary_jump < 10 * interior_variation + 1e-10
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
