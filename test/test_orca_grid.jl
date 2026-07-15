include("runtests_setup.jl")
include("download_utils.jl")

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using NCDatasets
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.DataWrangling.ORCA: default_south_rows_to_remove
using Statistics
using Test

# Pre-download ORCAOne mesh_mask and bathymetry through the artifacts fallback so
# subsequent ORCAGrid(...) calls find the files locally even when Zenodo is down.
for name in (:mesh_mask, :bottom_height)
    md = Metadatum(name; dataset=ORCAOne())
    download_dataset_with_fallback(metadata_path(md); dataset_name="ORCAOne $name") do
        download(md)
    end
end

@testset "ORCAOne Metadatum construction" begin
    bathy_meta = Metadatum(:bottom_height; dataset=ORCAOne())
    @test bathy_meta.name == :bottom_height
    @test bathy_meta.dataset isa ORCAOne

    mesh_meta = Metadatum(:mesh_mask; dataset=ORCAOne())
    @test mesh_meta.name == :mesh_mask
    @test mesh_meta.dataset isa ORCAOne
end


@testset "ORCATwelfth Metadatum construction" begin
    bathy_meta = Metadatum(:bottom_height; dataset=ORCATwelfth())
    @test bathy_meta.name == :bottom_height
    @test bathy_meta.dataset isa ORCATwelfth

    mesh_meta = Metadatum(:mesh_mask; dataset=ORCATwelfth())
    @test mesh_meta.name == :mesh_mask
    @test mesh_meta.dataset isa ORCATwelfth

    @test default_south_rows_to_remove(ORCATwelfth()) == 0
    @test occursin("eORCA12", metadata_path(mesh_meta))
    @test occursin("eORCA12", metadata_path(bathy_meta))
end

@testset "ORCAQuarter Metadatum construction" begin
    bathy_meta = Metadatum(:bottom_height; dataset=ORCAQuarter())
    @test bathy_meta.name == :bottom_height
    @test bathy_meta.dataset isa ORCAQuarter

    mesh_meta = Metadatum(:mesh_mask; dataset=ORCAQuarter())
    @test mesh_meta.name == :mesh_mask
    @test mesh_meta.dataset isa ORCAQuarter

    @test default_south_rows_to_remove(ORCAQuarter()) == 155
    @test occursin("eORCA025", metadata_path(mesh_meta))
    @test occursin("eORCA025", metadata_path(bathy_meta))
end

@testset "ORCAGrid with ORCAOne dataset on $(arch)" for arch in test_architectures
    south_rows_to_remove = 43
    grid = ORCAGrid(arch; dataset=ORCAOne(), Nz=5, z=(-5000, 0), halo=(4, 4, 4), south_rows_to_remove)
    @test grid.underlying_grid.Ny == 332 - south_rows_to_remove

    grid = ORCAGrid(arch; dataset=ORCAOne(), Nz=5, z=(-5000, 0), halo=(4, 4, 4), south_rows_to_remove=0)

    # Default returns ImmersedBoundaryGrid with bathymetry
    @test grid isa ImmersedBoundaryGrid
    underlying = grid.underlying_grid
    @test underlying isa Oceananigans.Grids.OrthogonalSphericalShellGrid
    @test underlying isa TripolarGrid
    @test underlying.Nx == 362
    @test underlying.Ny == 332
    @test underlying.Nz == 5

    # Coordinates span near-global domain
    @test minimum(underlying.λᶜᶜᵃ.parent) < -179
    @test maximum(underlying.λᶜᶜᵃ.parent) > 179
    @test minimum(underlying.φᶜᶜᵃ.parent) < -80
    @test maximum(underlying.φᶜᶜᵃ.parent) > 80
end

@testset "ORCAGrid without bathymetry on $(arch)" for arch in test_architectures
    grid = ORCAGrid(arch; dataset=ORCAOne(), Nz=5, z=(-5000, 0), halo=(4, 4, 4),
                    with_bathymetry=false)

    @test grid isa Oceananigans.Grids.OrthogonalSphericalShellGrid
    @test grid isa TripolarGrid
    @test !(grid isa ImmersedBoundaryGrid)
    @test grid.Nx == 362
    @test grid.Ny == 332 - default_south_rows_to_remove(ORCAOne())
    @test grid.Nz == 5
end

@testset "ORCAGrid with south_rows_to_remove on $(arch)" for arch in test_architectures
    Nremove = 40
    grid = ORCAGrid(arch; dataset=ORCAOne(), Nz=5, z=(-5000, 0), halo=(4, 4, 4),
                    south_rows_to_remove=Nremove)

    @test grid isa ImmersedBoundaryGrid
    underlying = grid.underlying_grid
    @test underlying.Nx == 362
    @test underlying.Ny == 332 - Nremove
    @test underlying.Nz == 5
end

@testset "ORCAGrid metric consistency" begin
    grid = ORCAGrid(CPU(); dataset=ORCAOne(), Nz=5, z=(-5000, 0), halo=(4, 4, 4), with_bathymetry=false)

    Nx, Ny = grid.Nx, grid.Ny
    Hx, Hy = grid.Hx, grid.Hy

    # No NaNs or Infs in any grid data
    for name in (:λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ,
                 :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ,
                 :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ,
                 :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
                 :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ)
        data = getproperty(grid, name)
        @test all(isfinite, Oceananigans.on_architecture(CPU(), data)) == true
    end

    # Metrics strictly positive over the full interior. Face-y fields on
    # RightFaceFolded have Ny+1 interior rows; the fold row Ny+1 must be checked.
    LYs = Dict(:Δxᶜᶜᵃ => Center, :Δxᶠᶜᵃ => Center, :Δxᶜᶠᵃ => Face, :Δxᶠᶠᵃ => Face,
               :Δyᶜᶜᵃ => Center, :Δyᶠᶜᵃ => Center, :Δyᶜᶠᵃ => Face, :Δyᶠᶠᵃ => Face,
               :Azᶜᶜᵃ => Center, :Azᶠᶜᵃ => Center, :Azᶜᶠᵃ => Face, :Azᶠᶠᵃ => Face)
    for (name, LY) in LYs
        data = getproperty(grid, name)
        Njf = Base.length(LY(), Oceananigans.Grids.RightFaceFolded(), Ny)
        interior = Oceananigans.on_architecture(CPU(), data)[1:Nx, 1:Njf]
        @test all(x -> x > 0, interior) == true
    end

    for name in (:Δxᶜᶠᵃ, :Δxᶠᶠᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ)
        data = Oceananigans.on_architecture(CPU(), getproperty(grid, name))
        @test all(x -> x > 0, data[1:Nx, Ny+1])
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
    φC   = grid.φᶜᶜᵃ[imid, 1:Ny]
    nsouth = count(j -> φF[j] < φC[j], 1:Ny)
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

# The eORCA1 mesh_mask ships both the staggered metrics and the T/F coordinates, so reconstructing it
# from `glamt/gphit/glamf/gphif` alone and comparing against the shipped `e1`/`e2` exercises the path
# taken by any mesh that stores coordinates only (eORCA025, eORCA12).
@testset "ORCA mesh reconstruction agrees with the staggered mesh" begin
    Bathymetry = NumericalEarth.Bathymetry
    path = metadata_path(Metadatum(:mesh_mask; dataset=ORCAOne()))

    ds = Dataset(path)
    staggered = Bathymetry.read_orca_staggered_mesh(ds)
    Nx, Ny = size(Bathymetry.read_2d_nemo_variable(ds, "glamt"))
    read_coordinate(name) = Bathymetry.orient_xy(Bathymetry.read_2d_nemo_variable(ds, name), Nx, Ny; name)
    λCC, φCC = read_coordinate("glamt"), read_coordinate("gphit")
    λFF, φFF = read_coordinate("glamf"), read_coordinate("gphif")
    close(ds)

    reconstructed = Bathymetry.reconstruct_orca_mesh_from_CC_FF_points(λCC, φCC, λFF, φFF;
                                                                       radius = Oceananigans.defaults.planet_radius)

    # Both read paths must agree on shape and on NEMO's y-indexing: `halo_filled_data` applies the
    # +1 Face-y shift exactly once, so neither path may pre-shift.
    for name in (:e1t, :e2t, :e1u, :e2u, :e1v, :e2v, :e1f, :e2f, :λFC, :λCF, :λFF, :φFF)
        @test size(getproperty(reconstructed, name)) == size(getproperty(staggered, name))
    end

    # NEMO stores degenerate padding metrics (e1t = 4 m) in the southern rows that `south_rows_to_remove`
    # discards, so compare only the rows the grid actually keeps.
    south = default_south_rows_to_remove(ORCAOne()) + 1
    for name in (:e1t, :e2t, :e1u, :e2u, :e1v, :e2v, :e1f, :e2f)
        reference = getproperty(staggered, name)[:, south:end]
        computed  = getproperty(reconstructed, name)[:, south:end]
        error     = abs.(computed .- reference) ./ max.(abs.(reference), 1e-6)

        # Reconstruction from coordinates is approximate; the tail sits at the tripolar poles where the
        # stored Float32 coordinates stop resolving the sub-metre spacing.
        @test median(error) < 1e-3
        @test count(>(0.01), error) / length(error) < 0.02
    end

    # A metric of exactly zero is never physical: it makes `minimum_xspacing` vanish and the
    # split-explicit substep count diverge. Zeros here mean the periodic overlap columns were wrapped
    # onto their own duplicates.
    for name in (:e1t, :e2t, :e1u, :e2u, :e2v)
        @test count(==(0), getproperty(reconstructed, name)) == 0
    end
end

@testset "ORCAOne bathymetry retrieval" begin
    bathy_md = Metadatum(:bottom_height; dataset=ORCAOne())
    download(bathy_md)
    path = metadata_path(bathy_md)
    @test isfile(path)

    ds = Dataset(path)
    bathy = ds["Bathymetry"][:, :]
    close(ds)

    @test size(bathy) == (362, 332)
    @test maximum(bathy) > 5000  # Deep ocean
end
