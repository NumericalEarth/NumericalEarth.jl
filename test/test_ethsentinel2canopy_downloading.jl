include("runtests_setup.jl")

using NumericalEarth.DataWrangling: metadata_path, native_grid
using Oceananigans.Grids: λnodes, φnodes

using ArchGDAL   # activates the windowed-COG read path
using NCDatasets: NCDataset

# Network-gated: windows the ETH 10 m Cloud-Optimized GeoTIFF tiles served from the
# public libdrive share (no credentials). Excluded from CI in runtests.jl like the
# other *_downloading tests; run manually.

# A 0.1° window of closed evergreen broadleaf forest north-east of Manaus, central
# Amazon. It falls inside the single 3° tile S03W060, so each read windows one COG.
const canopy_longitude = (-59.6, -59.5)
const canopy_latitude  = (-2.9, -2.8)
const canopy_region = BoundingBox(longitude = canopy_longitude, latitude = canopy_latitude)

const canopy_grid = LatitudeLongitudeGrid(CPU();
                                          size = (20, 20),
                                          longitude = canopy_longitude,
                                          latitude = canopy_latitude,
                                          topology = (Bounded, Bounded, Flat))

const canopy_height = canopy_height_field(canopy_grid, ETHSentinel2CanopyHeight())

@testset "Downloading ETH canopy height onto a model grid" begin
    h = Array(interior(canopy_height, :, :, 1))

    @test size(h) == size(canopy_grid)[1:2]
    @test all(isfinite, h)                 # unbroken forest: the window has no no-data pixels
    @test all(0 .≤ h .≤ 60)                # ETH heights saturate well below 60 m
    @test 15 < sum(h) / length(h) < 40     # closed-canopy Amazon

    uncertainty = canopy_height_field(canopy_grid, ETHSentinel2CanopyHeight();
                                      name = :canopy_height_uncertainty)
    sd = Array(interior(uncertainty, :, :, 1))

    @test all(0 .< sd .< h)
end

@testset "Tall-canopy fraction over closed forest" begin
    fraction = tall_canopy_fraction_field(canopy_grid, ETHSentinel2CanopyHeight())
    f = Array(interior(fraction, :, :, 1))

    # Closed evergreen canopy: every ~1 km lattice cell stands at least 2 m tall.
    @test all(f .≈ 1)
end

@testset "Downloading the ETH regional canopy-height file" begin
    metadatum = Metadatum(:canopy_height; dataset = ETHSentinel2CanopyHeight(), region = canopy_region)
    filepath = metadata_path(metadatum)
    isfile(filepath) && rm(filepath; force = true)
    download(metadatum)
    @test isfile(filepath)

    # The file is written on the grid the read indexes it with: same cell centers, same
    # counts, so nothing is displaced and no trailing cell repeats an edge.
    grid = native_grid(metadatum)
    file_longitude, file_latitude = NCDataset(filepath) do ds
        Array(ds["lon"][:]), Array(ds["lat"][:])
    end

    @test length(file_longitude) == size(grid, 1)
    @test length(file_latitude)  == size(grid, 2)
    @test file_longitude ≈ Array(λnodes(grid, Center()))
    @test file_latitude  ≈ Array(φnodes(grid, Center()))

    # The regional file is materialized at the native 10 m resolution: ~1200 cells
    # across 0.1°, against the 20 cells of the area-averaged read above.
    native = Field(metadatum)
    @test size(native)[1:2] == size(grid)[1:2]
    @test size(native, 1) ≥ 1200
    @test size(native, 2) ≥ 1200

    fine = filter(isfinite, Array(interior(native, :, :, 1)))
    @test length(fine) > 0.9 * (size(native, 1) * size(native, 2))

    # Coarse-graining conserves the mean canopy height of the native pixels.
    coarse = Array(interior(canopy_height, :, :, 1))
    @test sum(fine) / length(fine) ≈ sum(coarse) / length(coarse) rtol = 0.05
end

@testset "ETH canopy height is no-data over open ocean" begin
    ocean_grid = LatitudeLongitudeGrid(CPU();
                                       size = (4, 4),
                                       longitude = (-140.0, -139.9),
                                       latitude = (-20.0, -19.9),
                                       topology = (Bounded, Bounded, Flat))

    ocean_canopy = canopy_height_field(ocean_grid, ETHSentinel2CanopyHeight())

    # Open water carries the no-data byte, which masks to NaN — not a valid height of 0.
    @test all(isnan, Array(interior(ocean_canopy, :, :, 1)))
end

@testset "Aerodynamic roughness from the measured ETH canopy" begin
    leaf_area_index = Field{Center, Center, Nothing}(canopy_grid)
    set!(leaf_area_index, 5)

    ℓᵐ = Field{Center, Center, Nothing}(canopy_grid)
    d  = Field{Center, Center, Nothing}(canopy_grid)
    compute_aerodynamic_roughness!(ℓᵐ, d, DragPartitionRoughness(),
                                   (; leaf_area_index, canopy_height), canopy_grid)

    h = Array(interior(canopy_height, :, :, 1))
    roughness = Array(interior(ℓᵐ, :, :, 1))
    displacement = Array(interior(d, :, :, 1))

    @test all(0 .< roughness .< displacement .< h)

    # A uniform leaf area index makes both lengths a fixed fraction of the canopy height.
    @test all(roughness ./ h .≈ roughness[1] / h[1])
    @test all(displacement ./ h .≈ displacement[1] / h[1])
end
