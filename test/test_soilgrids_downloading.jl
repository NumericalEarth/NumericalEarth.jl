include("runtests_setup.jl")

using ArchGDAL  # activates NumericalEarthArchGDALExt (the Homolosine → EPSG:4326 warp)

using NCDatasets: NCDataset

using NumericalEarth.DataWrangling: BoundingBox, native_grid
using NumericalEarth.DataWrangling.SoilGrids
using NumericalEarth.DataWrangling.SoilGrids: Grid250m

using Oceananigans.Grids: λnodes, φnodes

# ISRIC needs no credentials. Excluded from CI in runtests.jl; run manually.

# Surrey countryside, south of London — real land, away from the Thames/City of London core
# (SoilGrids has no valid prediction over major open water / dense urban impervious cover,
# so a region picked for "is this densely built" tests elsewhere is a poor choice here).
const soilgrids_region = BoundingBox(longitude = (-0.35, -0.25), latitude = (51.20, 51.27))

@testset "Downloading SoilGrids2 from ISRIC directly" begin
    # Grid1000m shares this exact same download/warp/stack code path (soilgrids_raster_geometry
    # → soilgrids_depth_window → gdalwarp), differing only in the target extent that native_grid
    # computes from `region` — global for Grid1000m, this bounded window for Grid250m.
    # The global path is exercised offline (`test_soilgrids.jl`'s "raster geometry matches
    # native_grid" testset checks its Nx/Ny/extent); a *real* Grid1000m download is a global
    # ~19 GB pull per variable+statistic, too expensive to run here routinely.
    dataset = SoilGrids2(resolution = Grid250m)
    metadatum = Metadatum(:clay_fraction; dataset, region = soilgrids_region)

    # Start from a clean file so a stale cache cannot stand in for the download.
    rm(metadata_path(metadatum); force = true)
    download(metadatum)
    @test isfile(metadata_path(metadatum))

    NCDataset(metadata_path(metadatum)) do ds
        @test size(ds["clay"], 3) == 6   # six depth intervals
    end

    field = Field(metadatum, CPU())
    vals = filter(isfinite, Array(interior(field)))
    @test !isempty(vals)

    # Clay mass fraction (kg/kg) must be physically bounded in [0, 1].
    @test all(x -> 0 ≤ x ≤ 1, vals)
    @test length(unique(vals)) > 1   # a real spatial field, not a constant fill
end

@testset "Grid250m lands on the model's own native grid" begin
    dataset = SoilGrids2(resolution = Grid250m)
    metadatum = Metadatum(:clay_fraction; dataset, region = soilgrids_region)
    download(metadatum)

    grid = native_grid(metadatum)
    λ, φ = NCDataset(metadata_path(metadatum)) do ds
        (Array(ds["lon"][:]), Array(ds["lat"][:]))
    end

    λc = collect(λnodes(grid, Center()))
    φc = collect(φnodes(grid, Center()))
    Δλ = λc[2] - λc[1]
    Δφ = φc[2] - φc[1]

    @test length(λ) == length(λc)
    @test length(φ) == length(φc)
    @test maximum(abs.(λ .- λc)) < Δλ / 100
    @test maximum(abs.(φ .- φc)) < Δφ / 100
end

@testset "SoilGrids2 corrected depth-unit grid extent" begin
    dataset = SoilGrids2(resolution = Grid250m)
    metadatum = Metadatum(:clay_fraction; dataset, region = soilgrids_region)
    grid = native_grid(metadatum)

    # The fixed z_interfaces span 2 m, not the pre-fix 200 m.
    @test grid.Lz ≈ 2.0 rtol = 1e-6
end
