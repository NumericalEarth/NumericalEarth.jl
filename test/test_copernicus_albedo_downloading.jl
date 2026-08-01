include("runtests_setup.jl")
include("dataset_status.jl")

using CDSAPI  # loads NumericalEarthCDSAPIExt (the `satellite-albedo` download path)
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, native_grid, metadata_path
using Oceananigans.Grids: topology, Bounded
using Dates: DateTime

# Network-gated: downloads two ~global CGLS files (hundreds of MB). Excluded from
# the default suite in runtests.jl, mirroring the other *_downloading tests.
@testset "Copernicus land albedo download and regional read" begin
    dataset = CopernicusAlbedo()
    region = BoundingBox(longitude = (-114, -109), latitude = (33, 38))
    date = DateTime(2019, 7, 10)

    metadatum = Metadatum(:albedo; dataset, region, date)
    grid = native_grid(metadatum)
    α = @dataset_check "CopernicusAlbedo" "albedo" Field(metadatum)
    values = Array(interior(α))

    # The regional field is exactly the native-grid window (the windowed read fills
    # every cell with di = dj = 0 — no silent off-by-huge-offset misalignment).
    @test size(interior(α))[1:2] == (size(grid, 1), size(grid, 2))

    @test any(isfinite, values)
    valid = filter(!isnan, vec(values))
    @test !isempty(valid)
    @test all(x -> 0 ≤ x ≤ 1, valid)

    # A real window was read, not a constant fill: albedo varies across the box.
    @test length(unique(valid)) > 1

    # Sub-360° window must be Bounded in x so halos do not wrap.
    @test topology(α.grid)[1] == Bounded
end

# The climatology is built by averaging dekadal files, a different download path from the
# single-date product above. One month keeps it to the three dekads of a single year.
@testset "Copernicus land albedo climatology download" begin
    @dataset_check "CopernicusAlbedoClimatology" "albedo" begin
        dataset = CopernicusAlbedoClimatology(years = 2019:2019)
        metadatum = Metadatum(:albedo; dataset, date = DateTime(2018, 7, 1))
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)

        download(metadatum)
        isfile(filepath) || error("CopernicusAlbedoClimatology download produced no file at $(filepath)")
        filepath
    end
end
