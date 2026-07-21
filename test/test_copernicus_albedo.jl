include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Metadatum, native_grid,
    is_three_dimensional, default_inpainting,
    dataset_variable_name, metadata_filename,
    longitude_name, latitude_name, all_dates
using NumericalEarth.DataWrangling.CopernicusLandAlbedo: bluesky_blend, copernicus_albedo_decode,
    copernicus_albedo_dekadal_dates, albedo_satellite,
    albedo_cds_request_variables,
    albedo_read_window
using Oceananigans.Grids: λnodes, φnodes
using Dates: DateTime, Day, day, month, daysinmonth

@testset "Copernicus land albedo helpers" begin
    @test bluesky_blend(0.3, 0.5, 0.0) == 0.3
    @test bluesky_blend(0.3, 0.5, 1.0) == 0.5
    @test bluesky_blend(0.3, 0.5, 0.2) ≈ 0.34

    @test copernicus_albedo_decode(0.37) === 0.37f0
    @test copernicus_albedo_decode(0) === 0.0f0
    @test copernicus_albedo_decode(1) === 1.0f0
    @test isnan(copernicus_albedo_decode(-0.01))
    @test isnan(copernicus_albedo_decode(1.2))
    @test isnan(copernicus_albedo_decode(missing))
    @test isnan(copernicus_albedo_decode(NaN32))
    @test isnan(bluesky_blend(copernicus_albedo_decode(missing), 0.5f0, 0.2f0))
end

@testset "Copernicus land albedo dekadal dates" begin
    dates = copernicus_albedo_dekadal_dates(DateTime(2019, 1, 10), DateTime(2019, 12, 31))
    @test length(dates) == 36
    @test issorted(dates)
    @test all(d -> day(d) in (10, 20, daysinmonth(d)), dates)
    @test DateTime(2019, 2, 28) in dates
    @test DateTime(2019, 7, 31) in dates

    dates = all_dates(CopernicusAlbedo(), :albedo)
    @test first(dates) == DateTime(1998, 4, 10)
    @test last(dates) == DateTime(2020, 6, 30)
    @test issorted(dates)

    # SPOT covers the record until May 2014; PROBA-V takes over from June 2014.
    @test albedo_satellite(DateTime(2005, 7, 10)) == "spot"
    @test albedo_satellite(DateTime(2014, 5, 31)) == "spot"
    @test albedo_satellite(DateTime(2014, 6, 10)) == "proba"
    @test albedo_satellite(DateTime(2019, 7, 10)) == "proba"

    climatology_dates = all_dates(CopernicusAlbedoClimatology(), :albedo)
    @test length(climatology_dates) == 12
    @test month.(climatology_dates) == 1:12
end

@testset "Copernicus land albedo interface" begin
    dataset = CopernicusAlbedo(diffuse_fraction = 0.3)
    @test dataset.diffuse_fraction == 0.3
    @test size(dataset, :albedo) == (40320, 15680, 1)

    date = DateTime(2019, 7, 10)
    metadatum = Metadatum(:albedo; dataset, date)
    @test !is_three_dimensional(metadatum)
    @test isnothing(default_inpainting(metadatum))
    @test dataset_variable_name(metadatum) == "AL_DH_BB"
    @test longitude_name(metadatum) == "lon"
    @test latitude_name(metadatum) == "lat"
    @test location(metadatum) == (Center, Center, Nothing)
    @test albedo_cds_request_variables[:albedo] == ("albb_dh", "albb_bh")

    # Filenames are keyed by date and variable but not by region, so one global
    # download is reused across regions.
    region = BoundingBox(longitude = (-10, 10), latitude = (30, 50))
    @test metadata_filename(dataset, :albedo, date, region) ==
          metadata_filename(dataset, :albedo, date, nothing)
    @test metadata_filename(dataset, :albedo, date, region) !=
          metadata_filename(dataset, :albedo, date + Day(10), region)

    climatology = CopernicusAlbedoClimatology(years = 2018:2019)
    january = metadata_filename(climatology, :albedo, DateTime(2018, 1, 1), nothing)
    july = metadata_filename(climatology, :albedo, DateTime(2018, 7, 1), nothing)
    @test january != july
    @test occursin("2018-2019", january)
    @test occursin("m07", july)
end

@testset "Copernicus land albedo native grid and read window" begin
    dataset = CopernicusAlbedo()
    date = DateTime(2019, 7, 10)
    Nx_full, Ny_full, _ = size(dataset, :albedo)
    @test (Nx_full, Ny_full) == (40320, 15680)  # analytic 1/112° grid unchanged
    Δλ = 360 / Nx_full
    Δφ = 140 / Ny_full  # latitude spans 80 − (−60) = 140°

    # A small mid-latitude box, a box hugging the north edge (80°N), a box hugging the
    # south edge (−60°S), and an antimeridian-crossing box (global fallback).
    windowed = (BoundingBox(longitude = (-114, -109), latitude = (33, 38)),
                BoundingBox(longitude = (10, 14),      latitude = (77, 80)),
                BoundingBox(longitude = (10, 14),      latitude = (-60, -56)))
    fallback = BoundingBox(longitude = (175, 185), latitude = (0, 5))

    for region in windowed
        metadatum = Metadatum(:albedo; dataset, region, date)

        # The native grid is file-free — this must work with nothing downloaded.
        grid = native_grid(metadatum)

        win = albedo_read_window(metadatum)
        @test win !== nothing
        icols, jrows = win

        # The window is EXACTLY the native-grid cell count, so
        # region_info recomputes di = dj = 0 (bit-exact with global-then-slice).
        @test length(icols) == size(grid, 1)
        @test length(jrows) == size(grid, 2)

        # The window's implied cell-center bounds match the native grid's node extent.
        # `atol` sits well below the native cell size (Δ ≈ 0.0089°) so a one-cell
        # misalignment still fails, while Float32 node storage (~1e-5 noise) passes.
        λn = λnodes(grid, Center(), Center(), Center())
        φn = φnodes(grid, Center(), Center(), Center())
        # File coordinates label pixel centers, so the native interfaces sit half a cell
        # out (left = −180 − Δλ/2, bottom = −60 + Δφ/2); centers are interface + (i − ½)Δ.
        λ_win = (-180 - Δλ/2 + (first(icols) - 0.5) * Δλ, -180 - Δλ/2 + (last(icols) - 0.5) * Δλ)
        φ_win = ( -60 + Δφ/2 + (first(jrows) - 0.5) * Δφ,  -60 + Δφ/2 + (last(jrows) - 0.5) * Δφ)
        @test λ_win[1] ≈ minimum(λn)  atol = 1e-3
        @test λ_win[2] ≈ maximum(λn)  atol = 1e-3
        @test φ_win[1] ≈ minimum(φn)  atol = 1e-3
        @test φ_win[2] ≈ maximum(φn)  atol = 1e-3

        # Ascending → file-row (north→south) inversion, as in `retrieve_data`.
        file_rows = (Ny_full - last(jrows) + 1):(Ny_full - first(jrows) + 1)
        @test length(file_rows) == length(jrows)
        @test first(file_rows) ≥ 1 && last(file_rows) ≤ Ny_full
        # Northmost native cell (last(jrows)) sits at the smallest (northernmost) file row.
        @test first(file_rows) == Ny_full - last(jrows) + 1
        @test last(file_rows)  == Ny_full - first(jrows) + 1
    end

    # A window crossing the ±180 seam falls back to the global read path.
    metadatum = Metadatum(:albedo; dataset, region = fallback, date)
    @test native_grid(metadatum) isa Oceananigans.Grids.AbstractGrid  # still constructible
    @test albedo_read_window(metadatum) === nothing

    # No region at all is also the global path.
    @test albedo_read_window(Metadatum(:albedo; dataset, date)) === nothing
end
