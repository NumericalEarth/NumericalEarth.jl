include("runtests_setup.jl")

using NumericalEarth.DataWrangling.ASTERGED
using NumericalEarth.DataWrangling.ASTERGED: decode_mean, decode_sdev,
    broadband_emissivity, broadband_emissivity_map, broadband_uncertainty_map,
    mask_water, OGAWA_2003_BROADBAND_COEFFICIENTS, ASTERGED_WATER_CODE
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces,
    dataset_variable_name, validate_dataset_coverage, metadata_filename,
    is_three_dimensional, default_inpainting, missing_value

# The real HDF5 tile read + Earthdata download live in the ArchGDAL extension and
# need network + NASA Earthdata credentials, so only the pure decode/broadband/
# mask core and the dataset-interface logic are exercised here (verified live
# end-to-end separately; see future_plans/status/aster-ged_STATUS.md).

@testset "ASTER GED decode scaling" begin
    # Mean scale is 0.001; fill −9999 → NaN (not −9.999).
    @test decode_mean(950) ≈ 0.95
    @test decode_mean(982) ≈ 0.982
    @test isnan(decode_mean(-9999))
    @test decode_mean(-9999) != -9.999

    # SDev scale is 0.0001 — 10× smaller than Mean. Decoding with the Mean scale
    # would be a silent 10× error.
    @test decode_sdev(120) ≈ 0.012
    @test decode_sdev(120) != decode_mean(120)
    @test decode_mean(120) ≈ 10 * decode_sdev(120)
    @test isnan(decode_sdev(-9999))
end

@testset "ASTER GED broadband synthesis" begin
    coefficients = OGAWA_2003_BROADBAND_COEFFICIENTS

    # Coefficients form a convex combination (sum to unity, all nonnegative).
    @test sum(coefficients) ≈ 1
    @test all(coefficients .>= 0)
    @test length(coefficients) == 5

    # A convex combination of equal band emissivities returns that value.
    @test broadband_emissivity(fill(0.96, 5), coefficients) ≈ 0.96

    # Broadband stays within the range of the band emissivities → in [0.7, 1.0]
    # when all bands are typical land values.
    ε_bands = [0.92, 0.88, 0.90, 0.97, 0.98]
    ε_bb = broadband_emissivity(ε_bands, coefficients)
    @test minimum(ε_bands) <= ε_bb <= maximum(ε_bands)
    @test 0.7 <= ε_bb <= 1.0
end

@testset "ASTER GED band index is first dimension" begin
    coefficients = OGAWA_2003_BROADBAND_COEFFICIENTS

    # Synthetic tile: (5 bands, Nx, Ny). Bands differ so a transposed reduction
    # would give a different (wrong) answer.
    Nx, Ny = 4, 3
    decoded = Array{Float64}(undef, 5, Nx, Ny)
    for b in 1:5, i in 1:Nx, j in 1:Ny
        decoded[b, i, j] = 0.90 + 0.01 * b
    end

    ε_map = broadband_emissivity_map(decoded, coefficients)
    @test size(ε_map) == (Nx, Ny)

    expected = broadband_emissivity([0.91, 0.92, 0.93, 0.94, 0.95], coefficients)
    @test all(ε_map .≈ expected)
    @test all(0.7 .<= ε_map .<= 1.0)
end

@testset "ASTER GED fill values propagate to NaN" begin
    coefficients = OGAWA_2003_BROADBAND_COEFFICIENTS

    # A fill (−9999) in any band decodes to NaN and propagates to the broadband.
    raw = fill(Int16(950), 5, 2, 2)
    raw[3, 1, 1] = Int16(-9999)
    ε_map = broadband_emissivity_map(decode_mean.(raw), coefficients)
    @test isnan(ε_map[1, 1])
    @test all(isfinite, ε_map[2:end, :][:])
end

@testset "ASTER GED water masking" begin
    ε_map = fill(0.95, 3, 3)

    # LWmap with a water pixel (verified coding: land == 0, water == 1, so the
    # default ASTERGED_WATER_CODE == 1).
    lwmap = fill(0, 3, 3)
    lwmap[2, 2] = ASTERGED_WATER_CODE
    masked = mask_water(ε_map, lwmap)
    @test isnan(masked[2, 2])
    @test masked[1, 1] == 0.95
    @test count(isnan, masked) == 1

    # Verify the keyword overrides the default coding.
    lwmap2 = fill(1, 3, 3)
    lwmap2[1, 3] = 2
    masked2 = mask_water(ε_map, lwmap2; water_code = 2)
    @test isnan(masked2[1, 3])
    @test count(isnan, masked2) == 1
end

@testset "ASTER GED uncertainty broadband" begin
    coefficients = OGAWA_2003_BROADBAND_COEFFICIENTS

    # SDev raw of 120 → 0.012 per band. Broadband uncertainty ≤ max band σ.
    raw_sdev = fill(Int16(120), 5, 2, 2)
    σ_map = broadband_uncertainty_map(decode_sdev.(raw_sdev), coefficients)
    @test size(σ_map) == (2, 2)
    @test all(0 .< σ_map .<= 0.012 + eps())
end

@testset "ASTER GED dataset interfaces" begin
    for resolution in (:AG100, :AG1km)
        ds = ASTERGEDv3(; resolution)
        @test longitude_interfaces(ds) == (-180, 180)
        @test latitude_interfaces(ds) == (-90, 90)

        Nx, Ny, Nz = size(ds)
        @test Nz == 1
        @test Nx == (resolution === :AG100 ? 360_000 : 36_000)
        @test Ny == (resolution === :AG100 ? 180_000 : 18_000)

        region = BoundingBox(longitude = (110, 112), latitude = (0, 2))
        meta = Metadatum(:emissivity; dataset = ds, region)
        @test dataset_variable_name(meta) == "/Emissivity/Mean"
        @test is_three_dimensional(meta) == false
        @test default_inpainting(meta) === nothing
        @test missing_value(meta) == -9999

        meta_unc = Metadatum(:emissivity_uncertainty; dataset = ds, region)
        @test dataset_variable_name(meta_unc) == "/Emissivity/SDev"

        filename = metadata_filename(ds, :emissivity, nothing, region)
        @test startswith(filename, "ASTERGED_")
        @test occursin(resolution === :AG100 ? "AG100" : "AG1km", filename)
        @test endswith(filename, ".nc")
    end

    @test_throws ArgumentError ASTERGEDv3(resolution = :AG5km)
end

@testset "ASTER GED region-keyed filenames are distinct" begin
    ds = ASTERGEDv3()
    region_a = BoundingBox(longitude = (110, 112), latitude = (0, 2))
    region_b = BoundingBox(longitude = (0, 2), latitude = (50, 52))
    name_a = metadata_filename(ds, :emissivity, nothing, region_a)
    name_b = metadata_filename(ds, :emissivity, nothing, region_b)
    @test name_a != name_b
end

@testset "ASTER GED requires a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (10, 10),
                                 longitude = (110, 112),
                                 latitude = (0, 2),
                                 topology = (Bounded, Bounded, Flat))

    meta_global = Metadatum(:emissivity; dataset = ASTERGEDv3())
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)

    region = BoundingBox(longitude = (110, 112), latitude = (0, 2))
    meta_region = Metadatum(:emissivity; dataset = ASTERGEDv3(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end
