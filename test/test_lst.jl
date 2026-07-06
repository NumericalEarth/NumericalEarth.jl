include("runtests_setup.jl")

using NumericalEarth.DataWrangling.LandSurfaceTemperature
using NumericalEarth.DataWrangling.LandSurfaceTemperature: goes_lst, ecostress_lst,
                                                           granule_timestamp,
                                                           lst_masked_residual
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces, z_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, available_variables,
                                    is_three_dimensional, all_dates, missing_value,
                                    default_inpainting

# The real reads (anonymous GOES S3, geostationary→lat/lon reprojection, ECOSTRESS
# HDF5 + Earthdata) are gated behind ArchGDAL / credentials / network, so only the
# pure decode/parse core, the observation-operator kernel, and the dataset
# interface / coverage-validation logic are exercised here.

@testset "GOES-R LST decode" begin
    # K = 0.0025 · DN + 190.0. DN = 24000 → 250 K (in the valid 213–343 K range).
    @test goes_lst(24000) ≈ 250.0
    @test goes_lst(60000) ≈ 0.0025 * 60000 + 190.0  # 340 K, still valid

    # Fill value → NaN.
    @test isnan(goes_lst(65535))

    # Out of valid range → NaN (below 213 K and above 343 K).
    @test isnan(goes_lst(1000))    # 192.5 K < 213
    @test isnan(goes_lst(64000))   # 350 K   > 343

    # DQF masking: DQF = 3 (no retrieval / cloudy) → NaN even for a valid DN.
    @test isnan(goes_lst(24000, 3))
    @test goes_lst(24000, 0) ≈ 250.0
    @test goes_lst(24000, 2) ≈ 250.0
    @test isnan(goes_lst(24000, 255))  # any fill ≥ 3 masks
end

@testset "ECOSTRESS L2G decode" begin
    # float32 Kelvin passes through unchanged when clear.
    @test ecostress_lst(300.0f0, 0) ≈ 300.0f0
    @test ecostress_lst(213.5f0, 0) ≈ 213.5f0

    # Cloud mask (nonzero) → NaN.
    @test isnan(ecostress_lst(300.0f0, 1))
    @test isnan(ecostress_lst(300.0f0, 2))

    # NaN fill passes through as NaN.
    @test isnan(ecostress_lst(NaN32, 0))
end

@testset "Granule timestamp parse (irregular overpass times)" begin
    dt = granule_timestamp("ECOv002_L2G_LSTE_s2021186T031245_x")
    @test dt == DateTime(2021, 7, 5, 3, 12, 45)  # day-of-year 186 in 2021

    dt2 = granule_timestamp("OR_ABI-L2-LSTC_s2020001T120000_e")
    @test dt2 == DateTime(2020, 1, 1, 12, 0, 0)

    @test_throws ArgumentError granule_timestamp("no_timestamp_here")
end

@testset "LST observation operator kernel (cloud mask before residual)" begin
    # Clear, valid: variance-normalized residual (T_model − LST_obs) / LST_err.
    @test lst_masked_residual(300.0, 250.0, 2.0, false) ≈ 25.0

    # Cloudy pixel → masked to 0 *before* the residual, NOT 50/2 = 25.
    @test lst_masked_residual(300.0, 250.0, 2.0, true) == 0.0

    # NaN observation → masked.
    @test lst_masked_residual(300.0, NaN, 2.0, false) == 0.0

    # Non-positive uncertainty → masked.
    @test lst_masked_residual(300.0, 250.0, 0.0, false) == 0.0

    # Construct the scaffold and exercise `show`.
    H = LSTObservationOperator([300.0], [false], [2.0], :skin_temperature)
    @test H.variable_name == :skin_temperature
    @test occursin("H_LST", sprint(show, H))
end

@testset "GOES_LST dataset interface" begin
    ds = GOES_LST()
    @test ds.satellite == :goes16
    @test GOES_LST(satellite = :goes18).satellite == :goes18

    @test longitude_interfaces(ds) == (-180, 180)
    @test latitude_interfaces(ds) == (-90, 90)
    @test available_variables(ds)[:land_surface_temperature] == "LST"

    region = BoundingBox(longitude = (-105, -100), latitude = (35, 40))
    meta = Metadatum(:land_surface_temperature; dataset = ds, region,
                     date = DateTime(2020, 1, 1, 12))
    @test dataset_variable_name(meta) == "LST"
    @test is_three_dimensional(meta) == false
    @test isnan(missing_value(meta))
    @test default_inpainting(meta) === nothing

    fn = metadata_filename(ds, :land_surface_temperature, DateTime(2020, 1, 1, 12), region)
    @test startswith(fn, "GOES_GOES16_LST")
    @test endswith(fn, ".nc")

    # GOES all_dates is a regular hourly range.
    dates = all_dates(ds, :land_surface_temperature)
    @test step(dates) == Hour(1)
end

@testset "ECOSTRESS_L2G dataset interface" begin
    ds = ECOSTRESS_L2G()
    @test ds.version == "002"

    vars = available_variables(ds)
    @test vars[:land_surface_temperature] == "LST"
    @test vars[:lst_uncertainty] == "LST_err"
    @test vars[:cloud_mask] == "cloud"

    region = BoundingBox(longitude = (-105, -100), latitude = (35, 40))
    meta = Metadatum(:land_surface_temperature; dataset = ds, region,
                     date = DateTime(2021, 7, 1, 3, 12, 45))
    @test dataset_variable_name(meta) == "LST"
    @test is_three_dimensional(meta) == false
    @test isnan(missing_value(meta))
    @test default_inpainting(meta) === nothing

    # ECOSTRESS all_dates is IRREGULAR (opportunistic overpasses) — not a range,
    # and the spacing between consecutive dates is unequal.
    dates = all_dates(ds, :land_surface_temperature)
    @test dates isa AbstractVector
    @test length(dates) >= 2
    gaps = diff(dates)
    @test !all(g -> g == gaps[1], gaps)
end

@testset "LST datasets require a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (10, 10, 1),
                                 longitude = (-105, -100),
                                 latitude = (35, 40),
                                 z = (0, 1))

    for ds in (GOES_LST(), ECOSTRESS_L2G())
        meta_global = Metadatum(:land_surface_temperature; dataset = ds)
        @test_throws ErrorException validate_dataset_coverage(grid, meta_global)

        region = BoundingBox(longitude = (-105, -100), latitude = (35, 40))
        meta_region = Metadatum(:land_surface_temperature; dataset = ds, region)
        @test validate_dataset_coverage(grid, meta_region) === nothing

        # Region-keyed filenames are distinct.
        region_b = BoundingBox(longitude = (10, 15), latitude = (45, 50))
        fa = metadata_filename(ds, :land_surface_temperature, nothing, region)
        fb = metadata_filename(ds, :land_surface_temperature, nothing, region_b)
        @test fa != fb
    end
end
