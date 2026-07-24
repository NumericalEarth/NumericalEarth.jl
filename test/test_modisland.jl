using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling.MODISLand
using Oceananigans
using Oceananigans: location, Center
using Test

using Dates: DateTime
import Downloads

const DW = NumericalEarth.DataWrangling
const ML = NumericalEarth.DataWrangling.MODISLand

# The real HDF-EOS read + sinusoidal→lat/lon reprojection require ArchGDAL (with the
# HDF4 driver) and Earthdata credentials, so only the pure decode/mask/blend/
# aggregation logic and the dataset interface are exercised here — on synthetic
# arrays, with no file IO and no credentials.

@testset "MODISLand pure decode / mask / blend" begin
    @testset "Albedo decode (mask fill before scaling)" begin
        @test ML.decode_albedo(0) == 0.0
        @test ML.decode_albedo(500) ≈ 0.5      # 0.001 × 500
        @test ML.decode_albedo(32766) ≈ 32.766 # last valid DN
        @test isnan(ML.decode_albedo(32767))    # fill → NaN, not 32.767
    end

    @testset "Blue-sky blend endpoints" begin
        α_bs = 0.15
        α_ws = 0.22
        @test ML.bluesky_blend(α_bs, α_ws, 0.0) == α_bs   # fully direct
        @test ML.bluesky_blend(α_bs, α_ws, 1.0) == α_ws   # fully diffuse
        @test ML.bluesky_blend(α_bs, α_ws, 0.2) ≈ 0.8 * α_bs + 0.2 * α_ws
    end

    @testset "Albedo mandatory QA" begin
        @test ML.albedo_quality_ok(0x00)            # full BRDF inversion kept
        @test !ML.albedo_quality_ok(0x01)           # magnitude inversion dropped by default
        @test ML.albedo_quality_ok(0x01, 0x01)      # allowed when max = 1
        @test !ML.albedo_quality_ok(0xff)           # fill dropped
    end

    @testset "LAI decode (mask DN>100 before scaling)" begin
        @test ML.decode_lai(50) ≈ 5.0     # 0.1 × 50
        @test ML.decode_lai(100) ≈ 10.0   # last valid DN
        @test isnan(ML.decode_lai(101))   # first non-veg code → NaN
        @test isnan(ML.decode_lai(255))   # fill → NaN, not 25.5
        @test ML.decode_fpar(50) ≈ 0.5    # 0.01 × 50
        @test isnan(ML.decode_fpar(255))
    end

    @testset "LAI QA (MODLAND_QC==0 & SCF_QC∈{0,1})" begin
        @test ML.lai_quality_ok(0x00)          # MODLAND 0, SCF 0
        @test ML.lai_quality_ok(0x20)          # SCF_QC = 1 (bits 5–7)
        @test !ML.lai_quality_ok(0x40)         # SCF_QC = 2 (backup algorithm)
        @test !ML.lai_quality_ok(0x01)         # MODLAND_QC = 1
    end
end

@testset "MODISLand categorical aggregation" begin
    @testset "Fill masking" begin
        @test ML.mask_landcover(5) == 5.0
        @test isnan(ML.mask_landcover(255))
    end

    @testset "Mode preserves valid codes" begin
        codes = [1, 1, 2, 255, 1, 3]
        @test ML.mode_aggregate(codes) == 1        # majority, one of the inputs
        @test ML.mode_aggregate(codes) in codes    # never an invented intermediate
        @test ML.mode_aggregate([255, 255]) == 255 # all fill → fill
    end

    @testset "Class fractions sum to 1" begin
        codes = [1, 1, 2, 255, 3]  # 4 valid pixels (255 is fill)
        @test ML.class_fraction(codes, 1) ≈ 0.5
        @test ML.class_fraction(codes, 2) ≈ 0.25
        @test ML.class_fraction(codes, 3) ≈ 0.25
        total = sum(ML.class_fraction(codes, c) for c in unique(codes) if c != 255)
        @test total ≈ 1.0
        @test ML.class_fraction([255, 255], 1) == 0.0
    end
end

@testset "MODIS sinusoidal grid geometry" begin
    # Round-trip lon/lat → SIN → lon/lat should be the identity.
    for (longitude, latitude) in ((0.0, 0.0), (114.0, 3.0), (-75.0, 45.0))
        x, y = ML.longitude_latitude_to_sinusoidal(longitude, latitude)
        λ, φ = ML.sinusoidal_to_longitude_latitude(x, y)
        @test λ ≈ longitude atol = 1e-6
        @test φ ≈ latitude atol = 1e-6
    end

    # Tile (0, 0) upper-left corner is the global upper-left.
    x_min, y_max, x_max, y_min = ML.sinusoidal_tile_bounds(0, 0)
    @test x_min ≈ ML.MODIS_GLOBAL_UPPER_LEFT_X
    @test y_max ≈ ML.MODIS_GLOBAL_UPPER_LEFT_Y
    @test x_max - x_min ≈ ML.MODIS_TILE_SIDE
    @test y_max - y_min ≈ ML.MODIS_TILE_SIDE
end

@testset "CMR granule-search URL" begin
    bbox = BoundingBox(longitude = (114, 116), latitude = (2, 4))
    url = ML.cmr_granules_url("MCD43A3", "061", bbox)
    @test occursin("short_name=MCD43A3", url)
    @test occursin("version=061", url)
    @test occursin("bounding_box=114,2,116,4", url)  # W,S,E,N order
    url_temporal = ML.cmr_granules_url("MCD15A3H", "061", bbox; temporal = "2020-06-01,2020-09-01")
    @test occursin("temporal=2020-06-01,2020-09-01", url_temporal)
end

@testset "MODISLand dataset interface" begin
    region = BoundingBox(longitude = (114, 116), latitude = (2, 4))

    @testset "MCD43Albedo" begin
        dataset = MCD43Albedo()
        @test dataset.diffuse_fraction == 0.2
        @test MCD43Albedo(diffuse_fraction = 0.35).diffuse_fraction == 0.35
        @test DW.longitude_interfaces(dataset) == (-180, 180)
        @test DW.latitude_interfaces(dataset) == (-90, 90)
        Nx, Ny, Nz = size(dataset, :albedo)
        @test (Nx, Ny, Nz) == (86400, 43200, 1)

        md = Metadatum(:albedo; dataset, region)
        @test DW.is_three_dimensional(md) == false
        @test DW.default_inpainting(md) === nothing
        @test DW.longitude_name(md) == "lon"
        @test location(md) == (Center, Center, Center)
        @test DW.dataset_variable_name(md) == "Albedo_BSA_shortwave"
    end

    @testset "MCD15 LAI products" begin
        for dataset in (MCD15A3H(), MCD15A2H(), MOD15A2H())
            md = Metadatum(:leaf_area_index; dataset, region)
            @test DW.dataset_variable_name(md) == "Lai_500m"
            @test DW.is_three_dimensional(md) == false
            md_fpar = Metadatum(:fpar; dataset, region)
            @test DW.dataset_variable_name(md_fpar) == "Fpar_500m"
        end
    end

    @testset "MCD12Q1" begin
        @test MCD12Q1().legend == :PFT
        md = Metadatum(:plant_functional_type; dataset = MCD12Q1(), region)
        @test DW.dataset_variable_name(md) == "LC_Type5"
        md_igbp = Metadatum(:landcover_igbp; dataset = MCD12Q1(legend = :IGBP), region)
        @test DW.dataset_variable_name(md_igbp) == "LC_Type1"
    end

    @testset "Region- and date-keyed filenames are distinct" begin
        dataset = MCD43Albedo()
        region_b = BoundingBox(longitude = (0, 2), latitude = (50, 52))
        name_a = metadata_filename(dataset, :albedo, nothing, region)
        name_b = metadata_filename(dataset, :albedo, nothing, region_b)
        @test name_a != name_b
        @test startswith(name_a, "MCD43A3_albedo_")
        @test endswith(name_a, ".nc")
        name_dated = metadata_filename(dataset, :albedo, DateTime(2020, 7, 1), region)
        @test occursin("20200701", name_dated)
        @test name_dated != name_a
    end
end

@testset "MODISLand requires a bounded region" begin
    dataset = MCD43Albedo()
    md_global = Metadatum(:albedo; dataset)
    @test_throws ErrorException DW.validate_dataset_coverage(nothing, md_global)

    region = BoundingBox(longitude = (114, 116), latitude = (2, 4))
    md_region = Metadatum(:albedo; dataset, region)
    @test DW.validate_dataset_coverage(nothing, md_region) === nothing

    # Without the ArchGDAL extension loaded, the read path errors clearly rather
    # than silently producing nothing.
    @test_throws ErrorException Downloads.download(md_region)
end
