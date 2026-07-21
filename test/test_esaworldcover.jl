include("runtests_setup.jl")

using NumericalEarth: ESAWorldCover
using NumericalEarth.DataWrangling.WorldCover: mode_aggregate, class_fraction,
                                               class_fractions, vegetation_fraction,
                                               aggregate_blockwise,
                                               class_fraction_variable_name,
                                               ESA_WORLDCOVER_CLASS_CODES,
                                               ESA_WORLDCOVER_CLASS_NAMES,
                                               ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES,
                                               ESA_WORLDCOVER_VEGETATED_CLASSES,
                                               ESA_WORLDCOVER_MISSING_VALUE
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces,
                                    dataset_variable_name, validate_dataset_coverage,
                                    metadata_filename, is_three_dimensional,
                                    missing_value, available_variables

# The real COG read needs ArchGDAL, the anonymous S3 bucket, and network access,
# so only the dataset-interface logic and the pure categorical-aggregation
# helpers (the physics) are exercised here. The extension read path is verified
# manually / in a network-gated job.

@testset "ESA WorldCover class legend" begin
    # 11 classes with a NON-uniform step near the top (…90, 95, 100).
    @test ESA_WORLDCOVER_CLASS_CODES == (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
    @test length(ESA_WORLDCOVER_CLASS_CODES) == 11
    @test values(ESA_WORLDCOVER_CLASS_NAMES) == ESA_WORLDCOVER_CLASS_CODES
    @test ESA_WORLDCOVER_MISSING_VALUE == 0
    # 0 is not a valid class; the legend starts at 10.
    @test !(0 in ESA_WORLDCOVER_CLASS_CODES)
    # Step from 90 to 95 is 5, not 10 — must not assume a regular stride.
    @test 5 in diff(collect(ESA_WORLDCOVER_CLASS_CODES))
    @test issubset(ESA_WORLDCOVER_VEGETATED_CLASSES, ESA_WORLDCOVER_CLASS_CODES)

    # One per-class fraction variable per class.
    @test length(ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES) == 11
    @test class_fraction_variable_name(:cropland) == :cropland_fraction
    @test :cropland_fraction in ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES
end

@testset "mode aggregation of a synthetic patch" begin
    # Majority class (30, grassland) over a 3×3 patch.
    patch = UInt8[10 30 30
                  30 30 40
                  30 80 30]
    @test mode_aggregate(patch) == 30

    # No-data (0) is ignored, so the majority among valid pixels wins.
    with_nodata = UInt8[0 0 0
                        0 40 0
                        0 40 10]
    @test mode_aggregate(with_nodata) == 40

    # A patch that is entirely no-data returns 0 (no valid class).
    @test mode_aggregate(fill(UInt8(0), 4, 4)) == 0

    # Result is always a valid class code (never an invented intermediate).
    @test mode_aggregate(patch) in ESA_WORLDCOVER_CLASS_CODES
end

@testset "per-class fractions sum to 1 over valid pixels" begin
    # 2×2 tree, plus one crop and one no-data pixel.
    codes = UInt8[10 10 40
                  10 10 0]  # 5 valid (four 10s, one 40), one no-data
    fr = class_fractions(codes)
    @test fr.tree_cover == 4 / 5
    @test fr.cropland   == 1 / 5
    @test sum(values(fr)) ≈ 1.0

    # class_fraction agrees with the NamedTuple entry.
    @test class_fraction(codes, 10) == fr.tree_cover
    @test class_fraction(codes, 40) == fr.cropland

    # A uniform patch: one class is 1, the rest are 0.
    uniform = fill(UInt8(20), 5, 5)
    fru = class_fractions(uniform)
    @test fru.shrubland == 1.0
    @test sum(values(fru)) ≈ 1.0
end

@testset "no-data (0) masking" begin
    # All no-data → every fraction is 0 (they sum to 0, not 1).
    empty = fill(UInt8(0), 3, 3)
    @test all(values(class_fractions(empty)) .== 0)
    @test vegetation_fraction(empty) == 0.0
    @test class_fraction(empty, 10) == 0.0

    # No-data pixels are excluded from the denominator.
    codes = UInt8[10 0 0
                  0 0 0]  # 1 valid tree pixel out of 6
    @test class_fraction(codes, 10) == 1.0
end

@testset "vegetation fraction" begin
    # tree(10)+crop(40) vegetated; water(80)+built-up(50) not; one no-data.
    codes = UInt8[10 40 80
                  50 0 10]  # valid: 10,40,80,50,10 → 3 vegetated of 5
    @test vegetation_fraction(codes) == 3 / 5

    # Overriding the vegetated-class set is a modeling choice.
    @test vegetation_fraction(codes; vegetated_classes = (80,)) == 1 / 5

    # Equals the sum of the vegetated-class fractions.
    fr = class_fractions(codes)
    veg = sum(getproperty(fr, name) for name in keys(ESA_WORLDCOVER_CLASS_NAMES)
              if ESA_WORLDCOVER_CLASS_NAMES[name] in ESA_WORLDCOVER_VEGETATED_CLASSES)
    @test vegetation_fraction(codes) ≈ veg
end

@testset "integer-factor block aggregation keeps alignment" begin
    # 4×4 fine raster, factor 2 → 2×2 coarse. Each 2×2 block is uniform.
    codes = UInt8[10 10 20 20
                  10 10 20 20
                  30 30 40 40
                  30 30 40 40]
    coarse = aggregate_blockwise(codes, 2, mode_aggregate)
    @test size(coarse) == (2, 2)
    @test coarse == [10 20; 30 40]

    # Per-class fraction over blocks: block (1,1) is all tree cover.
    tree = aggregate_blockwise(codes, 2, block -> class_fraction(block, 10))
    @test tree == [1.0 0.0; 0.0 0.0]

    # Non-divisible sizes are rejected (no partial blocks / misalignment).
    @test_throws ArgumentError aggregate_blockwise(codes, 3, mode_aggregate)
end

@testset "ESA WorldCover dataset interface" begin
    dataset = ESAWorldCover()
    @test dataset.version == :v200
    @test dataset.aggregation_factor == 12
    @test ESAWorldCover(version = :v100).version == :v100
    @test ESAWorldCover(aggregation_factor = 120).aggregation_factor == 120

    @test longitude_interfaces(dataset) == (-180, 180)
    @test latitude_interfaces(dataset)  == (-60, 84)

    # Global size at the aggregated (~110 m) resolution, factor 12 over 10 m.
    Nx, Ny, Nz = size(dataset, :vegetation_fraction)
    @test Nz == 1
    @test Nx == 360000   # 360° at 0.001°
    @test Ny == 144000   # 144° at 0.001°

    # A coarser factor shrinks the presented native grid proportionally.
    @test size(ESAWorldCover(aggregation_factor = 120), :vegetation_fraction) == (36000, 14400, 1)

    variables = available_variables(dataset)
    @test Set(keys(variables)) ==
        Set((:landcover_class, :vegetation_fraction, ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES...))

    region = BoundingBox(longitude = (4, 7), latitude = (50, 53))
    for name in (:landcover_class, :vegetation_fraction, :cropland_fraction, :built_up_fraction)
        meta = Metadatum(name; dataset, region)
        @test dataset_variable_name(meta) == "Map"
        @test is_three_dimensional(meta) == false
    end

    # `0` is the no-data sentinel for the categorical class product, but a real
    # value for the derived fractions (a water cell has 0 vegetation fraction),
    # so the fractions carry NaN — which masks nothing — as their missing value.
    @test missing_value(Metadatum(:landcover_class; dataset, region)) == 0
    @test isnan(missing_value(Metadatum(:vegetation_fraction; dataset, region)))
    @test isnan(missing_value(Metadatum(:cropland_fraction; dataset, region)))

    filename = metadata_filename(dataset, :vegetation_fraction, nothing, region)
    @test startswith(filename, "ESA_WorldCover_v200_f12_vegetation_fraction_")
    @test endswith(filename, ".nc")
end

@testset "ESA WorldCover requires a bounded region" begin
    grid = LatitudeLongitudeGrid(CPU();
                                 size = (10, 10),
                                 longitude = (4, 7),
                                 latitude = (50, 53),
                                 topology = (Bounded, Bounded, Flat))

    meta_global = Metadatum(:vegetation_fraction; dataset = ESAWorldCover())
    @test_throws ErrorException validate_dataset_coverage(grid, meta_global)

    region = BoundingBox(longitude = (4, 7), latitude = (50, 53))
    meta_region = Metadatum(:vegetation_fraction; dataset = ESAWorldCover(), region)
    @test validate_dataset_coverage(grid, meta_region) === nothing
end

@testset "Region- and factor-keyed filenames are distinct" begin
    dataset = ESAWorldCover()
    region_a = BoundingBox(longitude = (4, 7), latitude = (50, 53))
    region_b = BoundingBox(longitude = (0, 3), latitude = (40, 43))
    @test metadata_filename(dataset, :vegetation_fraction, nothing, region_a) !=
          metadata_filename(dataset, :vegetation_fraction, nothing, region_b)
    # The aggregation factor is encoded, so caches at different resolutions don't collide.
    @test metadata_filename(dataset, :vegetation_fraction, nothing, region_a) !=
          metadata_filename(ESAWorldCover(aggregation_factor = 120), :vegetation_fraction, nothing, region_a)
end
