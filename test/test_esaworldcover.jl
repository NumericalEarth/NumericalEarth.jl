include("runtests_setup.jl")

using NumericalEarth: ESAWorldCover, WorldCoverVersion, WorldCoverV100, WorldCoverV200
using NumericalEarth.DataWrangling.WorldCover: class_counts, majority_class,
                                               class_fractions, vegetation_fraction,
                                               aggregate_blockwise, aggregate_landcover,
                                               worldcover_window,
                                               class_fraction_variable_name,
                                               ESA_WORLDCOVER_CLASS_CODES,
                                               ESA_WORLDCOVER_CLASS_NAMES,
                                               ESA_WORLDCOVER_FRACTION_VARIABLE_NAMES,
                                               ESA_WORLDCOVER_VEGETATED_CLASSES,
                                               ESA_WORLDCOVER_NATIVE_STEP,
                                               version_year, version_string
using Oceananigans.Grids: λnodes, φnodes
using NumericalEarth.DataWrangling: longitude_interfaces, latitude_interfaces, native_grid,
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

@testset "class counts over a synthetic patch" begin
    patch = UInt8[10 30 30
                  30 30 40
                  30 80 30]
    counts = class_counts(patch)
    @test keys(counts) == keys(ESA_WORLDCOVER_CLASS_NAMES)
    @test sum(counts) == length(patch)
    @test counts.grassland == 6
    @test counts.tree_cover == 1

    # No-data and any code outside the legend are counted by no class.
    @test sum(class_counts(UInt8[0 0; 0 255])) == 0
end

@testset "majority class of a synthetic patch" begin
    # Majority class (30, grassland) over a 3×3 patch.
    patch = UInt8[10 30 30
                  30 30 40
                  30 80 30]
    @test majority_class(patch) == 30

    # No-data (0) is ignored, so the majority among valid pixels wins.
    with_nodata = UInt8[0 0 0
                        0 40 0
                        0 40 10]
    @test majority_class(with_nodata) == 40

    # A patch that is entirely no-data returns 0 (no valid class).
    @test majority_class(fill(UInt8(0), 4, 4)) == 0

    # Ties break toward the smaller code.
    @test majority_class(UInt8[10 80]) == 10

    # Result is always a valid class code (never an invented intermediate).
    @test majority_class(patch) in ESA_WORLDCOVER_CLASS_CODES
end

@testset "per-class fractions are shares of the block area" begin
    # 2×2 tree, plus one crop and one no-data pixel.
    codes = UInt8[10 10 40
                  10 10 0]  # 5 valid (four 10s, one 40), one no-data
    fr = class_fractions(codes)
    @test fr.tree_cover == 4 / 6
    @test fr.cropland   == 1 / 6
    @test sum(values(fr)) ≈ 5 / 6   # the no-data pixel is the unmapped sixth

    # The fractions are the counts normalized by the block size.
    counts = class_counts(codes)
    @test values(fr) == values(counts) ./ length(codes)

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
    @test majority_class(empty) == 0

    # No-data pixels stay in the denominator as unmapped area.
    codes = UInt8[10 0 0
                  0 0 0]  # 1 tree pixel out of 6
    @test class_fractions(codes).tree_cover == 1 / 6
end

@testset "vegetation fraction" begin
    # tree(10)+crop(40) vegetated; water(80)+built-up(50) not; one no-data.
    codes = UInt8[10 40 80
                  50 0 10]  # 3 vegetated of 6 pixels, one of them no-data
    @test vegetation_fraction(codes) == 3 / 6

    # Overriding the vegetated-class set is a modeling choice.
    @test vegetation_fraction(codes; vegetated_classes = (80,)) == 1 / 6

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
    coarse = aggregate_blockwise(codes, 2, majority_class)
    @test size(coarse) == (2, 2)
    @test coarse == [10 20; 30 40]

    # Per-class fraction over blocks: block (1,1) is all tree cover.
    tree = aggregate_blockwise(codes, 2, block -> class_fractions(block).tree_cover)
    @test tree == [1.0 0.0; 0.0 0.0]

    # Non-divisible sizes are rejected (no partial blocks / misalignment).
    @test_throws ArgumentError aggregate_blockwise(codes, 3, majority_class)
end

@testset "ESA WorldCover dataset interface" begin
    dataset = ESAWorldCover()
    @test dataset.version == WorldCoverV200
    @test dataset.aggregation_factor == 12
    @test ESAWorldCover(version = WorldCoverV100).version == WorldCoverV100
    @test ESAWorldCover(aggregation_factor = 120).aggregation_factor == 120

    # Only a published release is representable, and a degenerate factor is
    # rejected at construction rather than at download time.
    @test_throws MethodError ESAWorldCover(version = :v300)
    @test_throws ArgumentError ESAWorldCover(aggregation_factor = 0)

    # Every published release carries its own year and cache token, so a new
    # release can't silently inherit another's S3 key.
    releases = instances(WorldCoverVersion)
    @test length(releases) == 2
    @test allunique(version_year(ESAWorldCover(; version)) for version in releases)
    @test allunique(version_string(ESAWorldCover(; version)) for version in releases)

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
    @test startswith(filename, "ESA_WorldCover_v200_f12_")
    @test endswith(filename, ".nc")
    # One materialized file holds every band, so the filename is variable-independent.
    @test metadata_filename(dataset, :vegetation_fraction, nothing, region) ==
          metadata_filename(dataset, :landcover_class, nothing, region)
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

@testset "aggregate_landcover matches the per-block helpers" begin
    # 4×4 raster with a fully-no-data block (top-left), factor 2 → 2×2 coarse.
    codes = UInt8[ 0  0 40 80
                   0  0 40 80
                  30 30 95 10
                  30 90 95 95]
    factor = 2
    aggregated = aggregate_landcover(codes, factor)

    @test aggregated.landcover_class == aggregate_blockwise(codes, factor, majority_class)
    @test aggregated.vegetation_fraction ==
          aggregate_blockwise(codes, factor, vegetation_fraction)
    for name in keys(ESA_WORLDCOVER_CLASS_NAMES)
        @test aggregated.class_fractions[name] ==
              aggregate_blockwise(codes, factor, block -> class_fractions(block)[name])
    end

    # Per-class fractions sum to 1 over blocks with valid pixels, 0 over the no-data block.
    total = sum(values(aggregated.class_fractions))
    @test all(f -> f ≈ 1 || f == 0, total)
    @test any(iszero, total)  # the all-no-data block
end

@testset "materialized window is a superset of and cell-aligned with the native grid" begin
    dataset = ESAWorldCover()
    factor = dataset.aggregation_factor
    Δ = factor * ESA_WORLDCOVER_NATIVE_STEP
    ε = Δ / 100

    # Edges exactly on cell faces (5.45, 4.8, …) are the case that used to shift the
    # field by one cell; arbitrary and western-hemisphere edges are covered too.
    regions = (BoundingBox(longitude = (5.45, 5.95),     latitude = (52.05, 52.45)),
               BoundingBox(longitude = (4.8, 5.0),       latitude = (52.3, 52.5)),
               BoundingBox(longitude = (5.4507, 5.933),  latitude = (52.018, 52.474)),
               BoundingBox(longitude = (-3.42, -3.05),   latitude = (55.88, 56.13)))

    for region in regions
        grid = native_grid(Metadatum(:vegetation_fraction; dataset, region))
        λc = Array(λnodes(grid, Center()))
        φc = Array(φnodes(grid, Center()))

        i₁, i₂, j₁, j₂ = worldcover_window(region.longitude, region.latitude, factor)
        west  = i₁ * ESA_WORLDCOVER_NATIVE_STEP
        south = j₁ * ESA_WORLDCOVER_NATIVE_STEP
        nx = (i₂ - i₁) ÷ factor
        ny = (j₂ - j₁) ÷ factor
        file_λ = collect(range(west  + Δ / 2, step = Δ, length = nx))
        file_φ = collect(range(south + Δ / 2, step = Δ, length = ny))

        # The window covers every native-grid center...
        @test file_λ[1] ≤ minimum(λc) + ε && maximum(λc) ≤ file_λ[end] + ε
        @test file_φ[1] ≤ minimum(φc) + ε && maximum(φc) ≤ file_φ[end] + ε
        # ...and each native center coincides with a file center, so the offset-based
        # read-back places data with no whole-cell or sub-cell registration shift.
        @test all(λ -> any(fλ -> abs(fλ - λ) < ε, file_λ), λc)
        @test all(φ -> any(fφ -> abs(fφ - φ) < ε, file_φ), φc)
    end
end
