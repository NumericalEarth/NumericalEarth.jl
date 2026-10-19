include("runtests_setup.jl")

using Oceananigans.OutputReaders: Cyclical
using NumericalEarth.DataWrangling: DataWrangling, BoundingBox, Metadatum, Metadata, native_grid,
    is_three_dimensional, default_inpainting, dataset_variable_name, metadata_filename,
    longitude_name, latitude_name, all_dates, native_times, available_variables,
    longitude_interfaces, latitude_interfaces, validate_dataset_coverage,
    retrieve_data, read_file_coords, region_info, fill_gaps!, cmr_granules_url,
    window_center, averaging_window, window_bounds, class_fractions, valid_area_mean!
using NumericalEarth.DataWrangling.MODISLand: MODISLand, mask_lai_fill, lai_screening_flags,
    lai_rejection_flags, modis_composite_dates,
    parse_granule_name, select_granules, regional_lattice,
    period_index, reduce_retained, stored_granule_layers,
    MODIS_LAI_SCALE, MODIS_FPAR_SCALE, MODIS_LAI_LANDCOVER_CODES,
    mask_landcover_fill, landcover_valid_range, landcover_layer, composite_window,
    MissingGranulesError, write_lattice_file,
    modis_lai_class_names, modis_plant_functional_type_names,
    retained_retrieval_metadatum, lai_screening_mask, recommended_lai_screening,
    mask_lai_landcover, modis_landcover_class_names, landcover_class_names,
    igbp_class_names, class_maximum_gap, zero_non_vegetated!
using Oceananigans.Grids: λnodes, φnodes, topology, Bounded
using NCDatasets: NCDataset
using Dates: DateTime, Day, Month, dayofyear
using Statistics: mean

const Δ = MODISLand.MODIS_LATTICE_SPACING

# `FparLai_QC`: bit 0 MODLAND_QC, bit 1 SENSOR, bit 2 DEADDETECTOR, bits 3-4 CLOUDSTATE,
# bits 5-7 SCF_QC.
lai_qc(; modland = 0, sensor = 0, dead = 0, cloud = 0, scf = 0) =
    UInt8(modland | (sensor << 1) | (dead << 2) | (cloud << 3) | (scf << 5))

# `FparExtra_QC`: bits 0-1 LANDSEA, bit 2 SNOW_ICE, bit 3 AEROSOL, bit 4 CIRRUS,
# bit 5 INTERNAL_CLOUDMASK, bit 6 CLOUD_SHADOW, bit 7 SCF_BIOME_MASK.
lai_extra_qc(; landsea = 0, snow = 0, aerosol = 0, cirrus = 0, cloudmask = 0, shadow = 0, biome = 0) =
    UInt8(landsea | (snow << 2) | (aerosol << 3) | (cirrus << 4) |
          (cloudmask << 5) | (shadow << 6) | (biome << 7))

# A file with the layout the download step writes: raw `UInt8` digital numbers and quality
# bytes on the regional lattice, latitude ascending, coordinates on cell centers.
function write_synthetic_modis_file(path, lattice; lai_at = Dict(), qc_at = Dict(), extra_at = Dict())
    Nx, Ny = lattice.Nx, lattice.Ny

    # A ramp unique per cell, so a shifted read cannot pass by coincidence.
    lai = UInt8[mod((i - 1) * Ny + (j - 1), 101) for i in 1:Nx, j in 1:Ny]
    qc = fill(lai_qc(), Nx, Ny)
    extra = fill(lai_extra_qc(), Nx, Ny)

    for ((i, j), value) in lai_at
        lai[i, j] = value
    end
    for ((i, j), value) in qc_at
        qc[i, j] = value
    end
    for ((i, j), value) in extra_at
        extra[i, j] = value
    end

    write_lattice_file(path, lattice, ("Lai_500m" => lai, "Fpar_500m" => lai, "LaiStdDev_500m" => lai,
                                       "FparLai_QC" => qc, "FparExtra_QC" => extra))

    return lai, qc, extra
end

# The land-cover companion of the fixture above: one `UInt8` layer per stored granule layer,
# on the same lattice, with a per-cell class ramp through the legend's valid range.
function write_synthetic_landcover_file(path, lattice, dataset; class_at = Dict())
    Nx, Ny = lattice.Nx, lattice.Ny

    valid = landcover_valid_range(dataset)
    classes = UInt8[first(valid) + mod((i - 1) * Ny + (j - 1), length(valid))
                    for i in 1:Nx, j in 1:Ny]

    for ((i, j), value) in class_at
        classes[i, j] = value
    end

    write_lattice_file(path, lattice, (landcover_layer(dataset) => classes,
                                       "QC" => fill(0x00, Nx, Ny), "LW" => fill(0x02, Nx, Ny)))

    return classes
end

@testset "MODIS digital-number decode" begin
    @test mask_lai_fill(0x00) === 0f0
    @test mask_lai_fill(0x64) === 100f0

    # 248 is "no standard deviation available", 249-255 the land-cover special codes.
    for DN in 0xf8:0xff
        @test isnan(mask_lai_fill(DN))
    end
    @test isnan(mask_lai_fill(0x65))   # 101, one past the valid range

    # The same codes the leaf-area decode rejects name the non-vegetated classes, and the two
    # helpers partition the digital-number range between them.
    for (class, code) in pairs(modis_landcover_class_names)
        @test mask_lai_landcover(UInt8(code)) === Float32(code)
        @test isnan(mask_lai_fill(UInt8(code)))
    end
    @test collect(MODIS_LAI_LANDCOVER_CODES) == sort(collect(values(modis_landcover_class_names)))

    # A valid retrieval carries no class, and the fill value names none either.
    for DN in (0x00, 0x32, 0x64, 0xf8, 0xff)
        @test isnan(mask_lai_landcover(DN))
    end
    @test !isnan(mask_lai_fill(0x64))
end

@testset "MODIS quality screening" begin
    @test lai_screening_mask() == 0x0000
    @test lai_screening_mask(:cloudy) == lai_screening_flags.cloudy
    @test lai_screening_mask(:cloudy, :cloudy) == lai_screening_flags.cloudy   # idempotent
    @test lai_screening_mask(:other_quality, :snow_or_ice) ==
          lai_screening_flags.other_quality | lai_screening_flags.snow_or_ice
    @test_throws ArgumentError lai_screening_mask(:not_a_criterion)

    # Every criterion is a distinct single bit of the screening word.
    masks = collect(values(lai_screening_flags))
    @test all(mask -> count_ones(mask) == 1, masks)
    @test length(unique(masks)) == length(masks)

    # The recommended screen is exactly these three criteria and nothing else.
    @test recommended_lai_screening() ==
          lai_screening_mask(:other_quality, :backup_algorithm, :cloudy)
    for criterion in (:dead_detector, :snow_or_ice, :high_aerosol, :cirrus,
                      :internal_cloud, :cloud_shadow)
        @test recommended_lai_screening() & lai_screening_flags[criterion] == 0
    end

    # A best-quality, clear, main-algorithm pixel over land fails nothing.
    @test lai_rejection_flags(lai_qc(), lai_extra_qc()) == 0x0000

    # MODLAND_QC.
    @test lai_rejection_flags(lai_qc(modland = 1), lai_extra_qc()) == lai_screening_flags.other_quality

    # SCF_QC: 0 and 1 are the main radiative-transfer retrievals, 2-4 are not.
    for scf in 0:1
        @test lai_rejection_flags(lai_qc(; scf), lai_extra_qc()) & lai_screening_flags.backup_algorithm == 0
    end
    for scf in 2:4
        @test lai_rejection_flags(lai_qc(; scf), lai_extra_qc()) & lai_screening_flags.backup_algorithm != 0
    end

    # CLOUDSTATE: 0 is clear and 3 is undefined-assumed-clear, so only 1 and 2 are cloudy.
    for cloud in (0, 3)
        @test lai_rejection_flags(lai_qc(; cloud), lai_extra_qc()) & lai_screening_flags.cloudy == 0
    end
    for cloud in (1, 2)
        @test lai_rejection_flags(lai_qc(; cloud), lai_extra_qc()) & lai_screening_flags.cloudy != 0
    end

    # The sensor bit says which platform retrieved the pixel; it is not a quality criterion.
    @test lai_rejection_flags(lai_qc(sensor = 1), lai_extra_qc()) == 0x0000

    @test lai_rejection_flags(lai_qc(dead = 1), lai_extra_qc()) == lai_screening_flags.dead_detector

    # `FparExtra_QC` criteria, one bit at a time.
    @test lai_rejection_flags(lai_qc(), lai_extra_qc(snow = 1)) == lai_screening_flags.snow_or_ice
    @test lai_rejection_flags(lai_qc(), lai_extra_qc(aerosol = 1)) == lai_screening_flags.high_aerosol
    @test lai_rejection_flags(lai_qc(), lai_extra_qc(cirrus = 1)) == lai_screening_flags.cirrus
    @test lai_rejection_flags(lai_qc(), lai_extra_qc(cloudmask = 1)) == lai_screening_flags.internal_cloud
    @test lai_rejection_flags(lai_qc(), lai_extra_qc(shadow = 1)) == lai_screening_flags.cloud_shadow

    # The land/sea and biome fields are descriptive, not disqualifying.
    for landsea in 0:3
        @test lai_rejection_flags(lai_qc(), lai_extra_qc(; landsea)) == 0x0000
    end
    @test lai_rejection_flags(lai_qc(), lai_extra_qc(biome = 1)) == 0x0000

    # Criteria accumulate rather than mask one another.
    both = lai_rejection_flags(lai_qc(modland = 1, cloud = 1), lai_extra_qc(snow = 1))
    @test both == lai_screening_flags.other_quality | lai_screening_flags.cloudy |
                  lai_screening_flags.snow_or_ice

    # The all-ones byte pair is the uncovered-cell sentinel the warp writes; it must fail
    # the default screen, or a cell outside the granules would read as a measurement.
    @test lai_rejection_flags(0xff, 0xff) & recommended_lai_screening() != 0
end

@testset "MODIS dataset interface" begin
    dataset = MCD15A2H()
    @test dataset.screened_flags == recommended_lai_screening()
    @test MCD15A2H(screened_flags = 0x0000).screened_flags == 0x0000

    # The 1/240° global lattice the granules are reprojected onto.
    Nx, Ny, Nz = size(dataset, :leaf_area_index)
    @test (Nx, Ny, Nz) == (86400, 43200, 1)

    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    metadatum = Metadatum(:leaf_area_index; dataset, region, date = DateTime(2020, 7, 3))
    @test !is_three_dimensional(metadatum)
    @test isnothing(default_inpainting(metadatum))
    @test dataset_variable_name(metadatum) == "Lai_500m"
    @test longitude_name(metadatum) == "lon"
    @test latitude_name(metadatum) == "lat"
    @test location(metadatum) == (Center, Center, Nothing)

    λ₁, λ₂ = longitude_interfaces(metadatum)
    φ₁, φ₂ = latitude_interfaces(metadatum)
    @test (λ₁, λ₂) == (-180, 180)
    @test (φ₁, φ₂) == (-90, 90)
    @test (λ₂ - λ₁) / Nx ≈ Δ
    @test (φ₂ - φ₁) / Ny ≈ Δ

    @test Set(keys(available_variables(dataset))) ==
          Set((:leaf_area_index, :fpar, :leaf_area_index_uncertainty, :landcover_code))
    @test dataset_variable_name(Metadatum(:fpar; dataset, region)) == "Fpar_500m"

    # The land-cover code is read from the leaf-area layer itself, and is not scaled.
    code = Metadatum(:landcover_code; dataset, region, date = DateTime(2020, 7, 3))
    @test dataset_variable_name(code) == "Lai_500m"
    @test isnothing(DataWrangling.conversion_units(code))
    @test !isnothing(DataWrangling.conversion_units(Metadatum(:leaf_area_index; dataset, region)))

    # The climatology adds the retained count, which is stored beside a reduction.
    climatology = MODISLAIClimatology()
    @test climatology.years == 2003:2019
    @test :retained_retrieval_count in keys(available_variables(climatology))

    @test MCD15A2H in supported_datasets()

    # A global read would pull the whole sinusoidal tile set, so a region is required.
    grid = native_grid(metadatum)
    @test isnothing(validate_dataset_coverage(grid, metadatum))
    @test_throws ErrorException validate_dataset_coverage(grid, Metadatum(:leaf_area_index; dataset))
end

@testset "MODIS composite dates" begin
    dataset = MCD15A2H()
    dates = all_dates(dataset, :leaf_area_index)

    @test issorted(dates)
    @test first(dates) == DateTime(2002, 7, 4)     # the combined product's first composite

    # The compositing period restarts at day-of-year 1 every January, so a full year holds
    # exactly 46 periods in leap and common years alike, and the last one is short.
    for year in (2019, 2020)
        year_dates = filter(d -> Dates.year(d) == year, dates)
        @test length(year_dates) == 46
        @test first(year_dates) == DateTime(year, 1, 1)
        @test dayofyear.(year_dates) == collect(1:8:361)
    end

    # Stepping uniformly by 8 days from the first date would drift out of phase; the
    # year-anchored sequence does not.
    @test DateTime(2020, 1, 1) in dates
    @test DateTime(2021, 1, 1) in dates
    @test DateTime(2020, 12, 26) in dates          # leap year: DOY 361
    @test DateTime(2019, 12, 27) in dates          # common year: the same DOY, a day later
    @test !(DateTime(2021, 1, 3) in dates)         # 8 days after 2020's last composite

    @test period_index(DateTime(2020, 1, 1), 8) == 1
    @test period_index(DateTime(2020, 1, 9), 8) == 2
    @test period_index(DateTime(2020, 12, 26), 8) == 46

    # The 46 climatological stamps come from a common year, so every period's day-of-year is
    # the one every year shares.
    climatology_dates = all_dates(MODISLAIClimatology(), :leaf_area_index)
    @test length(climatology_dates) == 46
    @test dayofyear.(climatology_dates) == collect(1:8:361)

    # A (start_date, end_date) window expands to the product's own cadence.
    metadata = Metadata(:leaf_area_index; dataset,
                        region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5)),
                        dates = (DateTime(2020, 6, 20), DateTime(2020, 7, 10)))
    @test metadata.dates == [DateTime(2020, 6, 17), DateTime(2020, 6, 25),
                             DateTime(2020, 7, 3), DateTime(2020, 7, 11)]
    @test length(native_times(metadata)) == length(metadata.dates)

    # The generator itself, independent of the product.
    quarterly = modis_composite_dates(DateTime(2020), DateTime(2021, 12, 31), 90)
    @test dayofyear.(quarterly) == [1, 91, 181, 271, 361, 1, 91, 181, 271, 361]
end

@testset "MODIS filenames" begin
    dataset = MCD15A2H()
    date = DateTime(2020, 7, 3)
    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    other_region = BoundingBox(longitude = (10, 11), latitude = (36.5, 37.5))

    # One warp per date and region serves every variable, so the raw filename is keyed by
    # date and region but not by variable.
    @test metadata_filename(dataset, :leaf_area_index, date, region) ==
          metadata_filename(dataset, :fpar, date, region)
    @test metadata_filename(dataset, :leaf_area_index, date, region) !=
          metadata_filename(dataset, :leaf_area_index, date, other_region)
    @test metadata_filename(dataset, :leaf_area_index, date, region) !=
          metadata_filename(dataset, :leaf_area_index, date + Day(8), region)

    filename = metadata_filename(dataset, :leaf_area_index, date, region)
    @test !occursin("-9", filename)          # usable on every filesystem
    @test endswith(filename, ".nc")
    @test occursin("20200703", filename)

    # A file is keyed by the lattice cells it holds: bounds a hair apart that select different
    # cells name different files, and the same box in either longitude convention one file.
    # (-92.5 + Δ/2 is the cell center at which the window gains its western cell.)
    east_of_center = BoundingBox(longitude = (-92.4979, -91.5), latitude = (36.5, 37.5))
    west_of_center = BoundingBox(longitude = (-92.4980, -91.5), latitude = (36.5, 37.5))
    @test metadata_filename(dataset, :leaf_area_index, date, east_of_center) !=
          metadata_filename(dataset, :leaf_area_index, date, west_of_center)

    wrapped = BoundingBox(longitude = (267.5, 268.5), latitude = (36.5, 37.5))
    @test metadata_filename(dataset, :leaf_area_index, date, wrapped) == filename

    # A climatology reduces one variable at a time, and a different span, screen or period is
    # a different product.
    climatology = MODISLAIClimatology(years = 2003:2019)
    other_years = MODISLAIClimatology(years = 2010:2019)
    unscreened = MODISLAIClimatology(dataset = MCD15A2H(screened_flags = 0x0000))
    stamp = DateTime(2018, 1, 1)
    @test metadata_filename(climatology, :leaf_area_index, stamp, region) !=
          metadata_filename(climatology, :fpar, stamp, region)
    @test metadata_filename(climatology, :leaf_area_index, stamp, region) !=
          metadata_filename(other_years, :leaf_area_index, stamp, region)
    @test metadata_filename(climatology, :leaf_area_index, stamp, region) !=
          metadata_filename(unscreened, :leaf_area_index, stamp, region)
    @test metadata_filename(climatology, :leaf_area_index, stamp, region) !=
          metadata_filename(climatology, :leaf_area_index, stamp + Day(8), region)
    selected_years = MODISLAIClimatology(years = [2003, 2019])
    @test metadata_filename(climatology, :leaf_area_index, stamp, region) !=
          metadata_filename(selected_years, :leaf_area_index, stamp, region)
    @test metadata_filename(climatology, :leaf_area_index, stamp, region) ==
          metadata_filename(MODISLAIClimatology(years = collect(2003:2019)), :leaf_area_index, stamp, region)

    # The retained count shares the file of the variable it counts.
    metadatum = Metadatum(:leaf_area_index; dataset = climatology, region, date = stamp)
    @test retained_retrieval_metadatum(metadatum).filename == metadatum.filename
    @test retained_retrieval_metadatum(metadatum).name == :retained_retrieval_count

    # A cold-cache read of the count builds the variable behind the shared file; a bare
    # count metadatum names no file any build writes, so it says so instead of building one.
    @test MODISLand.climatology_build_name(metadatum) == :leaf_area_index
    @test MODISLand.climatology_build_name(retained_retrieval_metadatum(metadatum)) == :leaf_area_index
    @test_throws ArgumentError MODISLand.climatology_build_name(
        Metadatum(:retained_retrieval_count; dataset = climatology, region, date = stamp))
end

@testset "MODIS granule selection" begin
    granule = parse_granule_name("MCD15A2H.A2020185.h10v05.061.2020340132006")
    @test granule.date == DateTime(2020, 7, 3)     # 2020 is a leap year: DOY 185 is July 3
    @test granule.tile == "h10v05"
    @test granule.production == 2020340132006
    @test parse_granule_name("MCD15A2H.A2019001.h10v05.061.2019020000000").date == DateTime(2019, 1, 1)
    @test_throws ArgumentError parse_granule_name("not-a-granule.hdf")

    host = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MCD15A2H.061/"
    url(name) = host * name * "/" * name * ".hdf"

    wanted = DateTime(2020, 7, 3)
    urls = [url("MCD15A2H.A2020185.h10v05.061.2020340132006"),
            url("MCD15A2H.A2020185.h11v05.061.2020340131841"),
            url("MCD15A2H.A2020177.h10v05.061.2020340131500"),   # the previous composite
            url("MCD15A2H.A2020185.h10v05.061.2019000000000")]   # an older reprocessing

    selected = select_granules(urls, wanted)
    @test length(selected) == 2
    @test all(u -> parse_granule_name(basename(u)).date == wanted, selected)
    @test Set(parse_granule_name(basename(u)).tile for u in selected) == Set(("h10v05", "h11v05"))

    # One granule per tile: the most recently processed wins.
    h10 = only(filter(u -> occursin("h10v05", u), selected))
    @test parse_granule_name(basename(h10)).production == 2020340132006

    # A date with no granule selects nothing rather than falling back to a neighbor.
    @test isempty(select_granules(urls, DateTime(2020, 7, 11)))

    # The record has holes where an instrument outage prevented a composite, so "no granule"
    # is a distinguishable condition: a climatology skips it, a single-date read cannot.
    outage = MissingGranulesError("no granules on 2016-02-18")
    @test sprint(showerror, outage) == "no granules on 2016-02-18"

    # The search URL carries the region and a one-day window around the composite's start,
    # because a bounding-box search returns whichever composites overlap the day.
    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    query = cmr_granules_url("MCD15A2H", "061", region; date = wanted)
    @test occursin("short_name=MCD15A2H", query)
    @test occursin("version=061", query)
    @test occursin("bounding_box=-92.5,36.5,-91.5,37.5", query)
    @test occursin("temporal=2020-07-03T00:00:00Z,2020-07-04T00:00:00Z", query)
end

@testset "MODIS regional lattice" begin
    dataset = MCD15A2H()
    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    metadatum = Metadatum(:leaf_area_index; dataset, region, date = DateTime(2020, 7, 3))

    lattice = regional_lattice(metadatum)
    grid = native_grid(metadatum)

    # The warp target is exactly the native grid, which is what pins the region offset of
    # the shared regrid to zero.
    @test (lattice.Nx, lattice.Ny) == (size(grid, 1), size(grid, 2))
    @test lattice.west ≈ grid.λᶜᵃᵃ[1] - Δ/2 atol = Δ/20
    @test lattice.south ≈ grid.φᵃᶜᵃ[1] - Δ/2 atol = Δ/20
    @test (lattice.east - lattice.west) / lattice.Nx ≈ Δ
    @test (lattice.north - lattice.south) / lattice.Ny ≈ Δ

    # Cell edges sit on the global lattice, so windows of different size share their cells.
    @test (lattice.west + 180) / Δ ≈ round((lattice.west + 180) / Δ)
    @test (lattice.south + 90) / Δ ≈ round((lattice.south + 90) / Δ)

    # A [0, 360] bounding box maps into the product's [-180, 180] convention.
    mapped = regional_lattice(Metadatum(:leaf_area_index; dataset,
                                        region = BoundingBox(longitude = (267.5, 268.5),
                                                             latitude = (36.5, 37.5)),
                                        date = DateTime(2020, 7, 3)))
    @test mapped.west ≈ lattice.west
    @test mapped.Nx == lattice.Nx

    # A window straddling the ±180 seam is rejected rather than silently spanning the globe.
    @test_throws ArgumentError regional_lattice(
        Metadatum(:leaf_area_index; dataset,
                  region = BoundingBox(longitude = (179, 181), latitude = (36.5, 37.5)),
                  date = DateTime(2020, 7, 3)))
    @test_throws ArgumentError regional_lattice(Metadatum(:leaf_area_index; dataset))
end

@testset "MODIS read path" begin
    mktempdir() do dir
        dataset = MCD15A2H()
        date = DateTime(2020, 7, 3)
        region = BoundingBox(longitude = (-92.5, -92.4), latitude = (36.5, 36.6))
        metadatum = Metadatum(:leaf_area_index; dataset, region, date, dir)
        lattice = regional_lattice(metadatum)
        Nx, Ny = lattice.Nx, lattice.Ny

        lai, _, _ = write_synthetic_modis_file(joinpath(dir, metadatum.filename), lattice;
                                              lai_at = Dict((2, 3) => 0xff,     # fill
                                                            (2, 4) => 0xfe,     # water
                                                            (3, 4) => 0xfa,     # urban
                                                            (3, 5) => 0xfd,     # barren or sparse vegetation
                                                            (3, 6) => 0xfc,     # snow or ice
                                                            (3, 7) => 0xfb,     # wetland
                                                            (3, 8) => 0xf9),    # unclassified
                                              qc_at = Dict((3, 4) => lai_qc(cloud = 1),
                                                           (4, 5) => lai_qc(modland = 1),
                                                           (5, 6) => lai_qc(scf = 3),
                                                           (6, 7) => lai_qc(cloud = 1),
                                                           (7, 8) => lai_qc(cloud = 3),
                                                           (8, 9) => lai_qc(dead = 1)),
                                              extra_at = Dict((9, 10) => lai_extra_qc(snow = 1)))

        # The stored file is the native grid, so the read hands `set_region_data!` exactly as
        # many cells as the grid has and the region offset is zero.
        leaf_area_index = retrieve_data(metadatum)
        λc, φc = read_file_coords(metadatum)
        grid = native_grid(metadatum)
        @test size(leaf_area_index) == (Nx, Ny)
        @test (length(λc), length(φc)) == (Nx, Ny)
        @test region_info(region, Field{Center, Center, Nothing}(grid), λc, φc) ==
              NumericalEarth.DataWrangling.BoundingBoxOffset(0, 0)

        node_atol = Δ / 20
        gλ = Array(λnodes(grid, Center(), Center(), Center()))
        gφ = Array(φnodes(grid, Center(), Center(), Center()))
        @test issorted(λc)
        @test issorted(φc)
        @test λc[1] ≈ gλ[1] atol = node_atol
        @test φc[1] ≈ gφ[1] atol = node_atol
        @test λc[end] ≈ gλ[end] atol = node_atol
        @test φc[end] ≈ gφ[end] atol = node_atol

        # `retrieve_data` returns raw digital numbers; the scale is applied on the way onto
        # the grid, so a fill can never be scaled into a plausible value.
        for (i, j) in ((1, 1), (3, 2), (Nx, Ny))
            @test leaf_area_index[i, j] ≈ Float32(lai[i, j])
        end

        # All out-of-range codes are rejected, and the default screen removes exactly the
        # criteria it names.
        @test isnan(leaf_area_index[2, 3])        # 255, fill
        @test isnan(leaf_area_index[2, 4])        # 254, water
        @test all(isnan, leaf_area_index[3, 4:8]) # 249:253 are classes, not measured zeros
        @test isnan(leaf_area_index[4, 5])        # MODLAND_QC other quality
        @test isnan(leaf_area_index[5, 6])        # back-up algorithm
        @test isnan(leaf_area_index[6, 7])        # significant cloud
        @test !isnan(leaf_area_index[7, 8])       # cloud state undefined, assumed clear
        @test !isnan(leaf_area_index[8, 9])       # dead detector is not in the default screen
        @test !isnan(leaf_area_index[9, 10])      # nor is snow
        @test count(isnan, leaf_area_index) == 10

        @test isnan(retrieve_data(Metadatum(:leaf_area_index_uncertainty; dataset, region, date, dir))[3, 4])

        # Screening only ever removes data.
        unscreened = Metadatum(:leaf_area_index; dataset = MCD15A2H(screened_flags = 0x0000),
                               region, date, dir)
        unscreened_leaf_area_index = retrieve_data(unscreened)
        @test count(isnan, unscreened_leaf_area_index) == 7
        @test all(i -> isnan(leaf_area_index[i]) ||
                       leaf_area_index[i] == unscreened_leaf_area_index[i],
                  eachindex(leaf_area_index))

        # A hand-picked mask screens exactly what it names, and nothing else.
        snow_screened = Metadatum(:leaf_area_index;
                                  dataset = MCD15A2H(screened_flags = lai_screening_mask(:snow_or_ice)),
                                  region, date, dir)
        snow_screened_leaf_area_index = retrieve_data(snow_screened)
        @test isnan(snow_screened_leaf_area_index[9, 10])
        @test !isnan(snow_screened_leaf_area_index[4, 5])
        @test count(isnan, snow_screened_leaf_area_index) == 8

        # The land-cover code is the complement of the leaf-area read on the same layer: it
        # carries a class exactly where there is no retrieval, untouched by the screen.
        codes = retrieve_data(Metadatum(:landcover_code; dataset, region, date, dir))
        @test size(codes) == (Nx, Ny)
        @test codes[2, 4] == 254        # water, where the leaf area is a gap
        @test codes[3, 4] == 250        # urban, without a leaf-area retrieval
        @test codes[3, 5] == 253        # barren or sparse vegetation, without a retrieval
        @test isnan(codes[2, 3])        # 255 is fill, not a class
        @test isnan(codes[4, 5])        # screened retrievals are not land cover
        @test isnan(codes[1, 1])        # a valid retrieval carries no class
        @test count(!isnan, codes) == 6

        # On the grid the product's scale factor is applied, and each variable gets its own.
        field = Field(metadatum)
        values = Array(interior(field, :, :, 1))
        @test size(values) == (Nx, Ny)
        @test values[1, 1] ≈ Float32(lai[1, 1]) * MODIS_LAI_SCALE
        @test values[Nx, Ny] ≈ Float32(lai[Nx, Ny]) * MODIS_LAI_SCALE
        @test isnan(values[2, 3])

        fpar = Array(interior(Field(Metadatum(:fpar; dataset, region, date, dir)), :, :, 1))
        @test fpar[1, 1] ≈ Float32(lai[1, 1]) * MODIS_FPAR_SCALE
        @test all(isnan, fpar[3, 4:8])

        # On one model column over the whole window, the area mean of the retrievals under it,
        # whether or not the model grid has a vertical axis.
        row_area = [sind(lattice.south + j * Δ) - sind(lattice.south + (j - 1) * Δ) for j in 1:Ny]
        area = repeat(row_area', Nx)
        valid = isfinite.(values)
        expected = sum(values[valid] .* area[valid]) / sum(area[valid])

        horizontal = (; longitude = (lattice.west, lattice.east), latitude = (lattice.south, lattice.north))
        for model_grid in (LatitudeLongitudeGrid(CPU(), Float32; size = (1, 1), horizontal...,
                                                 topology = (Bounded, Bounded, Flat)),
                           LatitudeLongitudeGrid(CPU(), Float32; size = (1, 1, 2), horizontal...,
                                                 z = (0, 100), topology = (Bounded, Bounded, Bounded)))
            @test only(Array(interior(Field(metadatum, model_grid)))) ≈ expected rtol = 1e-4
        end

        # A sub-360° window must be Bounded in x so halos do not wrap.
        @test topology(grid)[1] == Bounded

        # Every layer the screen and the three variables need is stored in the one file.
        @test Set(stored_granule_layers(dataset)) ==
              Set(("Lai_500m", "Fpar_500m", "LaiStdDev_500m", "FparLai_QC", "FparExtra_QC"))
    end
end

@testset "MODIS land-cover legends and decode" begin
    @test MCD12Q1().legend == :IGBP
    @test MCD12Q1(legend = :PFT).legend == :PFT

    # A misspelled legend names the valid set rather than failing later on a missing layer.
    @test_throws ArgumentError MCD12Q1(legend = :nonsense)
    message = try
        MCD12Q1(legend = :nonsense)
    catch err
        err.msg
    end
    @test all(legend -> occursin(String(legend), message), (:IGBP, :LAI, :PFT))

    # IGBP has no class 0, and the other two legends use it for water — so the range is per
    # legend, not one shared constant.
    @test landcover_valid_range(MCD12Q1()) == 1:17
    @test landcover_valid_range(MCD12Q1(legend = :LAI)) == 0:10
    @test landcover_valid_range(MCD12Q1(legend = :PFT)) == 0:11

    # A valid code round-trips as itself, unscaled, and the fill value does not.
    @test mask_landcover_fill(0x01, 1:17) === 1f0
    @test mask_landcover_fill(0x11, 1:17) === 17f0
    @test isnan(mask_landcover_fill(0x00, 1:17))
    @test isnan(mask_landcover_fill(0xff, 1:17))
    @test isnan(mask_landcover_fill(0x12, 1:17))       # 18, one past the range
    @test mask_landcover_fill(0x00, 0:10) === 0f0      # water, under the leaf-area legend

    # Each legend's table covers exactly its own valid range.
    @test landcover_class_names(MCD12Q1()) === igbp_class_names
    @test landcover_class_names(MCD12Q1(legend = :LAI)) === modis_lai_class_names
    @test landcover_class_names(MCD12Q1(legend = :PFT)) === modis_plant_functional_type_names
    for legend in (:IGBP, :LAI, :PFT)
        dataset = MCD12Q1(; legend)
        @test sort(collect(values(landcover_class_names(dataset)))) ==
              collect(landcover_valid_range(dataset))
    end
end

@testset "MODIS land-cover dataset interface" begin
    dataset = MCD12Q1()
    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    metadatum = Metadatum(:landcover_class; dataset, region, date = DateTime(2015))

    @test Set(keys(available_variables(dataset))) ==
          Set((:landcover_class, :quality_flag, :land_water_mask))
    @test dataset_variable_name(metadatum) == "LC_Type1"
    @test dataset_variable_name(Metadatum(:landcover_class; dataset = MCD12Q1(legend = :PFT),
                                          region, date = DateTime(2015))) == "LC_Type5"

    # The land-cover product shares the leaf-area product's lattice exactly, which is what
    # lets a class field and a leaf-area series be paired cell for cell with no aggregation.
    @test size(dataset, :landcover_class) == size(MCD15A2H(), :leaf_area_index)
    @test longitude_interfaces(metadatum) == (-180, 180)
    @test !is_three_dimensional(metadatum)
    @test isnothing(default_inpainting(metadatum))
    @test location(metadatum) == (Center, Center, Nothing)
    @test MCD12Q1 in supported_datasets()

    # The leaf-area scale factor must not reach a class code: it would turn cropland (12)
    # into 1.2 without erroring, and every downstream comparison would quietly fail.
    @test isnothing(DataWrangling.conversion_units(metadatum))
    @test isnothing(DataWrangling.conversion_units(Metadatum(:quality_flag; dataset, region,
                                                             date = DateTime(2015))))

    @test Set(stored_granule_layers(dataset)) == Set(("LC_Type1", "QC", "LW"))
    @test Set(stored_granule_layers(MCD12Q1(legend = :LAI))) == Set(("LC_Type3", "QC", "LW"))

    # One map per calendar year, stamped 1 January — not 46 composites a year.
    dates = all_dates(dataset, :landcover_class)
    @test issorted(dates)
    @test first(dates) == DateTime(2001, 1, 1)
    @test all(d -> dayofyear(d) == 1, dates)
    @test length(dates) == length(unique(Dates.year.(dates)))

    # A global read would pull every land tile, so a region is required here too.
    @test isnothing(validate_dataset_coverage(native_grid(metadatum), metadatum))
    @test_throws ErrorException validate_dataset_coverage(native_grid(metadatum),
                                                          Metadatum(:landcover_class; dataset))

    # The granule name parses with the same rule as the leaf-area product's.
    granule = parse_granule_name("MCD12Q1.A2015001.h10v05.061.2019192025610")
    @test granule.date == DateTime(2015, 1, 1)
    @test granule.tile == "h10v05"
end

@testset "MODIS land-cover filenames" begin
    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    other_region = BoundingBox(longitude = (10, 11), latitude = (36.5, 37.5))
    date = DateTime(2015)

    # One warp per year and region serves every variable, but a legend is a different layer.
    @test metadata_filename(MCD12Q1(), :landcover_class, date, region) ==
          metadata_filename(MCD12Q1(), :land_water_mask, date, region)
    @test metadata_filename(MCD12Q1(), :landcover_class, date, region) !=
          metadata_filename(MCD12Q1(legend = :PFT), :landcover_class, date, region)
    @test metadata_filename(MCD12Q1(), :landcover_class, date, region) !=
          metadata_filename(MCD12Q1(), :landcover_class, DateTime(2016), region)
    @test metadata_filename(MCD12Q1(), :landcover_class, date, region) !=
          metadata_filename(MCD12Q1(), :landcover_class, date, other_region)

    filename = metadata_filename(MCD12Q1(), :landcover_class, date, region)
    @test occursin("IGBP", filename)
    @test occursin("2015", filename)
    @test !occursin("-9", filename)
    @test endswith(filename, ".nc")
end

@testset "MODIS land-cover read path" begin
    mktempdir() do dir
        region = BoundingBox(longitude = (-92.5, -92.4), latitude = (36.5, 36.6))
        date = DateTime(2015)
        dataset = MCD12Q1()
        metadatum = Metadatum(:landcover_class; dataset, region, date, dir)
        lattice = regional_lattice(metadatum)

        classes = write_synthetic_landcover_file(joinpath(dir, metadatum.filename), lattice,
                                                 dataset;
                                                 class_at = Dict((2, 3) => 0xff,   # fill
                                                                 (2, 4) => 0x00))  # no IGBP 0

        codes = retrieve_data(metadatum)
        @test size(codes) == (lattice.Nx, lattice.Ny)
        @test isnan(codes[2, 3])
        @test isnan(codes[2, 4])
        for (i, j) in ((1, 1), (3, 2), (lattice.Nx, lattice.Ny))
            @test codes[i, j] == Float32(classes[i, j])
        end

        # On the grid the codes stay codes: no scale factor is applied and no fractional
        # class appears.
        field = Array(interior(Field(metadatum), :, :, 1))
        @test field[1, 1] == Float32(classes[1, 1])
        @test all(code -> isnan(code) || code == round(code), field)
        @test all(code -> isnan(code) || code in 1:17, field)

        # The class field lands on exactly the cells the leaf-area read lands on, so the two
        # pair with no aggregation in between.
        lai_metadatum = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region,
                                  date = DateTime(2019, 7, 4), dir)
        write_synthetic_modis_file(joinpath(dir, lai_metadatum.filename),
                                   regional_lattice(lai_metadatum))

        𝒜 = retrieve_data(lai_metadatum)
        λc, φc = read_file_coords(metadatum)
        λl, φl = read_file_coords(lai_metadatum)
        @test size(codes) == size(𝒜)
        @test λc == λl
        @test φc == φl
        @test region_info(region, Field{Center, Center, Nothing}(native_grid(metadatum)),
                          λc, φc) == NumericalEarth.DataWrangling.BoundingBoxOffset(0, 0)

        # `QC` is enumerated, not bit-packed, so it decodes as a class too.
        quality = retrieve_data(Metadatum(:quality_flag; dataset, region, date, dir))
        @test all(==(0f0), quality)
        @test all(==(2f0), retrieve_data(Metadatum(:land_water_mask; dataset, region, date, dir)))
    end
end

# TODO: turn the `@test_broken`s back into `@test`s once the Oceananigans compat admits a release
# with CliMA/Oceananigans.jl#6253. Before it, `regrid!` regrids grids that share a size in one
# dimension only along the other, and throws for grids of equal horizontal size.
@testset "Class fractions conserve horizontal area [$arch]" for arch in test_architectures
    source_grid = LatitudeLongitudeGrid(arch; size = (4, 4),
                                        longitude = (-4, 0), latitude = (50, 70),
                                        topology = (Bounded, Bounded, Flat))
    classes = Field{Center, Center, Nothing}(source_grid)
    landed(grid) = map(fraction -> Array(interior(fraction, :, :, 1)), class_fractions(grid, classes, (1, 2)))
    set!(classes, (longitude, latitude) -> ifelse(longitude < -2, 1, 2))

    for latitude_cells in (2, 4), longitude_shift in (0, 360)
        grid = LatitudeLongitudeGrid(arch; size = (4, latitude_cells),
                                     longitude = [-3.5, -2.5, -1.5, -1, -0.5] .+ longitude_shift,
                                     latitude = (52, 68), topology = (Bounded, Bounded, Flat))
        @test_broken all(landed(grid) .≈ (repeat([1, 0.5, 0, 0], 1, latitude_cells),
                                          repeat([0, 0.5, 1, 1], 1, latitude_cells)))
    end

    set!(classes, (longitude, latitude) -> ifelse(latitude < 60, 1, 2))
    grid = LatitudeLongitudeGrid(arch; size = (2, 4), longitude = (-4, 0),
                                 latitude = [50, 55, 65, 68, 70],
                                 topology = (Bounded, Bounded, Flat))
    southern_fraction = (sind(60) - sind(55)) / (sind(65) - sind(55))
    @test_broken all(landed(grid) .≈ (repeat([1 southern_fraction 0 0], 2, 1),
                                      repeat([0 1-southern_fraction 1 1], 2, 1)))

    @test_broken first(landed(source_grid)) ≈ (Array(interior(classes, :, :, 1)) .== 1)
end

@testset "Valid-area means [$arch]" for arch in test_architectures
    source_grid = LatitudeLongitudeGrid(arch; size = (4, 4),
                                        longitude = (-4, 0), latitude = (50, 70),
                                        topology = (Bounded, Bounded, Flat))
    source = Field{Center, Center, Nothing}(source_grid)
    set!(source, (longitude, latitude) -> ifelse((longitude < -3) & (latitude < 55), NaN, latitude))

    # Rows of the source, by the area of one cell and the latitude it carries.
    row_area = [sind(55) - sind(50), sind(60) - sind(55), sind(65) - sind(60), sind(70) - sind(65)]
    row_latitude = [52.5, 57.5, 62.5, 67.5]
    south = (3 * row_area[1] * row_latitude[1] + 4 * row_area[2] * row_latitude[2]) /
            (3 * row_area[1] + 4 * row_area[2])
    north = (row_area[3] * row_latitude[3] + row_area[4] * row_latitude[4]) / (row_area[3] + row_area[4])

    for longitude_shift in (0, 360)
        grid = LatitudeLongitudeGrid(arch; size = (1, 3), longitude = (-4, 0) .+ longitude_shift,
                                     latitude = [50, 60, 70, 80], topology = (Bounded, Bounded, Flat))
        mean_latitude = Array(interior(valid_area_mean!(Field{Center, Center, Nothing}(grid), source), 1, :, 1))
        @test mean_latitude[1] ≈ south
        @test mean_latitude[2] ≈ north
        @test isnan(mean_latitude[3])
    end
end

@testset "MODIS class fractions on a model grid" begin
    mktempdir() do dir
        region = BoundingBox(longitude = (-92.5, -92.4), latitude = (36.5, 36.6))
        date = DateTime(2015)
        dataset = MCD12Q1()
        metadatum = Metadatum(:landcover_class; dataset, region, date, dir)
        lattice = regional_lattice(metadatum)

        # Longitude stripes, uniform in latitude, splitting each of the two target cells
        # 7:5 so the fractions are exact column counts: grassland then cropland west of
        # -92.45, fill (read as no-data, counted as water) then urban east of it.
        function stripe_class(λ)
            λ < -92.5 + 7Δ  && return 0x0a  # grassland
            λ < -92.45       && return 0x0c  # cropland
            λ < -92.45 + 7Δ && return 0xff  # fill
            return 0x0d                      # urban
        end

        class_at = Dict((i, j) => stripe_class(lattice.west + (i - 1/2) * Δ)
                        for i in 1:lattice.Nx, j in 1:lattice.Ny)
        write_synthetic_landcover_file(joinpath(dir, metadatum.filename), lattice, dataset;
                                       class_at)

        grid = LatitudeLongitudeGrid(CPU(); size = (2, 2),
                                     longitude = (-92.5, -92.4), latitude = (36.5, 36.6),
                                     topology = (Bounded, Bounded, Flat))

        (; fractions, majority_class) = class_fractions(grid, dataset; date, region, dir)
        @test keys(fractions) == keys(igbp_class_names)

        landed(field) = Array(interior(field, :, :, 1))
        @test all(isapprox.(landed(fractions.grassland)[1, :], 7/12, atol = 1e-3))
        @test all(isapprox.(landed(fractions.cropland)[1, :],  5/12, atol = 1e-3))
        @test all(isapprox.(landed(fractions.water)[2, :],     7/12, atol = 1e-3))
        @test all(isapprox.(landed(fractions.urban)[2, :],     5/12, atol = 1e-3))
        @test all(iszero, landed(fractions.mixed_forest))

        total = sum(landed(fraction) for fraction in fractions)
        @test all(isapprox.(total, 1, atol = 1e-3))

        majority = landed(majority_class)
        @test all(majority[1, :] .== igbp_class_names.grassland)
        @test all(majority[2, :] .== igbp_class_names.water)
        @test landed(Field(metadatum, grid)) == majority

        # The other land-cover variables are codes too, and on the native grid they are copied.
        @test all(==(2), landed(Field(Metadatum(:land_water_mask; dataset, region, date, dir), grid)))
        @test isequal(landed(Field(metadatum, native_grid(metadatum))), retrieve_data(metadatum))

        # So are the codes the leaf-area product writes in place of a retrieval: the western
        # cells are 7:5 water to urban, which a bilinear regrid would blend into snow (252),
        # and the eastern ones hold retrievals, so no code covers them.
        lai_metadatum = Metadatum(:landcover_code; dataset = MCD15A2H(), region,
                                  date = DateTime(2019, 7, 4), dir)
        lai_lattice = regional_lattice(lai_metadatum)
        function stripe_code(λ)
            λ < -92.5 + 7Δ && return 0xfe  # water
            λ < -92.45      && return 0xfa  # urban
            return 0x10                     # a retrieval
        end
        lai_at = Dict((i, j) => stripe_code(lai_lattice.west + (i - 1/2) * Δ)
                      for i in 1:lai_lattice.Nx, j in 1:lai_lattice.Ny)
        write_synthetic_modis_file(joinpath(dir, lai_metadatum.filename), lai_lattice; lai_at)

        codes = landed(Field(lai_metadatum, grid))
        @test all(codes[1, :] .== modis_landcover_class_names.water)
        @test all(isnan, codes[2, :])

        for longitude_shift in (0, 360)
            model_grid = LatitudeLongitudeGrid(CPU(); size = (2, 2, 3),
                                               longitude = (-92.5, -92.4) .+ longitude_shift,
                                               latitude = (36.5, 36.6), z = (-3, 0),
                                               topology = (Bounded, Bounded, Bounded))
            @test landed(Field(metadatum, model_grid)) == majority
            @test isequal(landed(Field(lai_metadatum, model_grid)), codes)
            water_mask = Metadatum(:land_water_mask; dataset, region, date, dir)
            quality = Metadatum(:quality_flag; dataset, region, date, dir)
            @test all(==(2), landed(Field(water_mask, model_grid)))
            @test all(iszero, landed(Field(quality, model_grid)))
        end
    end
end

@testset "MODIS class-keyed temporal tolerance" begin
    classes = Float32[igbp_class_names.evergreen_broadleaf_forest,
                      igbp_class_names.deciduous_broadleaf_forest,
                      NaN]
    @test class_maximum_gap(classes) == [6, 1, 3]
    @test class_maximum_gap(classes; default = 2) == [6, 1, 2]

    # Two columns, one gap, two answers.
    𝒜 = fill(NaN32, 2, 1, 12)
    𝒜[1, 1, :] .= 1f0:12f0
    𝒜[2, 1, :] .= 1f0:12f0
    𝒜[:, :, 4:6] .= NaN32
    pair = Float32[igbp_class_names.evergreen_broadleaf_forest;
                   igbp_class_names.deciduous_broadleaf_forest;;]
    @test_logs (:warn,) fill_gaps!(𝒜; max_gap = class_maximum_gap(pair))
    @test 𝒜[1, 1, 4:6] ≈ Float32[4, 5, 6]
    @test all(isnan, 𝒜[2, 1, 4:6])
end

# A stand-in for an end-stamped window, so the MODIS tests stay clear of another product's dataset.
struct EndStampedMetadatum
    dates :: DateTime
end

DataWrangling.averaging_window(metadatum::EndStampedMetadatum) =
    (metadatum.dates - Day(8), metadatum.dates)

@testset "MODIS composite window and stamp offset" begin
    dataset = MCD15A2H()

    # An 8-day composite is stamped at the start of its window, so its value sits four days
    # later. The cadence restarts every January, so the last window of a year is short.
    @test composite_window(dataset, DateTime(2018, 1, 1)) ==
          (DateTime(2018, 1, 1), DateTime(2018, 1, 9))
    @test composite_window(dataset, DateTime(2018, 12, 27)) ==
          (DateTime(2018, 12, 27), DateTime(2019, 1, 1))          # five days, common year
    @test composite_window(dataset, DateTime(2020, 12, 26)) ==
          (DateTime(2020, 12, 26), DateTime(2021, 1, 1))          # six days, leap year

    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    metadatum = Metadatum(:leaf_area_index; dataset, region, date = DateTime(2018, 1, 1))
    @test averaging_window(metadatum) == (DateTime(2018, 1, 1), DateTime(2018, 1, 9))
    @test window_center(metadatum) == DateTime(2018, 1, 5)

    # A class map is not a temporal composite, so its window is a point and its offset zero.
    class_metadatum = Metadatum(:landcover_class; dataset = MCD12Q1(), region,
                                date = DateTime(2015))
    @test averaging_window(class_metadatum) == (DateTime(2015), DateTime(2015))
    @test window_center(class_metadatum) == DateTime(2015)

    # A stamp that closes its window instead puts the value half a period earlier.
    @test window_center(EndStampedMetadatum(DateTime(2018, 1, 9))) == DateTime(2018, 1, 5)

    # The 46 climatological stamps then span exactly one year rather than 46 × 8 = 368 days,
    # which is what a cyclic series has to wrap on.
    climatology = Metadata(:leaf_area_index; dataset = MODISLAIClimatology(), region)
    times = native_times(climatology)
    @test times[1] == 4 * 86400
    @test times[2] - times[1] == 8 * 86400
    @test times[end] - times[end - 1] == 6.5 * 86400
    @test times[end] - times[1] + (times[end] - times[end - 1]) == 365 * 86400
end

@testset "Averaging bounds from metadata" begin
    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    dates = [DateTime(2018, 12, 11), DateTime(2018, 12, 19), DateTime(2018, 12, 27)]
    metadata = Metadata(:leaf_area_index; dataset = MCD15A2H(), region, dates)

    # The stamps close with the end of the last composite, which is the short five-day period
    # ending a common year rather than another eight days.
    @test window_bounds(metadata) == [dates; DateTime(2019, 1, 1)]

    grid = LatitudeLongitudeGrid(size = (1, 1, 1), longitude = (-92.5, -91.5),
                                 latitude = (36.5, 37.5), z = (0, 1))
    ramp = FieldTimeSeries{Center, Center, Center}(grid, native_times(metadata))
    for n in 1:3
        interior(ramp[n]) .= n
    end

    averaged = time_average(ramp, window_bounds(metadata), Month(1))

    # The last composite is the short five-day period, so it carries five days of weight
    # against the eight of each of its predecessors.
    @test interior(averaged)[1, 1, 1, 1] ≈ (8 * 1 + 8 * 2 + 5 * 3) / 21

    # An instantaneous product has no window with which to close the series.
    class_map = Metadata(:landcover_class; dataset = MCD12Q1(), region,
                         dates = [DateTime(2015)])
    @test_throws ArgumentError window_bounds(class_map)

    # Dates that skip composites do not tile time, so the interior bounds would credit each
    # sample with the skipped period as well.
    every_other = Metadata(:leaf_area_index; dataset = MCD15A2H(), region,
                           dates = [DateTime(2019, 1, 1), DateTime(2019, 1, 17)])
    @test_throws ArgumentError window_bounds(every_other)
end

@testset "Cyclic coverage of a whole climatology" begin
    region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))
    climatology = MODISLAIClimatology()

    # All 46 periods wrap onto themselves — December's neighbor is January — so the cyclic
    # coverage warning stays quiet; a partial slice still warns about its spliced ends.
    whole = Metadata(:leaf_area_index; dataset = climatology, region)
    partial = Metadata(:leaf_area_index; dataset = climatology, region,
                       dates = all_dates(climatology, :leaf_area_index)[1:3])

    @test DataWrangling.spans_whole_cycle(whole)
    @test !DataWrangling.spans_whole_cycle(partial)

    @test_logs DataWrangling.validate_time_coverage(whole, Cyclical())
    @test_logs (:warn, r"holds window means") DataWrangling.validate_time_coverage(partial, Cyclical())
end

@testset "Zeroing the non-vegetated classes" begin
    𝒜 = fill(NaN32, 3, 1, 4)
    𝒜[1, 1, :] .= 1f0:4f0                          # forest, observed
    𝒜[2, 1, 2] = 5                                 # a stray retrieval over water
    𝒜[3, 1, 2] = 7                                 # a retrieval in an urban class

    codes = Float32[igbp_class_names.deciduous_broadleaf_forest;
                    igbp_class_names.water;
                    igbp_class_names.urban;;]

    zero_non_vegetated!(𝒜, codes)

    @test all(iszero, 𝒜[3, 1, :])
    @test all(isnan, 𝒜[2, 1, :])

    # A vegetated column is untouched.
    @test 𝒜[1, 1, :] == Float32[1, 2, 3, 4]

    # Cells with no class at all are left alone — nothing says they carry no canopy.
    unclassified = fill(NaN32, 2, 1, 3)
    zero_non_vegetated!(unclassified, fill(NaN32, 2, 1))
    @test all(isnan, unclassified)

    # Which classes count is a keyword, because a different legend numbers them differently.
    grassland = reshape(Float32[NaN, 3], 2, 1, 1)
    zero_non_vegetated!(grassland, Float32[igbp_class_names.grassland; igbp_class_names.water;;];
                        classes = (igbp_class_names.grassland,), water_classes = ())
    @test iszero(grassland[1, 1, 1])
    @test grassland[2, 1, 1] == 3

    @test_throws ArgumentError zero_non_vegetated!(fill(NaN32, 3, 1, 4), fill(NaN32, 2, 1))
end

@testset "MODIS climatology reduction" begin
    # The NaN-aware reduction, and the count that goes with it.
    @test reduce_retained(mean, Float32[1, 2, 3]) == (2f0, 3)
    @test reduce_retained(mean, Float32[1, NaN, 3]) == (2f0, 2)
    @test reduce_retained(maximum, Float32[1, NaN, 3]) == (3f0, 2)
    @test reduce_retained(mean, Float32[NaN, 5]) == (5f0, 1)

    # A cell no year could observe stays visibly empty rather than reading as zero.
    empty_value, empty_count = reduce_retained(mean, Float32[NaN, NaN])
    @test isnan(empty_value)
    @test empty_count == 0
end

@testset "MODIS climatology build" begin
    mktempdir() do dir
        years = 2017:2019
        climatology = MODISLAIClimatology(; years)
        region = BoundingBox(longitude = (-92.5, -92.4), latitude = (36.5, 36.6))
        period = 27

        # Pre-place the contributing composites so the builder runs without network. Each
        # year screens out a different cell, and 2019 has none at all in one cell — so the
        # reduction, the count, and the empty case are all exercised.
        source_dates = [d for d in all_dates(MCD15A2H(), :leaf_area_index)
                        if Dates.year(d) in years && period_index(d, 8) == period]
        @test length(source_dates) == length(years)

        lattice = nothing
        for (n, date) in enumerate(source_dates)
            metadatum = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date, dir)
            lattice = regional_lattice(metadatum)
            write_synthetic_modis_file(joinpath(dir, metadatum.filename), lattice;
                                       lai_at = Dict((1, 1) => UInt8(10 * n),
                                                     (5, 5) => n == 3 ? 0x28 : 0xfd,
                                                     (6, 6) => 0xfa,
                                                     (7, 7) => 0xfd,
                                                     (8, 8) => 0x00),
                                       qc_at = Dict((3, 3) => lai_qc(cloud = 1),
                                                    (4, 4) => n == 3 ? lai_qc(cloud = 1) : lai_qc()),
                                       extra_at = Dict())
        end
        # One cell is cloudy in every year, so nothing survives there.
        for date in source_dates
            metadatum = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date, dir)
            NCDataset(joinpath(dir, metadatum.filename), "a") do ds
                ds["FparLai_QC"][2, 2] = lai_qc(cloud = 1)
            end
        end

        paths = build_lai_climatology!(climatology; region, periods = period:period, dir)
        @test length(paths) == 1
        @test isfile(only(paths))

        stamp = DateTime(2018, 1, 1) + Day(8 * (period - 1))
        metadatum = Metadatum(:leaf_area_index; dataset = climatology, region, date = stamp, dir)
        @test basename(only(paths)) == metadatum.filename

        𝒜 = Array(interior(Field(metadatum), :, :, 1))
        n = Array(interior(Field(retained_retrieval_metadatum(metadatum)), :, :, 1))

        # The reduction is the mean of the retained digital numbers, scaled on read.
        @test 𝒜[1, 1] ≈ mean((10, 20, 30)) * MODIS_LAI_SCALE
        @test n[1, 1] == 3

        # A cell cloudy in every year has nothing to composite.
        @test isnan(𝒜[2, 2])
        @test n[2, 2] == 0

        # A cell cloudy in one year composites the other two.
        @test n[4, 4] == 2
        @test !isnan(𝒜[4, 4])

        @test 𝒜[5, 5] ≈ 4
        @test n[5, 5] == 1
        @test all(isnan, (𝒜[6, 6], 𝒜[7, 7]))
        @test n[6, 6] == n[7, 7] == 0
        @test 𝒜[8, 8] == 0
        @test n[8, 8] == 3

        selected = Metadatum(:leaf_area_index; dataset = MODISLAIClimatology(years = [2017, 2019]),
                             region, date = stamp, dir)
        selected_count = Field(retained_retrieval_metadatum(selected))
        @test Array(interior(selected_count, :, :, 1))[1, 1] == 2

        # An already-built period is skipped, so an interrupted build resumes.
        modified = mtime(only(paths))
        @test build_lai_climatology!(climatology; region, periods = period:period, dir) == paths
        @test mtime(only(paths)) == modified

        # A peak reducer selects the largest retained value, in its own file beside the mean's.
        peak = Metadatum(:leaf_area_index; dataset = MODISLAIClimatology(; years, reducer = maximum),
                         region, date = stamp, dir)
        @test peak.filename != metadatum.filename
        @test Array(interior(Field(peak), :, :, 1))[1, 1] ≈ 30 * MODIS_LAI_SCALE

        # The count is stored beside a reduction, so it is not something the builder makes.
        @test_throws ArgumentError build_lai_climatology!(climatology;
                                                         name = :retained_retrieval_count,
                                                         region, periods = period:period, dir)
    end
end

@testset "Native-grid FieldTimeSeries reads without interpolating" begin
    # A series built on the dataset's own grid must reproduce the single-date read exactly:
    # interpolating onto an identical grid spreads every gap into its neighbors.
    mktempdir() do dir
        dataset = MCD15A2H()
        region = BoundingBox(longitude = (-92.5, -92.4), latitude = (36.5, 36.6))
        dates = [DateTime(2020, 7, 3), DateTime(2020, 7, 11)]

        for date in dates
            metadatum = Metadatum(:leaf_area_index; dataset, region, date, dir)
            write_synthetic_modis_file(joinpath(dir, metadatum.filename), regional_lattice(metadatum);
                                       lai_at = Dict((5, 5) => 0xff, (20, 20) => 0xfe))
        end

        metadata = Metadata(:leaf_area_index; dataset, region, dates, dir)
        fts = FieldTimeSeries(metadata; time_indices_in_memory = 1)

        for (n, date) in enumerate(dates)
            expected = Array(interior(Field(Metadatum(:leaf_area_index; dataset, region, date, dir)), :, :, 1))
            got = Array(interior(fts[n], :, :, 1))
            @test size(got) == size(expected)
            @test count(isnan, got) == count(isnan, expected)     # the gaps did not spread
            @test count(isnan, got) == 2                          # exactly the two seeded codes
            @test all(i -> isnan(got[i]) || got[i] == expected[i], eachindex(got))
        end

        # The single-date regrid takes the same copy path when its target is the native grid.
        grid = native_grid(metadata, CPU())
        metadatum = Metadatum(:leaf_area_index; dataset, region, date = dates[1], dir)
        regridded = Array(interior(Field(metadatum, grid), :, :, 1))
        @test count(isnan, regridded) == 2
    end
end

@testset "Cyclic gap filling" begin
    # A seasonal series is one period of a periodic signal, so its ends are neighbors.
    # The gap spans indices 5 and 1, so it interpolates from 4 (at index 4) to 2 (at index
    # 2): index 5 sits a third of the way across and index 1 two thirds.
    wrapped = Float32[NaN, 2, 3, 4, NaN]
    fill_gaps!(wrapped; max_gap = 2, cyclic = true)
    @test wrapped ≈ Float32[8/3, 2, 3, 4, 10/3] rtol = 1e-6

    # Without the wrap the same gaps are filled with the nearest valid value, the
    # open-series behavior every existing caller relies on.
    open = Float32[NaN, 2, 3, 4, NaN]
    fill_gaps!(open; max_gap = 2)
    @test open == Float32[2, 2, 3, 4, 4]

    # An edge gap is held to `max_gap` as well, rather than persisting one value indefinitely.
    long_open = Float32[NaN, NaN, NaN, 4, 5]
    @test_logs (:warn,) fill_gaps!(long_open; max_gap = 2)
    @test all(isnan, long_open[1:3])

    # An interior gap is filled identically either way.
    interior_cyclic = Float32[1, NaN, NaN, 4, 5]
    interior_open = copy(interior_cyclic)
    fill_gaps!(interior_cyclic; max_gap = 2, cyclic = true)
    fill_gaps!(interior_open; max_gap = 2)
    @test interior_cyclic == interior_open
    @test interior_cyclic ≈ Float32[1, 2, 3, 4, 5]

    # A gap longer than `max_gap` is left alone, with a warning, so a long outage is never
    # papered over by interpolation.
    long = Float32[NaN, NaN, NaN, 4, 5]
    @test_logs (:warn,) fill_gaps!(long; max_gap = 2, cyclic = true)
    @test all(isnan, long[1:3])

    # An all-NaN column has nothing to interpolate from and is untouched.
    empty_column = fill(NaN32, 4)
    fill_gaps!(empty_column; cyclic = true)
    @test all(isnan, empty_column)

    # A single valid point fills the whole period with itself.
    lone = Float32[NaN, NaN, 7, NaN]
    fill_gaps!(lone; max_gap = 3, cyclic = true)
    @test all(lone .≈ 7)

    # The array method fills each spatial column independently along the last dimension.
    stacked = fill(NaN32, 2, 1, 5)
    stacked[1, 1, :] .= Float32[NaN, 2, 3, 4, NaN]
    stacked[2, 1, :] .= Float32[1, NaN, 3, NaN, 5]
    fill_gaps!(stacked; max_gap = 2, cyclic = true)
    @test stacked[1, 1, 1] ≈ 8/3 rtol = 1e-6
    @test stacked[2, 1, 2] ≈ 2
    @test stacked[2, 1, 4] ≈ 4
end
