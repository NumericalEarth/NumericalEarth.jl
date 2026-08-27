include("runtests_setup.jl")

using ArchGDAL  # loads NumericalEarthArchGDALExt (GDAL's HDF4 driver, the granule warp)

using NumericalEarth.DataWrangling: DataWrangling, BoundingBox, Metadatum, metadata_path,
    native_grid, read_file_coords, region_info, all_dates
using NumericalEarth.DataWrangling.MODISLand: granule_urls, parse_granule_name, regional_lattice,
    stored_granule_layers, landcover_valid_range, MissingGranulesError
using Oceananigans.Grids: topology, Bounded
using NCDatasets: NCDataset
using Dates: DateTime
using Statistics: mean

# Network-gated: fetches three MODIS granules (tens of MB) from NASA Earthdata, so it needs
# EARTHDATA_USERNAME / EARTHDATA_PASSWORD and GDAL's HDF4 driver. Excluded from the default
# suite in runtests.jl, mirroring the other *_downloading tests. Granule discovery is
# anonymous, so the first testset runs without credentials.

# A 0.2° window over the Ozark plateau: closed oak-hickory forest, covered by one sinusoidal
# tile, and 48 × 48 cells of the product's 1/240° lattice.
const modis_region = BoundingBox(longitude = (-92.5, -92.3), latitude = (36.5, 36.7))

# Day-of-year 185, the composite of period 24 — peak growing season over this window.
const modis_composite_date = DateTime(2020, 7, 3)
const modis_landcover_date = DateTime(2019, 1, 1)

# One directory for every testset, so the 2020 composite the climatology needs is the one
# already fetched below, while a fresh run still reads no cache of a previous one.
const modis_download_directory = mktempdir()

@testset "MODIS granule discovery" begin
    metadatum = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region = modis_region,
                          date = modis_composite_date)
    urls = granule_urls(metadatum)
    granules = parse_granule_name.(basename.(urls))

    @test !isempty(urls)
    @test all(granule -> granule.date == modis_composite_date, granules)
    @test allunique(granule.tile for granule in granules)   # one granule per tile

    # 2016-02-18 is one of the record's holes. A climatology skips a date the archive does
    # not carry, so the condition has to be distinguishable rather than a plain error.
    outage = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region = modis_region,
                       date = DateTime(2016, 2, 18))
    @test_throws MissingGranulesError granule_urls(outage)
end

@testset "Downloading a MCD15A2H composite" begin
    dataset = MCD15A2H()
    metadatum = Metadatum(:leaf_area_index; dataset, region = modis_region,
                          date = modis_composite_date, dir = modis_download_directory)
    download(metadatum)
    path = metadata_path(metadatum)
    @test isfile(path)

    # One warp per layer, one file per date and region: all three variables and both quality
    # bytes come out of this single download.
    NCDataset(path) do ds
        @test all(layer -> haskey(ds, layer), stored_granule_layers(dataset))
    end

    # The warped file is exactly the native-grid window, so the shared regrid reads it with
    # no offset — the part of the lattice design a synthetic fixture cannot check.
    grid = native_grid(metadatum)
    lattice = regional_lattice(metadatum)
    λc, φc = read_file_coords(metadatum)
    @test (length(λc), length(φc)) == (size(grid, 1), size(grid, 2))
    @test region_info(modis_region, Field{Center, Center, Nothing}(grid), λc, φc) ==
          DataWrangling.BoundingBoxOffset(0, 0, 0)

    𝒜 = Array(interior(Field(metadatum), :, :, 1))
    @test size(𝒜) == (lattice.Nx, lattice.Ny)

    valid = filter(!isnan, vec(𝒜))
    @test !isempty(valid)
    @test all(𝒜 -> 0 ≤ 𝒜 ≤ 10, valid)          # the product's own ceiling, once scaled
    @test length(unique(valid)) > 1            # a real window, not a constant fill
    @test mean(valid) > 1                      # a closed canopy in July

    # A sub-360° window must be Bounded in x so halos do not wrap.
    @test topology(grid)[1] == Bounded

    # FPAR reads the same file, and takes its own scale factor.
    fpar_metadatum = Metadatum(:fpar; dataset, region = modis_region,
                               date = modis_composite_date, dir = modis_download_directory)
    @test metadata_path(fpar_metadatum) == path
    fpar = filter(!isnan, vec(Array(interior(Field(fpar_metadatum), :, :, 1))))
    @test !isempty(fpar)
    @test all(f -> 0 ≤ f ≤ 1, fpar)
end

@testset "Warping a second region from cached granules" begin
    granule_cache = joinpath(modis_download_directory, "granules")
    granules = sort!(readdir(granule_cache))
    @test !isempty(granules)
    @test all(endswith(".hdf"), granules)

    stamps = [mtime(joinpath(granule_cache, granule)) for granule in granules]

    # A window inside the one just warped is served by the same granules, so the second
    # warp is local: the cache neither grows nor is rewritten.
    inner = BoundingBox(longitude = (-92.45, -92.35), latitude = (36.55, 36.65))
    metadatum = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region = inner,
                          date = modis_composite_date, dir = modis_download_directory)
    download(metadatum)
    @test isfile(metadata_path(metadatum))

    @test sort!(readdir(granule_cache)) == granules
    @test [mtime(joinpath(granule_cache, granule)) for granule in granules] == stamps
end

@testset "Downloading a MCD12Q1 land-cover map" begin
    dataset = MCD12Q1()
    metadatum = Metadatum(:landcover_class; dataset, region = modis_region,
                          date = modis_landcover_date, dir = modis_download_directory)
    download(metadatum)
    @test isfile(metadata_path(metadatum))

    codes = Array(interior(Field(metadatum), :, :, 1))
    valid = filter(!isnan, vec(codes))
    @test !isempty(valid)

    # Nearest-neighbor throughout, and no scale factor: the codes come off the warp as codes.
    @test all(code -> code == round(code), valid)
    @test all(code -> code in landcover_valid_range(dataset), valid)
    @test length(unique(valid)) > 1
    @test !all(code -> code in igbp_non_vegetated_classes, valid)   # a forested window

    # The class field lands on exactly the cells the leaf-area read lands on, so the two
    # pair with no aggregation in between.
    lai_metadatum = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region = modis_region,
                              date = modis_composite_date, dir = modis_download_directory)
    λc, φc = read_file_coords(metadatum)
    λl, φl = read_file_coords(lai_metadatum)
    @test λc == λl
    @test φc == φl
end

@testset "Compositing downloaded leaf-area retrievals" begin
    # Two years keeps the build to one further granule; the record's span is what a reported
    # climatology would use.
    climatology = MODISLAIClimatology(years = 2019:2020)
    period = period_index(modis_composite_date, climatology)
    stamp = all_dates(climatology, :leaf_area_index)[period]

    metadatum = Metadatum(:leaf_area_index; dataset = climatology, region = modis_region,
                          date = stamp, dir = modis_download_directory)
    download(metadatum)
    @test isfile(metadata_path(metadatum))

    𝒜 = Array(interior(Field(metadatum), :, :, 1))
    retained = Array(interior(Field(retained_retrieval_metadatum(metadatum)), :, :, 1))
    @test size(retained) == size(𝒜)
    @test all(n -> 0 ≤ n ≤ length(climatology.years), vec(retained))

    # The count is what makes a gap honest: a cell no year could observe stays NaN.
    @test all(i -> (retained[i] == 0) == isnan(𝒜[i]), eachindex(𝒜))
    @test all(𝒜 -> 0 ≤ 𝒜 ≤ 10, filter(!isnan, vec(𝒜)))

    # One of the composited years is the single date read above, so compositing can only
    # have filled gaps, never opened them.
    single = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region = modis_region,
                       date = modis_composite_date, dir = modis_download_directory)
    @test count(!isnan, 𝒜) ≥ count(!isnan, Array(interior(Field(single), :, :, 1)))
end
