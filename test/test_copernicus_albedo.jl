include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Metadatum,
    is_three_dimensional, default_inpainting,
    dataset_variable_name, metadata_filename,
    longitude_name, latitude_name, all_dates
using NumericalEarth.DataWrangling.CopernicusLandAlbedo: bluesky_blend, copernicus_albedo_decode,
    copernicus_albedo_dekadal_dates, albedo_satellite,
    albedo_cds_request_variables
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
    @test location(metadatum) == (Center, Center, Center)
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
