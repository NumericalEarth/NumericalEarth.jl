include("runtests_setup.jl")
include("download_utils.jl")

using CopernicusMarine
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, metadata_path
using Oceananigans.Fields: interior
using Oceananigans.Grids: Bounded, Flat, LatitudeLongitudeGrid, topology
using Oceananigans.OutputReaders: time_indices

const AVISO_TEST_DIR = mktempdir()

@testset "AVISO CopernicusMarine fetch padding" begin
    CMExt = Base.get_extension(NumericalEarth, :NumericalEarthCopernicusMarineExt)
    bbox = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = AVISOMonthly()

    lon = CMExt.longitude_bounds_kw(bbox, dataset)
    lat = CMExt.latitude_bounds_kw(bbox, dataset)

    @test lon.minimum_longitude ≈ 200 - 1/4
    @test lon.maximum_longitude ≈ 202 + 1/4
    @test lat.minimum_latitude  ≈ 35  - 1/4
    @test lat.maximum_latitude  ≈ 37  + 1/4

    polar = BoundingBox(longitude=(0, 10), latitude=(-89.95, 89.95))
    plat = CMExt.latitude_bounds_kw(polar, dataset)
    @test plat.minimum_latitude == -90
    @test plat.maximum_latitude == 90
end

@testset "Downloading AVISO data" begin
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    date = DateTime(2020, 1, 1)

    for dataset in (AVISODaily(), AVISOMonthly())
        metadatum = Metadatum(:sea_level_anomaly; dataset, date, region, dir=AVISO_TEST_DIR)
        filepath = metadata_path(metadatum)
        download(metadatum)
        @test isfile(filepath)
    end
end

@testset "Download and set AVISO sea_level_anomaly" begin
    dataset = AVISOMonthly()
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dates = DateTime(2020, 1, 1):Month(1):DateTime(2020, 2, 1)
    metadata = Metadata(:sea_level_anomaly; dates, dataset, region, dir=AVISO_TEST_DIR)

    download(metadata)
    for datum in metadata
        @test isfile(metadata_path(datum))
    end

    for arch in test_architectures
        datum = first(metadata)
        source = Field(datum, arch; inpainting=nothing)
        @test source isa Field
        @test size(interior(source), 3) == 1
        @test topology(source.grid) == (Bounded, Bounded, Flat)
        @test all(isfinite.(Array(interior(source))))

        target_grid = LatitudeLongitudeGrid(arch;
                                            size = (4, 4, 3),
                                            longitude = (200.5, 201.5),
                                            latitude = (35.5, 36.5),
                                            z = (-1000, 0))
        target = CenterField(target_grid)
        set!(target, datum; inpainting=nothing)
        @test all(isfinite.(Array(interior(target))))

        fts = FieldTimeSeries(metadata, arch; inpainting=nothing)
        @test fts isa FieldTimeSeries
        @test size(interior(fts), 3) == 1
        @test length(fts.times) == length(dates)
        @test time_indices(fts) == Tuple(1:length(dates))
    end
end
