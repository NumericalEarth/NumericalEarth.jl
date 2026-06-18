include("runtests_setup.jl")

using CondaPkg
CondaPkg.add("h5py"; channel="conda-forge", version=">=3.0,<3.13")
CondaPkg.add("hdf5"; channel="conda-forge", version="<2")

using CopernicusMarine
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, is_three_dimensional, metadata_path, z_interfaces

using Oceananigans: location
using Oceananigans.Fields: interior
using Oceananigans.Grids: Bounded, Flat, LatitudeLongitudeGrid, topology
using Oceananigans.OutputReaders: time_indices

@testset "AVISO monthly metadata tests" begin
    dataset = AVISOMonthly()
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    start_date = DateTime(2020, 1, 1)
    dates = start_date:Month(1):DateTime(2020, 2, 1)
    metadata = Metadata(:free_surface; dates, dataset, region)

    @test size(metadata) == (2880, 1440, 1, length(dates))

    download(metadata)
    for datum in metadata
        @test isfile(metadata_path(datum))
    end

    for arch in test_architectures
        datum = first(metadata)

        @test !is_three_dimensional(datum)
        @test location(datum) === (Center, Center, Nothing)
        @test z_interfaces(datum) === (-1.0, 0.0)

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

        target_data = Array(interior(target))
        @test all(isfinite.(target_data))

        restoring = DatasetRestoring(metadata, arch; rate=1/1000, inpainting=nothing)
        fts = restoring.field_time_series
        @test fts isa FieldTimeSeries
        @test size(interior(fts), 3) == 1
        @test length(fts.times) == length(dates)
        @test time_indices(fts) == Tuple(1:length(dates))
    end
end
