include("runtests_setup.jl")
include("download_utils.jl")

using CondaPkg
CondaPkg.add("h5py"; channel="conda-forge", version=">=3.0,<3.13")
CondaPkg.add("hdf5"; channel="conda-forge", version="<2")

using CopernicusMarine

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, metadata_path

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
    variables = (:free_surface, :sea_level_anomaly, :zonal_geostrophic_velocity, :meridional_geostrophic_velocity)
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = AVISOMonthly()
    date = DateTime(2020, 1, 1)


    for variable in variables
        metadatum = Metadatum(variable; dataset, date, region)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        download(metadatum)
        @test isfile(filepath)
    end
end
