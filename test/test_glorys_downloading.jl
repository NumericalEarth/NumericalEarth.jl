include("runtests_setup.jl")
include("download_utils.jl")

# The CopernicusMarine standalone executable bundles its own HDF5/h5py, so the
# previous in-process CondaPkg h5py/hdf5 pinning is no longer required.
using CopernicusMarine

using NumericalEarth.DataWrangling: BoundingBox, is_three_dimensional, z_interfaces
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily
using Oceananigans.Fields: location

@testset "GLORYS CopernicusMarine fetch padding" begin
    # `restrict` center-brackets the native grid, so the CopernicusMarine subset
    # must over-fetch a couple of native cells to cover it (otherwise
    # set_region_data! indexes past the file at the domain edge). No network needed.
    CMExt = Base.get_extension(NumericalEarth, :NumericalEarthCopernicusMarineExt)
    bbox = BoundingBox(longitude=(200, 202), latitude=(35, 37))

    lon = CMExt.longitude_bounds_kw(bbox)
    lat = CMExt.latitude_bounds_kw(bbox)

    @test lon.minimum_longitude ≈ 200 - 2/12
    @test lon.maximum_longitude ≈ 202 + 2/12
    @test lat.minimum_latitude  ≈ 35  - 2/12
    @test lat.maximum_latitude  ≈ 37  + 2/12

    # Latitude padding clamps to the poles.
    polar = BoundingBox(longitude=(0, 10), latitude=(-89.95, 89.95))
    plat = CMExt.latitude_bounds_kw(polar)
    @test plat.minimum_latitude == -90
    @test plat.maximum_latitude == 90
end

@testset "Downloading GLORYS data" begin
    variables = (:temperature, :salinity, :u_velocity, :v_velocity, :free_surface)
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = GLORYSDaily()
    for variable in variables
        metadatum = Metadatum(variable; dataset, region)
        filepath = NumericalEarth.DataWrangling.metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        download(metadatum)
        @test isfile(filepath)
    end
end

@testset "Downloading GLORYSBGC data" begin
    variables = (:phytoplankton, :nitrate)
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = GLORYSAnalysisForecastBGCDaily()
    for variable in variables
        metadatum = Metadatum(variable; dataset, region, date = DateTime(now()))
        filepath = NumericalEarth.DataWrangling.metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        download(metadatum)
        @test isfile(filepath)
    end
end

@testset "Download and set GLORYS free_surface" begin
    for arch in test_architectures
        region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
        dataset = GLORYSDaily()
        md = Metadatum(:free_surface; dataset, region)

        @test !is_three_dimensional(md)
        @test location(md) === (Center, Center, Nothing)
        @test z_interfaces(md) === (-1.0, 0.0)

        source = Field(md, arch; inpainting=nothing)
        @test source isa Field
        @test size(interior(source), 3) == 1

        target_grid = LatitudeLongitudeGrid(arch;
                                            size = (4, 4, 3),
                                            longitude = (200.5, 201.5),
                                            latitude = (35.5, 36.5),
                                            z = (-1000, 0))
        target = CenterField(target_grid)
        set!(target, md; inpainting=nothing)

        interior_target = Array(interior(target))
        @test all(isfinite.(interior_target))
    end
end
