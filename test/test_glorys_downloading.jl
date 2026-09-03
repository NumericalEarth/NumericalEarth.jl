include("runtests_setup.jl")
include("download_utils.jl")
include("dataset_status.jl")

# The CopernicusMarine standalone executable bundles its own HDF5/h5py, so the
# previous in-process CondaPkg h5py/hdf5 pinning is no longer required.
using CopernicusMarine

using NumericalEarth.DataWrangling: BoundingBox, is_three_dimensional, z_interfaces, metadata_path
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily, GLORYSMonthly, GLORYSStatic
using Oceananigans.Fields: location

@testset "GLORYS CopernicusMarine fetch padding" begin
    # `restrict` center-brackets the native grid, so the CopernicusMarine subset
    # must over-fetch a couple of native cells to cover it (otherwise
    # set_region_data! indexes past the file at the domain edge). No network needed.
    CMExt = Base.get_extension(NumericalEarth, :NumericalEarthCopernicusMarineExt)
    bbox = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = GLORYSDaily()

    lon = CMExt.longitude_bounds_kw(bbox, dataset)
    lat = CMExt.latitude_bounds_kw(bbox, dataset)

    @test lon.minimum_longitude ≈ 200 - 2/12
    @test lon.maximum_longitude ≈ 202 + 2/12
    @test lat.minimum_latitude  ≈ 35  - 2/12
    @test lat.maximum_latitude  ≈ 37  + 2/12

    # Latitude padding clamps to the poles.
    polar = BoundingBox(longitude=(0, 10), latitude=(-89.95, 89.95))
    plat = CMExt.latitude_bounds_kw(polar, dataset)
    @test plat.minimum_latitude == -90
    @test plat.maximum_latitude == 90
end

@testset "Downloading GLORYS data" begin
    variables = (:temperature, :salinity, :u_velocity, :v_velocity, :free_surface)
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = GLORYSDaily()
    for variable in variables
        @dataset_check "GLORYSDaily" string(variable) begin
            metadatum = Metadatum(variable; dataset, region)
            filepath = metadata_path(metadatum)
            isfile(filepath) && rm(filepath; force=true)
            download(metadatum)
            isfile(filepath) || error("GLORYSDaily $(variable) download produced no file at $(filepath)")
            filepath
        end
    end
end

# `GLORYSMonthly` and `GLORYSStatic` share the CopernicusMarine path with `GLORYSDaily` but
# resolve to different product IDs, so a monthly or static product going away is invisible
# to the daily check above. One variable each is enough to catch that.
@testset "Downloading GLORYS monthly and static products" begin
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))

    @dataset_check "GLORYSMonthly" "temperature" begin
        metadatum = Metadatum(:temperature; dataset=GLORYSMonthly(), region)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        download(metadatum)
        isfile(filepath) || error("GLORYSMonthly temperature download produced no file at $(filepath)")
        filepath
    end

    @dataset_check "GLORYSStatic" "depth" begin
        metadatum = Metadatum(:depth; dataset=GLORYSStatic(), region)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        download(metadatum)
        isfile(filepath) || error("GLORYSStatic depth download produced no file at $(filepath)")
        filepath
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
