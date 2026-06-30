include("runtests_setup.jl")
include("download_utils.jl")

# The CopernicusMarine standalone executable bundles its own HDF5/h5py, so the
# previous in-process CondaPkg h5py/hdf5 pinning is no longer required.
using CopernicusMarine

using NumericalEarth.DataWrangling: BoundingBox, is_three_dimensional, z_interfaces, native_grid, metadata_path
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

    # `z = (z_bottom, z_top)` (negative-downward) maps to positive-downward
    # Copernicus depth bounds. No padding is applied in the vertical.
    deep = BoundingBox(longitude=(200, 202), latitude=(35, 37), z=(-1000, -100))
    depth = CMExt.depth_bounds_kw(deep)
    @test depth.minimum_depth == 100
    @test depth.maximum_depth == 1000

    # A bbox without `z` leaves the subset request unrestricted in depth.
    @test CMExt.depth_bounds_kw(bbox) == NamedTuple()
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

@testset "GLORYS z-restricted download builds a matching grid" begin
    # Restricting z must shrink BOTH the download and the native grid's vertical
    # extent: the file holds only the requested depth levels, so the grid's Nz
    # follows that file rather than the dataset's full 50-level water column.
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37), z=(-500, 0))
    md = Metadatum(:temperature; dataset=GLORYSDaily(), region)
    filepath = metadata_path(md)
    isfile(filepath) && rm(filepath; force=true)
    download(md)
    @test isfile(filepath)

    grid = native_grid(md)
    Nz_full = size(md)[3]
    @test size(grid, 3) < Nz_full
    @test size(grid, 3) == length(z_interfaces(md)) - 1

    # The field loads onto the z-restricted grid without indexing past the file.
    field = Field(md; inpainting=nothing)
    @test field isa Field
    @test size(interior(field), 3) == size(grid, 3)
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
