include("runtests_setup.jl")

using NumericalEarth.DataWrangling.ECOSTRESS: ECOSTRESSL2G, ecostress_lst,
                                              granule_timestamp, ecostress_overpasses,
                                              ecostress_cmr_url, ECOSTRESS_RESOLUTION
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, Metadata, native_grid,
                                    retrieve_data,
                                    dataset_variable_name, metadata_filename,
                                    available_variables, is_three_dimensional,
                                    all_dates, missing_value, default_inpainting,
                                    longitude_name, latitude_name,
                                    longitude_interfaces, latitude_interfaces,
                                    validate_dataset_coverage

using Oceananigans.Grids: λnodes, φnodes
using NCDatasets: NCDataset, defDim, defVar
using Dates: DateTime

@testset "ECOSTRESS LST decode" begin
    # Kelvin passes through unchanged when clear-sky and physical.
    @test ecostress_lst(300.0f0, 0) === 300.0f0
    @test ecostress_lst(213.5f0, 0) === 213.5f0

    # Cloud mask (nonzero) → NaN.
    @test isnan(ecostress_lst(300.0f0, 1))
    @test isnan(ecostress_lst(300.0f0, 2))

    # 0-fill / non-positive → NaN (off-swath / no-retrieval).
    @test isnan(ecostress_lst(0.0f0, 0))
    @test isnan(ecostress_lst(-5.0f0, 0))

    # NaN passes through as NaN.
    @test isnan(ecostress_lst(NaN32, 0))
end

@testset "ECOSTRESS granule timestamp parse" begin
    dt = granule_timestamp("ECOv002_L2G_LSTE_16928_005_20210701T082749_0710_01")
    @test dt == DateTime(2021, 7, 1, 8, 27, 49)

    @test_throws ArgumentError granule_timestamp("no_timestamp_here")
end

@testset "ECOSTRESS CMR request URL (network-free)" begin
    region = BoundingBox(longitude = (-101, -100), latitude = (33.5, 34.5))
    url = ecostress_cmr_url("002", region, DateTime(2021, 7, 1), DateTime(2021, 7, 3))
    @test occursin("short_name=ECO_L2G_LSTE", url)
    @test occursin("version=002", url)
    @test occursin("bounding_box=-101,33.5,-100,34.5", url)
    @test occursin("temporal=2021-07-01T00:00:00Z,2021-07-03T00:00:00Z", url)

    # An unbounded region cannot be searched.
    @test_throws ArgumentError ecostress_cmr_url("002", BoundingBox(), DateTime(2021, 7, 1), DateTime(2021, 7, 3))
end

@testset "ECOSTRESSL2G dataset interface" begin
    dataset = ECOSTRESSL2G()
    @test dataset.version == "002"
    @test ECOSTRESSL2G(version = "001").version == "001"
    @test occursin("ECOSTRESSL2G", sprint(show, dataset))

    vars = available_variables(dataset)
    @test vars[:land_surface_temperature] == "LST"
    @test vars[:lst_uncertainty] == "LST_err"

    @test longitude_interfaces(dataset) == (-180, 180)
    @test latitude_interfaces(dataset) == (-52, 52)

    region = BoundingBox(longitude = (-101, -100), latitude = (33.5, 34.5))
    metadatum = Metadatum(:land_surface_temperature; dataset, region,
                          date = DateTime(2021, 7, 1, 8, 27, 49))
    @test dataset_variable_name(metadatum) == "LST"
    @test is_three_dimensional(metadatum) == false
    @test isnan(missing_value(metadatum))
    @test default_inpainting(metadatum) === nothing
    @test longitude_name(metadatum) == "lon"
    @test latitude_name(metadatum) == "lat"
    @test location(metadatum) == (Center, Center, Center)
    @test eltype(metadatum) == Float32

    # The uncertainty shares the dataset but maps to its own layer name.
    err = Metadatum(:lst_uncertainty; dataset, region, date = DateTime(2021, 7, 1, 8, 27, 49))
    @test dataset_variable_name(err) == "LST_err"
end

@testset "ECOSTRESS irregular dates require explicit discovery" begin
    @test_throws ErrorException all_dates(ECOSTRESSL2G(), :land_surface_temperature)
end

@testset "ECOSTRESS filenames + coverage validation" begin
    dataset = ECOSTRESSL2G()
    date = DateTime(2021, 7, 1, 8, 27, 49)
    region_a = BoundingBox(longitude = (-101, -100), latitude = (33.5, 34.5))
    region_b = BoundingBox(longitude = (10, 11), latitude = (45, 46))

    # One local file per overpass+region holds every variable — the filename is keyed
    # by date and region but not by variable name.
    @test metadata_filename(dataset, :land_surface_temperature, date, region_a) ==
          metadata_filename(dataset, :lst_uncertainty, date, region_a)

    # Distinct per region and per date.
    @test metadata_filename(dataset, :land_surface_temperature, date, region_a) !=
          metadata_filename(dataset, :land_surface_temperature, date, region_b)
    @test metadata_filename(dataset, :land_surface_temperature, date, region_a) !=
          metadata_filename(dataset, :land_surface_temperature, DateTime(2021, 7, 2), region_a)

    grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 1),
                                 longitude = (-101, -100), latitude = (33.5, 34.5),
                                 z = (0, 1))
    # Global (unbounded) metadatum is rejected; a BoundingBox is accepted.
    global_md = Metadatum(:land_surface_temperature; dataset)
    @test_throws ErrorException validate_dataset_coverage(grid, global_md)
    region_md = Metadatum(:land_surface_temperature; dataset, region = region_a)
    @test validate_dataset_coverage(grid, region_md) === nothing
end

# The real fetch (Earthdata download + GDAL HDF5 read) is credential/ArchGDAL-gated,
# so here we synthesize the regional lon/lat NetCDF the download step would write and
# exercise the generic read + windowing machinery end-to-end. Placing the file at the
# metadatum's path makes `Downloads.download` a no-op (the file already exists).
@testset "ECOSTRESS reads a regional LST raster onto its grid" begin
    dataset = ECOSTRESSL2G()
    region = BoundingBox(longitude = (-100.03, -100.0), latitude = (34.0, 34.03))
    date = DateTime(2021, 7, 1, 8, 27, 49)

    mktempdir() do dir
        metadatum = Metadatum(:land_surface_temperature; dataset, region, date, dir)

        # Build the file exactly on the native-grid cell centers so the raster maps
        # one-to-one onto the grid the reader reconstructs.
        grid = native_grid(metadatum)
        Nx, Ny = grid.Nx, grid.Ny
        @test Nx > 20 && Ny > 20   # the 0.03° window resolves at ECOSTRESS_RESOLUTION
        λc = Array(λnodes(grid, Center(), Center(), Center()))
        φc = Array(φnodes(grid, Center(), Center(), Center()))

        LST = Float32[290 + 0.1f0 * i + 0.05f0 * j for i in 1:Nx, j in 1:Ny]
        LST[3, 3] = NaN32          # interior cloud/no-retrieval gaps, clear of the interp box
        LST[Nx - 2, Ny - 2] = NaN32
        LST_err = fill(1.5f0, Nx, Ny)

        NCDataset(metadata_path(metadatum), "c") do ds
            defDim(ds, "lon", Nx)
            defDim(ds, "lat", Ny)
            defVar(ds, "lon", Float64, ("lon",))[:] = λc
            defVar(ds, "lat", Float64, ("lat",))[:] = φc
            defVar(ds, "LST", Float32, ("lon", "lat"))[:, :] = LST
            defVar(ds, "LST_err", Float32, ("lon", "lat"))[:, :] = LST_err
        end

        # retrieve_data returns the decoded 2-D raster with a singleton 3rd dim.
        raw = retrieve_data(metadatum)
        @test size(raw) == (Nx, Ny, 1)

        field = Field(metadatum)
        interior_data = Array(interior(field))
        @test size(interior_data) == (Nx, Ny, 1)

        # Cloud gaps survive as NaN (never inpainted); the rest matches the raster.
        @test isequal(interior_data[:, :, 1], LST)
        @test count(isnan, interior_data) == 2
        finite = filter(isfinite, interior_data)
        @test all(f -> 280 < f < 320, finite)

        # And it interpolates onto a coarser LES-like grid over a clear-sky central box.
        target = LatitudeLongitudeGrid(CPU(); size = (12, 12),
                                       longitude = (-100.018, -100.012),
                                       latitude = (34.012, 34.018),
                                       topology = (Bounded, Bounded, Flat))
        les_field = Field(metadatum, target)
        les_data = Array(interior(les_field))
        @test size(les_data) == (12, 12, 1)
        @test all(isfinite, les_data)
        @test all(f -> 280 < f < 320, les_data)
    end
end
