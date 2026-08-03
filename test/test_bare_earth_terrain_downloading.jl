include("runtests_setup.jl")
include("download_utils.jl")

using NumericalEarth.Bathymetry: bare_earth_elevation, regrid_topography
using NumericalEarth.ETOPO
using Oceananigans.Grids: λnodes, φnodes
using NCDatasets: Dataset

# ETOPO stands in for a DSM here — it is the surface-elevation dataset available without a
# token. Excluded from CI in runtests.jl because it downloads the global ETOPO file.

land_grid(arch; size = (8, 8)) =
    LatitudeLongitudeGrid(arch; size, longitude = (6, 10), latitude = (44, 47),
                          topology = (Bounded, Bounded, Flat))

height_field(grid, data) = set!(Field{Center, Center, Nothing}(grid), data)

etopo_metadatum = Metadatum(:bottom_height; dataset = ETOPO2022())

@testset "Availability of the ETOPO2022 surface elevation" begin
    filepath = metadata_path(etopo_metadatum)
    download_dataset_with_fallback(filepath; dataset_name = "ETOPO2022") do
        download(etopo_metadatum)
    end
    @test isfile(filepath)
end

@testset "bare_earth_elevation — grid-level DSM regrid + subtraction" begin
    for arch in test_architectures
        grid = land_grid(arch)

        object         = height_field(grid, 25.0)
        dsm_elevation  = regrid_topography(grid; dataset = ETOPO2022())
        bare_elevation = bare_earth_elevation(grid, object; dataset = ETOPO2022())

        reference = max.(Array(interior(dsm_elevation, :, :, 1)) .- 25.0, 0)
        @test Array(interior(bare_elevation, :, :, 1)) ≈ reference
    end
end

@testset "regrid_topography — windowed region on a global file" begin
    # ETOPO is one global file, so a windowed region has to read its own slice out of it. A
    # target grid at the dataset's own 1 arc-minute resolution puts target cell centers on
    # ETOPO cell centers, so the regridded elevation must reproduce the file values there —
    # a slice offset by even one cell shows up as hundreds of meters in Alpine terrain.
    grid = LatitudeLongitudeGrid(CPU(); size = (240, 180), longitude = (6, 10), latitude = (44, 47),
                                 topology = (Bounded, Bounded, Flat))
    region = BoundingBox(longitude = (5, 11), latitude = (43, 48))

    windowed = regrid_topography(grid; dataset = ETOPO2022(), region, cache = false)
    elevation = Array(interior(windowed, :, :, 1))
    @test !any(isnan, elevation)

    metadatum = Metadatum(:bottom_height; dataset = ETOPO2022(), region)
    dataset = Dataset(metadata_path(metadatum))
    file_elevation = dataset["z"][:, :]
    λfile = dataset["lon"][:]
    φfile = dataset["lat"][:]
    close(dataset)

    i★ = [argmin(abs.(λfile .- λ)) for λ in λnodes(grid, Center())]
    j★ = [argmin(abs.(φfile .- φ)) for φ in φnodes(grid, Center())]
    reference = max.(coalesce.(file_elevation[i★, j★], 0), 0)  # regrid_topography clamps ocean to 0

    @test sum(abs, elevation .- reference) / length(elevation) < 10  # metres
end
