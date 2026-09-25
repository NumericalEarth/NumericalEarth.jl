include("runtests_setup.jl")
include("download_utils.jl")

using NumericalEarth.Bathymetry: bare_earth_elevation, regrid_topography
using NumericalEarth.ETOPO

# ETOPO stands in for a DSM here — it is the surface-elevation dataset available without a
# token. Excluded from CI in runtests.jl because it downloads the global ETOPO file.
# `land_grid` and `height_field` come from runtests_setup.jl.

etopo_metadatum = Metadatum(:bottom_height; dataset = ETOPO2022())

# Windowed and unwindowed reads interpolate from the same ETOPO cells, so they agree to the
# labeling precision of the two native grids: Float32 near 180° resolves a thousandth of a cell,
# which over the steepest Alpine cells is a few tenths of a meter. A slice offset by one cell moves
# the same terrain by tens to hundreds of meters.
window_tolerance = 1 # meter

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
    # ETOPO is one global file, so a windowed region reads its own block out of it. Windowing is
    # only an optimization, so the unwindowed read of the same grid is the reference.
    grid = LatitudeLongitudeGrid(CPU(); size = (240, 180), longitude = (6, 10), latitude = (44, 47),
                                 topology = (Bounded, Bounded, Flat))

    windowed = regrid_topography(grid; dataset = ETOPO2022(), cache = false,
                                 region = BoundingBox(longitude = (5, 11), latitude = (43, 48)))
    unwindowed = regrid_topography(grid; dataset = ETOPO2022(), cache = false)

    windowed_elevation = Array(interior(windowed, :, :, 1))
    @test !any(isnan, windowed_elevation)
    @test maximum(abs, windowed_elevation .- Array(interior(unwindowed, :, :, 1))) < window_tolerance
end

@testset "regrid_topography — a windowed region across the antimeridian" begin
    # The window runs past the file's last column and continues into its first, which the read has
    # to follow rather than sliding the window back inside the file.
    grid = LatitudeLongitudeGrid(CPU(); size = (120, 60), longitude = (170, 190), latitude = (-10, 10),
                                 topology = (Bounded, Bounded, Flat))

    windowed = regrid_topography(grid; dataset = ETOPO2022(), cache = false,
                                 region = BoundingBox(longitude = (169, 191), latitude = (-11, 11)))
    unwindowed = regrid_topography(grid; dataset = ETOPO2022(), cache = false)

    @test maximum(abs, Array(interior(windowed, :, :, 1)) .-
                       Array(interior(unwindowed, :, :, 1))) < window_tolerance
end
