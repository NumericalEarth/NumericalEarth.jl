include("runtests_setup.jl")

using NumericalEarth.Bathymetry: bare_earth_elevation, bathymetry_regridding_key
using NumericalEarth.DataWrangling: validate_dataset_coverage, validate_region_covers_grid,
                                    default_region, dataset_bounding_box, native_grid,
                                    file_cell_index, region_info, file_window, wrapped_i,
                                    set_region_data!
using NumericalEarth.DataWrangling.CopernicusDEM: GLO30
using NumericalEarth.ETOPO
using Oceananigans.Grids: λnodes

# Tests that download a surface-elevation dataset live in
# test_bare_earth_terrain_downloading.jl. `land_grid` and `height_field` come from
# runtests_setup.jl.

# ETOPO's 1 arc-minute cell centers, the lattice a windowed read matches a grid node against.
etopo_longitudes() = collect(-180 + 1/120 : 1/60 : 180)
etopo_latitudes()  = collect(-90 + 1/120 : 1/60 : 90)

window_field(region) =
    Field{Center, Center, Nothing}(native_grid(Metadatum(:bottom_height; dataset = ETOPO2022(), region);
                                              halo = (10, 10, 1)))

@testset "bare_earth_elevation — object-height subtraction" begin
    for arch in test_architectures
        grid   = land_grid(arch)
        Nx, Ny = size(grid)

        surface  = [100.0 + 10 * (i + j) for i in 1:Nx, j in 1:Ny]  # 120–180 m DSM
        canopy   = fill(30.0, Nx, Ny); canopy[1, 1]   = 500.0       # taller than the DSM here
        building = fill(10.0, Nx, Ny); building[Nx, Ny] = 80.0      # buildings win here

        z = bare_earth_elevation(height_field(grid, surface),
                                 (height_field(grid, canopy),
                                  height_field(grid, building)))

        @test z isa Field{Center, Center, Nothing}
        zi = Array(interior(z, :, :, 1))

        # z_bare = max(surface − max(canopy, building), 0), combined per cell.
        reference = max.(surface .- max.(canopy, building), 0)
        @test zi ≈ reference

        # Clamped at sea level where an object is taller than the surface.
        @test zi[1, 1] == 0
        # Never negative anywhere.
        @test all(zi .>= 0)
        # The taller object is the one removed.
        @test zi[Nx, Ny] ≈ surface[Nx, Ny] - 80.0
    end
end

@testset "bare_earth_elevation — missing object heights count as zero" begin
    for arch in test_architectures
        grid   = land_grid(arch)
        Nx, Ny = size(grid)

        # Canopy defined only over one cell, buildings only over another; NaN elsewhere.
        canopy   = fill(NaN, Nx, Ny); canopy[1, 1]   = 30.0
        building = fill(NaN, Nx, Ny); building[2, 2] = 20.0

        z = bare_earth_elevation(height_field(grid, fill(100.0, Nx, Ny)),
                                 (height_field(grid, canopy),
                                  height_field(grid, building)))
        zi = Array(interior(z, :, :, 1))

        @test !any(isnan, zi)         # NaN object heights never leak into the terrain
        @test zi[1, 1] ≈ 70.0         # canopy removed
        @test zi[2, 2] ≈ 80.0         # building removed
        @test zi[3, 3] ≈ 100.0        # no object → surface unchanged
    end
end

@testset "regrid_bathymetry — a window and a global read get different cache keys" begin
    grid = land_grid(CPU())
    parameters = (; height_above_water = nothing, minimum_depth = 0,
                    interpolation_passes = 1, major_basins = 1)

    global_key = bathymetry_regridding_key(grid, Metadatum(:bottom_height; dataset = ETOPO2022()); parameters...)

    region = default_region(GLO30(), grid)
    windowed_key = bathymetry_regridding_key(grid, Metadatum(:bottom_height; dataset = ETOPO2022(), region); parameters...)

    # A window and a global read of the same dataset+grid must not collide on disk.
    @test global_key.region != windowed_key.region
    @test hash(global_key) != hash(windowed_key)
end

@testset "bare_earth_elevation — region auto-derived per dataset" begin
    grid = land_grid(CPU())

    # The global 30 m product must be windowed: a GLO30 metadatum with no region is rejected.
    @test_throws ErrorException validate_dataset_coverage(grid, Metadatum(:bottom_height; dataset = GLO30()))

    # GLO30 derives a bounded window from the grid, so the metadatum passes validation
    # without the caller supplying a BoundingBox.
    region = default_region(GLO30(), grid)
    @test region isa BoundingBox
    @test validate_dataset_coverage(grid, Metadatum(:bottom_height; dataset = GLO30(), region)) === nothing

    # ETOPO is a global file read whole, so it needs no window.
    @test default_region(ETOPO2022(), grid) === nothing
end

@testset "windowed reads — matching a grid node against the file's labels" begin
    longitudes = etopo_longitudes()

    # A file labeling cells half a cell from the grid's centers (ERA5's quarter-degree latitude
    # labels against centers on 2.375 + k/4) picks the label at or after each center.
    @test file_cell_index(2.375, collect(2.0:0.25:4.0)) == 3

    # Labels on the grid's own lattice match exactly, Float32 promotion of the node included.
    @test file_cell_index(Float64(Float32(longitudes[11100])), longitudes) == 11100

    # At the file's west edge the Float32 node lands just below the first label, which must not
    # wrap by 360° and select the far end of the file.
    west = BoundingBox(longitude = (-180, -174), latitude = (0, 4))
    @test region_info(west, window_field(west), longitudes, etopo_latitudes()).di == 0
end

@testset "windowed reads — a window across the seam continues into the file" begin
    longitudes = etopo_longitudes()
    latitudes  = etopo_latitudes()

    region = BoundingBox(longitude = (170, 190), latitude = (-10, 10))
    field  = window_field(region)
    Nx, _, _ = size(field)

    offset = region_info(region, field, longitudes, latitudes)
    nodes  = λnodes(field.grid, Center())

    first_column = wrapped_i(1 + offset.di, offset.Nλ)
    last_column  = wrapped_i(Nx + offset.di, offset.Nλ)

    @test longitudes[first_column] ≈ minimum(nodes)
    @test last_column < first_column                            # wrapped past the file's last column
    @test longitudes[last_column] + 360 ≈ maximum(nodes)

    # A wrapped window is not one contiguous block, so the whole variable is read.
    @test isnothing(file_window(region, field, longitudes, latitudes))
end

@testset "windowed reads — an interior window reads one block of the file" begin
    longitudes = etopo_longitudes()
    latitudes  = etopo_latitudes()

    region = BoundingBox(longitude = (5, 11), latitude = (43, 48))
    field  = window_field(region)
    Nx, Ny, _ = size(field)

    offset = region_info(region, field, longitudes, latitudes)
    window = file_window(region, field, longitudes, latitudes)
    @test !isnothing(window)

    longitude_range, latitude_range = window
    @test length(longitude_range) == Nx
    @test length(latitude_range) == Ny

    # The block starts at the column the kernel would have offset to, so reading the block leaves
    # nothing to offset.
    @test first(longitude_range) == offset.di + 1
    @test region_info(region, field, longitudes[longitude_range], latitudes[latitude_range]).di == 0
end

@testset "windowed reads — data lands on the coordinates that carry it" begin
    # Cell centers -0.5 … 4.5: the native grid of a (0, 4) box on a 1° lattice, one cell wider
    # than the box on each side because `restrict` brackets it with cell centers.
    grid   = LatitudeLongitudeGrid(CPU(); size = (6, 6), longitude = (-1, 5), latitude = (-1, 5),
                                   topology = (Bounded, Bounded, Flat))
    field  = Field{Center, Center, Nothing}(grid)
    region = BoundingBox(longitude = (0, 4), latitude = (0, 4))
    metadatum = Metadatum(:bottom_height; dataset = ETOPO2022(), region)

    # A file that is exactly the grid's window: no offset, values stay put.
    exact = collect(-0.5:1.0:4.5)
    data  = reshape(Float64[10i + j for i in 1:6, j in 1:6], 6, 6, 1)
    offset = region_info(region, field, exact, exact)
    @test (offset.di, offset.dj) == (0, 0)
    set_region_data!(field, data, exact, exact, metadatum)
    @test interior(field, :, :, 1) == data[:, :, 1]

    # A file over-fetched by two cells per side: the same values, located by coordinate.
    padded      = collect(-2.5:1.0:6.5)
    padded_data = reshape(Float64[10i + j for i in 1:10, j in 1:10], 10, 10, 1)
    offset = region_info(region, field, padded, padded)
    @test (offset.di, offset.dj) == (2, 2)
    set_region_data!(field, padded_data, padded, padded, metadatum)
    @test interior(field, :, :, 1) == padded_data[3:8, 3:8, 1]

    # A file spanning the region rather than the grid reaches neither edge, and is rejected
    # instead of writing its values one cell off.
    short = collect(0.5:1.0:3.5)
    short_data = reshape(Float64[10i + j for i in 1:4, j in 1:4], 4, 4, 1)
    @test_throws ArgumentError region_info(region, field, short, short)
    @test_throws ArgumentError set_region_data!(field, short_data, short, short, metadatum)

    # A file starting north of the grid's first row is rejected as well.
    @test_throws ArgumentError region_info(region, field, exact, collect(0.5:1.0:7.5))
end

@testset "windowed reads — guards on region and padding" begin
    grid = land_grid(CPU())

    # An explicit region that misses the grid is rejected instead of extrapolated.
    @test_throws ArgumentError validate_region_covers_grid(grid, BoundingBox(longitude = (0, 1), latitude = (0, 1)))
    @test validate_region_covers_grid(grid, BoundingBox(longitude = (5, 11), latitude = (43, 48))) === nothing
    @test validate_region_covers_grid(grid, default_region(GLO30(), grid)) === nothing
    @test validate_region_covers_grid(grid, nothing) === nothing

    # A grid crossing the dataset's longitude seam cannot be covered by one window.
    antimeridian = LatitudeLongitudeGrid(CPU(); size = (8, 8), longitude = (170, 190),
                                         latitude = (-25, -10), topology = (Bounded, Bounded, Flat))
    @test_throws ArgumentError dataset_bounding_box(GLO30(), antimeridian)

    # Padding is a number or one value per axis.
    @test_throws ArgumentError BoundingBox(grid; padding = (0.5,))
    @test_throws ArgumentError BoundingBox(grid; padding = [0.5, 0.25])

    # A window smaller than the requested halo still builds a grid.
    tiny = BoundingBox(longitude = (6.0, 6.05), latitude = (44.0, 44.05))
    tiny_grid = native_grid(Metadatum(:bottom_height; dataset = ETOPO2022(), region = tiny); halo = (10, 10, 1))
    @test tiny_grid.Hx ≤ size(tiny_grid, 1)
    @test tiny_grid.Hy ≤ size(tiny_grid, 2)
end
