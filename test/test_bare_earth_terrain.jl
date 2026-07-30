include("runtests_setup.jl")

using NumericalEarth.Bathymetry: bare_earth_elevation, BathymetryRegridding
using NumericalEarth.DataWrangling: validate_dataset_coverage, default_region
using NumericalEarth.DataWrangling.CopernicusDEM: GLO30
using NumericalEarth.ETOPO

# Tests that download a surface-elevation dataset live in
# test_bare_earth_terrain_downloading.jl.

# A 2-D land grid (Flat in the vertical), matching how terrain fields are built.
land_grid(arch; size = (8, 8)) =
    LatitudeLongitudeGrid(arch; size, longitude = (6, 10), latitude = (44, 47),
                          topology = (Bounded, Bounded, Flat))

height_field(grid, data) = set!(Field{Center, Center, Nothing}(grid), data)

@testset "bare_earth_elevation — object-height subtraction" begin
    for arch in test_architectures
        grid   = land_grid(arch)
        Nx, Ny = size(grid)

        surface  = [100.0 + 10 * (i + j) for i in 1:Nx, j in 1:Ny]  # 120–180 m DSM
        canopy   = fill(30.0, Nx, Ny); canopy[1, 1]   = 500.0       # taller than the DSM here
        building = fill(10.0, Nx, Ny); building[Nx, Ny] = 80.0      # buildings win here

        z = bare_earth_elevation(height_field(grid, surface),
                                 height_field(grid, canopy),
                                 height_field(grid, building))

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
                                 height_field(grid, canopy),
                                 height_field(grid, building))
        zi = Array(interior(z, :, :, 1))

        @test !any(isnan, zi)         # NaN object heights never leak into the terrain
        @test zi[1, 1] ≈ 70.0         # canopy removed
        @test zi[2, 2] ≈ 80.0         # building removed
        @test zi[3, 3] ≈ 100.0        # no object → surface unchanged
    end
end

@testset "bare_earth_elevation — no objects reduces to the surface" begin
    for arch in test_architectures
        grid   = land_grid(arch)
        Nx, Ny = size(grid)
        surface = [50.0 + i - j for i in 1:Nx, j in 1:Ny]

        z = bare_earth_elevation(height_field(grid, surface))
        @test Array(interior(z, :, :, 1)) ≈ max.(surface, 0)
    end
end

@testset "regrid_bathymetry — global regrids keep a region-free cache key" begin
    grid = land_grid(CPU())

    global_config = BathymetryRegridding(grid, Metadatum(:bottom_height; dataset = ETOPO2022()))
    @test isnothing(global_config.region)

    region = default_region(GLO30(), grid)
    windowed_config = BathymetryRegridding(grid, Metadatum(:bottom_height; dataset = ETOPO2022(), region))
    @test windowed_config.region isa String

    # A window and a global read of the same dataset+grid must not collide on disk.
    @test hash(global_config) != hash(windowed_config)
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
