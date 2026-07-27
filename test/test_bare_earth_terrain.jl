include("runtests_setup.jl")

using NumericalEarth.Bathymetry: bare_earth_elevation, regrid_topography
using NumericalEarth.DataWrangling: validate_dataset_coverage, default_region
using NumericalEarth.DataWrangling.CopernicusDEM: GLO30
using NumericalEarth.ETOPO

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

@testset "bare_earth_elevation — grid-level DSM regrid + subtraction" begin
    for arch in test_architectures
        grid = land_grid(arch)

        # ETOPO stands in for a DSM here (it is the dataset available without a token). ETOPO
        # reads globally, GLO-30 is windowed to the grid — the only difference is the region
        # `regrid_topography` auto-derives; the subtraction below is identical for both.
        object = height_field(grid, 25.0)
        z_dsm  = regrid_topography(grid; dataset = ETOPO2022())
        z_bare = bare_earth_elevation(grid, object; dataset = ETOPO2022())

        reference = max.(Array(interior(z_dsm, :, :, 1)) .- 25.0, 0)
        @test Array(interior(z_bare, :, :, 1)) ≈ reference
        @test all(Array(interior(z_bare, :, :, 1)) .>= 0)
    end
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
