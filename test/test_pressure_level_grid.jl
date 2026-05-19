include("runtests_setup.jl")

using Oceananigans
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: Flat, Bounded, topology
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Statistics

using NumericalEarth.Grids: PressureLevelGrid, PressureLevelVerticalDiscretization

# Build a small static-Field-backed `PressureLevelVerticalDiscretization` from
# a per-cell geopotential array. Returns the (Φ, Φ_sfc, plvd) triple.
function make_plvd(arch=CPU(); Nx=2, Ny=2, Nz=5,
                                heights = collect(1.0:Nz),  # one entry per level, in km
                                g = 9.81)
    Φ_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz),
                                   longitude=(0, 1), latitude=(0, 1), z=(0, 1))
    Φ = CenterField(Φ_grid)

    # Per-cell geopotential: φ(i, j, k) = (100*i + j + 10*k) * 1000 m * g
    Φ_data = [(100i + j + 1000 * heights[k]) * g for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    interior(Φ) .= Φ_data

    Φ_sfc_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, 1),
                                       longitude=(0, 1), latitude=(0, 1), z=(0, 1))
    Φ_sfc = CenterField(Φ_sfc_grid)
    interior(Φ_sfc) .= 0  # entire column is above the surface

    plvd = PressureLevelVerticalDiscretization(Φ;
                                               gravitational_acceleration=g,
                                               surface_geopotential=Φ_sfc)
    return Φ, Φ_sfc, plvd
end

# Build the corresponding `LatitudeLongitudeGrid`. Topology can be set to
# (Flat, Flat, Bounded) for a column source.
function make_plg(arch=CPU(); Nx=2, Ny=2, Nz=5, topology=(Bounded, Bounded, Bounded), kw...)
    Φ, Φ_sfc, plvd = make_plvd(arch; Nx, Ny, Nz, kw...)
    size = topology[1] === Flat && topology[2] === Flat ? Nz : (Nx, Ny, Nz)
    grid = LatitudeLongitudeGrid(arch; size, longitude=(0, 1), latitude=(0, 1),
                                 z=plvd, topology)
    return grid, Φ, Φ_sfc, plvd
end

@testset "PressureLevelVerticalDiscretization" begin
    g = 9.81

    @testset "constructor and grid generation" begin
        _, _, plvd = make_plvd()
        @test plvd isa PressureLevelVerticalDiscretization
        @test plvd.gravitational_acceleration == g

        grid = LatitudeLongitudeGrid(CPU(); size=(2, 2, 5),
                                     longitude=(0, 1), latitude=(0, 1), z=plvd)
        @test grid isa PressureLevelGrid
        # `Lz` was derived from `extrema(geopotential) / g`.
        Nz = grid.Nz
        Φi = interior(plvd.geopotential)
        z_lo, z_hi = extrema(Φi) ./ g
        @test grid.Lz ≈ (z_hi - z_lo)
        @test sprint(show, plvd) == "PressureLevelVerticalDiscretization with 5 levels, g = 9.81 m/s²"
    end

    @testset "generate_coordinate dim/axis guards" begin
        _, _, plvd = make_plvd()
        gen = Oceananigans.Grids.generate_coordinate
        # `dim != 3` should throw.
        @test_throws ArgumentError gen(Float64, (Bounded, Bounded, Bounded),
                                        (2, 2, 5), (1, 1, 1), plvd, :z, 1, CPU())
        # `coordinate_name != :z` should throw.
        @test_throws ArgumentError gen(Float64, (Bounded, Bounded, Bounded),
                                        (2, 2, 5), (1, 1, 1), plvd, :x, 3, CPU())
    end

    @testset "clip_subsurface! on a Field-backed Φ" begin
        Nx, Ny, Nz = 2, 2, 4
        Φ_grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz),
                                       longitude=(0, 1), latitude=(0, 1), z=(0, 1))
        Φ = CenterField(Φ_grid)
        # Levels [1, 2, 3, 4] km, all positive.
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            interior(Φ)[i, j, k] = 1000.0 * k * g
        end

        # Surface at 2.5 km everywhere — should clip k=1, 2 up to k=2.5's value.
        Φ_sfc_grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, 1),
                                           longitude=(0, 1), latitude=(0, 1), z=(0, 1))
        Φ_sfc = CenterField(Φ_sfc_grid)
        interior(Φ_sfc) .= 2500.0 * g

        # Wrapping into a PLVD constructor runs `clip_subsurface!`.
        plvd = PressureLevelVerticalDiscretization(Φ;
                                                   gravitational_acceleration=g,
                                                   surface_geopotential=Φ_sfc)
        # After clipping: k=1, 2 levels become 2500 m * g; k=3, 4 untouched.
        for i in 1:Nx, j in 1:Ny
            @test interior(plvd.geopotential)[i, j, 1] ≈ 2500.0 * g
            @test interior(plvd.geopotential)[i, j, 2] ≈ 2500.0 * g
            @test interior(plvd.geopotential)[i, j, 3] ≈ 3000.0 * g
            @test interior(plvd.geopotential)[i, j, 4] ≈ 4000.0 * g
        end
    end

    @testset "rnodes / znodes on the grid return the column-mean Vector" begin
        grid, _, _, _ = make_plg()
        Nz = grid.Nz

        z_grid = znodes(grid, Center())
        @test z_grid isa Vector{Float64}
        @test length(z_grid) == Nz

        # All three znodes/rnodes signatures should agree.
        @test znodes(grid, Center()) == znodes(grid, Center(), Center(), Center())
        @test znodes(grid, Center()) == znodes(grid, nothing, nothing, Center())
    end

    @testset "znodes(::Field) on a horizontally-resolved grid → 3-D Field" begin
        grid, _, _, _ = make_plg()
        f = CenterField(grid)
        z_field = znodes(f)
        @test z_field isa Field
        @test size(z_field) == size(f)

        # Per-cell heights match `rnode(i, j, k, grid, ...)`.
        for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
            @test interior(z_field)[i, j, k] ≈
                Oceananigans.Grids.rnode(i, j, k, grid, Center(), Center(), Center())
        end

        # Column-mean over horizontals matches the grid-level znodes.
        @test vec(mean(z_field, dims=(1, 2))) ≈ znodes(grid, Center())
    end

    @testset "znodes(::Field) on horizontally-absent locations → Vector" begin
        # Case A: Flat-Flat topology (e.g. ERA5 Column region).
        col_grid, _, _, _ = make_plg(; topology=(Flat, Flat, Bounded), Nx=1, Ny=1)
        f_col = CenterField(col_grid)
        z_col = znodes(f_col)
        @test z_col isa Vector{Float64}
        @test z_col == znodes(col_grid, Center())

        # Case B: Reduced field with (Nothing, Nothing, Center) location.
        grid, _, _, _ = make_plg()
        f = CenterField(grid)
        interior(f) .= rand(size(f)...)
        fbar = compute!(Field(mean(f, dims=(1, 2))))
        @test instantiated_location(fbar) === (nothing, nothing, Center())
        z_red = znodes(fbar)
        @test z_red isa Vector{Float64}
        @test z_red ≈ znodes(grid, Center())
    end

    @testset "znodes(::FieldTimeSeries) follows the same dispatch" begin
        grid, _, _, _ = make_plg()
        fts = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 1.0, 2.0])
        z = znodes(fts)
        @test z isa Field
        @test size(z) == (grid.Nx, grid.Ny, grid.Nz)

        col_grid, _, _, _ = make_plg(; topology=(Flat, Flat, Bounded), Nx=1, Ny=1)
        fts_col = FieldTimeSeries{Center, Center, Center}(col_grid, [0.0, 1.0])
        @test znodes(fts_col) isa Vector{Float64}
    end

    @testset "TimeSeriesInterpolation-backed Φ ignores halo zeros" begin
        # Regression for PR #241 review: `parent(fts)` includes halo cells
        # filled with zeros, so `extrema` and `mean` over it were dominated
        # by the halos. We must read `interior(fts)` instead.
        Nx, Ny, Nz = 4, 4, 4
        Φ_grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz),
                                       longitude=(0, 1), latitude=(0, 1), z=(0, 1))
        Φ_fts = FieldTimeSeries{Center, Center, Center}(Φ_grid, [0.0, 1.0])
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            Φ_fts[1][i, j, k] = 1000.0 * k * g     # heights {1, 2, 3, 4} km
            Φ_fts[2][i, j, k] = 5000.0 * k * g     # heights {5, 10, 15, 20} km
        end

        tsi  = TimeSeriesInterpolation(Φ_fts, Φ_fts.grid; clock = Clock(time = 0.0))
        plvd = PressureLevelVerticalDiscretization(tsi; gravitational_acceleration = g)
        grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz),
                                     longitude=(0, 1), latitude=(0, 1), z=plvd)
        # Time-mean column-mean: each k averages [k*1000, k*5000]
        @test znodes(grid, Center()) ≈ [3000.0, 6000.0, 9000.0, 12000.0]
        # Lz = max - min = 20*1000 - 1*1000 = 19000.
        @test grid.Lz ≈ 19000.0
    end
end
