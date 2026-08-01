include("runtests_setup.jl")

using Oceananigans
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: Flat, Bounded, topology
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Statistics

using NumericalEarth.Grids: PressureLevelGrid, PressureLevelVerticalDiscretization,
                            column_fractional_z_index

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
    set!(Φ, Φ_data)

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

for arch in test_architectures
    @testset "PressureLevelVerticalDiscretization [$(typeof(arch))]" begin
        g = 9.81

        @testset "constructor and grid generation" begin
            _, _, plvd = make_plvd(arch)
            @test plvd isa PressureLevelVerticalDiscretization
            @test plvd.gravitational_acceleration == g

            grid = LatitudeLongitudeGrid(arch; size=(2, 2, 5),
                                        longitude=(0, 1), latitude=(0, 1), z=plvd)
            @test grid isa PressureLevelGrid
            # `Lz` was derived from `extrema(geopotential) / g`.
            Nz = grid.Nz
            Φi = interior(plvd.geopotential)
            z_lo, z_hi = extrema(Φi) ./ g
            @test grid.Lz ≈ (z_hi - z_lo)
            @test sprint(show, plvd) == "PressureLevelVerticalDiscretization with 5 levels, g = 9.81 m/s²"

            # Regression: `show(io, grid)` used to crash with FieldError because
            # the default `LatitudeLongitudeGrid` show reaches into `grid.z.cᵃᵃᶠ`,
            # which PLVD doesn't carry.
            s2 = sprint(show, grid)
            s3 = sprint(show, MIME"text/plain"(), grid)
            for s in (s2, s3)
                @test occursin("PressureLevelVerticalDiscretization", s)
                @test occursin("Lz", s)
                @test !occursin("FieldError", s)
            end
        end

        @testset "generate_coordinate dim/axis guards" begin
            _, _, plvd = make_plvd(arch)
            gen = Oceananigans.Grids.generate_coordinate
            # `dim != 3` should throw.
            @test_throws ArgumentError gen(Float64, (Bounded, Bounded, Bounded),
                                            (2, 2, 5), (1, 1, 1), plvd, :z, 1, arch)
            # `coordinate_name != :z` should throw.
            @test_throws ArgumentError gen(Float64, (Bounded, Bounded, Bounded),
                                            (2, 2, 5), (1, 1, 1), plvd, :x, 3, arch)
        end

        @testset "clip_subsurface! on a Field-backed Φ" begin
            Nx, Ny, Nz = 2, 2, 4
            Φ_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz),
                                        longitude=(0, 1), latitude=(0, 1), z=(0, 1))
            Φ = CenterField(Φ_grid)
            # Levels [1, 2, 3, 4] km, all positive.
            set!(Φ, [1000.0 * k * g for i in 1:Nx, j in 1:Ny, k in 1:Nz])

            # Surface at 2.5 km everywhere — should clip k=1, 2 up to k=2.5's value.
            Φ_sfc_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, 1),
                                            longitude=(0, 1), latitude=(0, 1), z=(0, 1))
            Φ_sfc = CenterField(Φ_sfc_grid)
            interior(Φ_sfc) .= 2500.0 * g

            # Wrapping into a PLVD constructor runs `clip_subsurface!`.
            plvd = PressureLevelVerticalDiscretization(Φ;
                                                    gravitational_acceleration=g,
                                                    surface_geopotential=Φ_sfc)
            # After clipping: k=1, 2 levels become 2500 m * g; k=3, 4 untouched.
            Φᶜ = Array(interior(plvd.geopotential))
            for i in 1:Nx, j in 1:Ny
                @test Φᶜ[i, j, 1] ≈ 2500.0 * g
                @test Φᶜ[i, j, 2] ≈ 2500.0 * g
                @test Φᶜ[i, j, 3] ≈ 3000.0 * g
                @test Φᶜ[i, j, 4] ≈ 4000.0 * g
            end
        end

        @testset "column_fractional_z_index snaps to the first above-ground level" begin
            Nx, Ny, Nz = 2, 2, 5
            Φ_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz),
                                        longitude=(0, 1), latitude=(0, 1), z=(0, 1))
            Φ = CenterField(Φ_grid)
            # level heights 1..5 km
            set!(Φ, [1000.0 * k * g for i in 1:Nx, j in 1:Ny, k in 1:Nz])
            Φ_sfc_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, 1),
                                            longitude=(0, 1), latitude=(0, 1), z=(0, 1))
            Φ_sfc = CenterField(Φ_sfc_grid)
            interior(Φ_sfc) .= 2500.0 * g            # surface at 2.5 km ⇒ clips k=1,2; first above-ground = k=3
            plvd = PressureLevelVerticalDiscretization(Φ; gravitational_acceleration=g,
                                                    surface_geopotential=Φ_sfc)
            grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), longitude=(0, 1), latitude=(0, 1),
                                        z=plvd, topology=(Bounded, Bounded, Bounded))

            # Clipped column heights (m): [2500, 2500, 3000, 4000, 5000]. Levels 1, 2 still hold the raw
            # sub-surface data, so a target at/below the surface — or between it and the first above-ground
            # level (k=3) — must snap to k=3, never extrapolate into the clipped plateau [1, 3).
            # `column_fractional_z_index` indexes the geopotential field directly, so calling it
            # from the host on a device grid is a scalar read by construction.
            @allowscalar begin
                @test column_fractional_z_index(2000.0, 1.0, 1.0, grid) == 3   # below surface
                @test column_fractional_z_index(2500.0, 1.0, 1.0, grid) == 3   # at surface
                @test column_fractional_z_index(2800.0, 1.0, 1.0, grid) == 3   # surface → first above-ground
                # Above the first above-ground level, normal interpolation is unchanged.
                @test column_fractional_z_index(3500.0, 1.0, 1.0, grid) ≈ 3.5
            end

            # No clip (surface below the whole column) ⇒ first above-ground level is 1, behavior unchanged.
            grid0, _, _, _ = make_plg(arch)
            @allowscalar @test column_fractional_z_index(0.0, 1.0, 1.0, grid0) == 1
        end

        @testset "rnodes / znodes on the grid return the column-mean Vector" begin
            grid, _, _, _ = make_plg(arch)
            Nz = grid.Nz

            z_grid = znodes(grid, Center())
            @test z_grid isa Vector{Float64}
            @test length(z_grid) == Nz

            # All three znodes/rnodes signatures should agree.
            @test znodes(grid, Center()) == znodes(grid, Center(), Center(), Center())
            @test znodes(grid, Center()) == znodes(grid, nothing, nothing, Center())
        end

        @testset "znodes(::Field) on a horizontally-resolved grid → 3-D Field" begin
            grid, _, _, _ = make_plg(arch)
            f = CenterField(grid)
            z_field = znodes(f)
            @test z_field isa Field
            @test size(z_field) == size(f)

            # Per-cell heights match `rnode(i, j, k, grid, ...)`.
            zᶜ = Array(interior(z_field))
            # `rnode` reads the geopotential field itself, so it is a host-side read of device
            # memory -- unavoidable for a per-cell accessor, hence @allowscalar.
            @allowscalar for i in 1:grid.Nx, j in 1:grid.Ny, k in 1:grid.Nz
                @test zᶜ[i, j, k] ≈
                    Oceananigans.Grids.rnode(i, j, k, grid, Center(), Center(), Center())
            end

            # Column-mean over horizontals matches the grid-level znodes.
            # `znodes` is already on the host, so the reduction has to come back too.
            z_column_mean = Array(interior(Field(mean(z_field, dims=(1, 2)))))
            @test vec(z_column_mean) ≈ znodes(grid, Center())
        end

        @testset "znodes(::Field) on horizontally-absent locations → Vector" begin
            # Case A: Flat-Flat topology (e.g. ERA5 Column region).
            col_grid, _, _, _ = make_plg(arch; topology=(Flat, Flat, Bounded), Nx=1, Ny=1)
            f_col = CenterField(col_grid)
            z_col = znodes(f_col)
            @test z_col isa Vector{Float64}
            @test z_col == znodes(col_grid, Center())

            # Case B: Reduced field with (Nothing, Nothing, Center) location.
            grid, _, _, _ = make_plg(arch)
            f = CenterField(grid)
            set!(f, rand(size(f)...))
            fbar = compute!(Field(mean(f, dims=(1, 2))))
            @test instantiated_location(fbar) === (nothing, nothing, Center())
            z_red = znodes(fbar)
            @test z_red isa Vector{Float64}
            @test z_red ≈ znodes(grid, Center())
        end

        @testset "znodes(::FieldTimeSeries) follows the same dispatch" begin
            grid, _, _, _ = make_plg(arch)
            fts = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 1.0, 2.0])
            z = znodes(fts)
            @test z isa Field
            @test size(z) == (grid.Nx, grid.Ny, grid.Nz)

            col_grid, _, _, _ = make_plg(arch; topology=(Flat, Flat, Bounded), Nx=1, Ny=1)
            fts_col = FieldTimeSeries{Center, Center, Center}(col_grid, [0.0, 1.0])
            @test znodes(fts_col) isa Vector{Float64}
        end

        @testset "TimeSeriesInterpolation-backed Φ ignores halo zeros" begin
            # Regression for PR #241 review: `parent(fts)` includes halo cells
            # filled with zeros, so `extrema` and `mean` over it were dominated
            # by the halos. We must read `interior(fts)` instead.
            Nx, Ny, Nz = 4, 4, 4
            Φ_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz),
                                        longitude=(0, 1), latitude=(0, 1), z=(0, 1))
            Φ_fts = FieldTimeSeries{Center, Center, Center}(Φ_grid, [0.0, 1.0])
            set!(Φ_fts[1], [1000.0 * k * g for i in 1:Nx, j in 1:Ny, k in 1:Nz])  # heights {1, 2, 3, 4} km
            set!(Φ_fts[2], [5000.0 * k * g for i in 1:Nx, j in 1:Ny, k in 1:Nz])  # heights {5, 10, 15, 20} km

            tsi  = TimeSeriesInterpolation(Φ_fts, Φ_fts.grid; clock = Clock(time = 0.0))
            plvd = PressureLevelVerticalDiscretization(tsi; gravitational_acceleration = g)
            grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz),
                                        longitude=(0, 1), latitude=(0, 1), z=plvd)
            # Time-mean column-mean: each k averages [k*1000, k*5000]
            @test znodes(grid, Center()) ≈ [3000.0, 6000.0, 9000.0, 12000.0]
            # Lz = max - min = 20*1000 - 1*1000 = 19000.
            @test grid.Lz ≈ 19000.0
        end

        @testset "TimeSeriesInterpolation-backed Φ heights follow the clock" begin
            # The whole point of the FTS-backed vertical: `rnode` must return
            # different per-cell heights as the shared clock advances.
            Nx, Ny, Nz = 2, 2, 3
            Φ_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz),
                                        longitude=(0, 1), latitude=(0, 1), z=(0, 1))
            Φ_fts = FieldTimeSeries{Center, Center, Center}(Φ_grid, [0.0, 10.0])
            set!(Φ_fts[1], [1000.0 * k * g for i in 1:Nx, j in 1:Ny, k in 1:Nz])  # t=0:  heights {1, 2, 3} km
            set!(Φ_fts[2], [2000.0 * k * g for i in 1:Nx, j in 1:Ny, k in 1:Nz])  # t=10: heights {2, 4, 6} km

            clock = Clock(time = 0.0)
            tsi   = TimeSeriesInterpolation(Φ_fts, Φ_fts.grid; clock)
            plvd  = PressureLevelVerticalDiscretization(tsi; gravitational_acceleration = g)
            grid  = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz),
                                        longitude=(0, 1), latitude=(0, 1), z=plvd)

            rnode = Oceananigans.Grids.rnode
            ℓ = (Center(), Center(), Center())

            @allowscalar begin
                @test rnode(1, 1, 2, grid, ℓ...) ≈ 2000.0    # k=2 at t=0 → 2 km
                clock.time = 10.0
                @test rnode(1, 1, 2, grid, ℓ...) ≈ 4000.0    # same grid, later snapshot → 4 km
                clock.time = 5.0
                @test rnode(1, 1, 2, grid, ℓ...) ≈ 3000.0    # linear-in-time between snapshots
            end
        end
    end
end
