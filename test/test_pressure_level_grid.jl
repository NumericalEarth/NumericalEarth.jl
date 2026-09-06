include("runtests_setup.jl")

using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: Flat, Bounded, topology
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Statistics

using NumericalEarth.Grids: PressureLevelGrid, PressureLevelVerticalDiscretization,
                            column_fractional_z_index
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters, R_d, R_v
using NumericalEarth.DataWrangling.ERA5: reconstruct_near_surface_snapshot!,
                                         ERA5_gravitational_acceleration

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

    @testset "column_fractional_z_index snaps to the first above-ground level" begin
        Nx, Ny, Nz = 2, 2, 5
        Φ_grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz),
                                       longitude=(0, 1), latitude=(0, 1), z=(0, 1))
        Φ = CenterField(Φ_grid)
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            interior(Φ)[i, j, k] = 1000.0 * k * g   # level heights 1..5 km
        end
        Φ_sfc_grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, 1),
                                           longitude=(0, 1), latitude=(0, 1), z=(0, 1))
        Φ_sfc = CenterField(Φ_sfc_grid)
        interior(Φ_sfc) .= 2500.0 * g            # surface at 2.5 km ⇒ clips k=1,2; first above-ground = k=3
        plvd = PressureLevelVerticalDiscretization(Φ; gravitational_acceleration=g,
                                                   surface_geopotential=Φ_sfc)
        grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz), longitude=(0, 1), latitude=(0, 1),
                                     z=plvd, topology=(Bounded, Bounded, Bounded))

        # Clipped column heights (m): [2500, 2500, 3000, 4000, 5000]. Levels 1, 2 still hold the raw
        # sub-surface data, so a target at/below the surface — or between it and the first above-ground
        # level (k=3) — must snap to k=3, never extrapolate into the clipped plateau [1, 3).
        @test column_fractional_z_index(2000.0, 1.0, 1.0, grid) == 3   # below surface
        @test column_fractional_z_index(2500.0, 1.0, 1.0, grid) == 3   # at surface
        @test column_fractional_z_index(2800.0, 1.0, 1.0, grid) == 3   # surface → first above-ground
        # Above the first above-ground level, normal interpolation is unchanged.
        @test column_fractional_z_index(3500.0, 1.0, 1.0, grid) ≈ 3.5

        # No clip (surface below the whole column) ⇒ first above-ground level is 1, behavior unchanged.
        grid0, _, _, _ = make_plg()
        @test column_fractional_z_index(0.0, 1.0, 1.0, grid0) == 1
    end

    @testset "near-surface reconstruction anchors low pressure-level columns on $(arch)" for arch in test_architectures
        FT = Float32
        Nx, Ny, Nz = 2, 1, 5
        grid = RectilinearGrid(arch, FT; size = (Nx, Ny, Nz),
                               x = (0, 1), y = (0, 1), z = (0, 1000),
                               topology = (Bounded, Bounded, Bounded))
        surface_grid = RectilinearGrid(arch, FT; size = (Nx, Ny),
                                       x = (0, 1), y = (0, 1),
                                       topology = (Bounded, Bounded, Flat))
        volume_field(value) = set!(CenterField(grid), FT(value))
        surface_field(value) = set!(Field{Center, Center, Nothing}(surface_grid), FT(value))

        source_geopotential = CenterField(grid)
        set!(source_geopotential, (x, y, z) -> FT(ERA5_gravitational_acceleration) * z)
        geopotential = CenterField(grid)
        pressure = CenterField(grid)
        source_state = (temperature = volume_field(270),
                        eastward_velocity = volume_field(20),
                        northward_velocity = volume_field(10),
                        specific_humidity = volume_field(0.003),
                        cloud_liquid = volume_field(0.001),
                        rain = volume_field(0.002),
                        cloud_ice = volume_field(0.003),
                        snow = volume_field(0.004))
        state = (temperature = volume_field(0),
                 eastward_velocity = volume_field(0),
                 northward_velocity = volume_field(0),
                 specific_humidity = volume_field(0),
                 cloud_liquid = volume_field(0),
                 rain = volume_field(0),
                 cloud_ice = volume_field(0),
                 snow = volume_field(0))

        surface_geopotential = Field{Center, Center, Nothing}(surface_grid)
        set!(surface_geopotential,
             (x, y) -> ifelse(x < FT(1//2), 0,
                              FT(500 * ERA5_gravitational_acceleration)))
        surface_pressure = Field{Center, Center, Nothing}(surface_grid)
        set!(surface_pressure, (x, y) -> ifelse(x < FT(1//2), FT(102000), FT(94000)))
        surface_state = (temperature = surface_field(290),
                         eastward_velocity = surface_field(5),
                         northward_velocity = surface_field(-3),
                         specific_humidity = surface_field(0.01),
                         pressure = surface_pressure)
        pressure_level_values = FT[100000, 97500, 95000, 92500, 90000]
        pressure_levels = on_architecture(arch, pressure_level_values)
        thermodynamics_parameters = AtmosphereThermodynamicsParameters(FT)
        reference_height = FT(10)

        reconstruct_near_surface_snapshot!(grid, geopotential, state, pressure,
                                           source_geopotential, source_state,
                                           surface_geopotential, surface_state,
                                           pressure_levels, thermodynamics_parameters,
                                           reference_height)

        qᵛ = FT(0.01)
        Rᵐ = R_d(thermodynamics_parameters) * (1 - qᵛ) +
             R_v(thermodynamics_parameters) * qᵛ
        pʳ_sea = FT(102000) * exp(-FT(ERA5_gravitational_acceleration) * reference_height /
                                  (Rᵐ * FT(290)))
        pʳ_land = FT(94000) * exp(-FT(ERA5_gravitational_acceleration) * reference_height /
                                  (Rᵐ * FT(290)))

        p = Array(interior(pressure))
        Φ = Array(interior(geopotential))
        T = Array(interior(state.temperature))
        u = Array(interior(state.eastward_velocity))
        v = Array(interior(state.northward_velocity))
        qᵛ = Array(interior(state.specific_humidity))
        qᶜˡ = Array(interior(state.cloud_liquid))
        qʳ = Array(interior(state.rain))
        qᶜⁱ = Array(interior(state.cloud_ice))
        qˢ = Array(interior(state.snow))

        bottom = 1
        second = bottom + 1
        third = second + 1
        top = size(grid, 3)

        # Sea-level column: the first slot becomes the 10 m anchor and 1000 hPa shifts to k=2.
        @test p[1, 1, bottom] ≈ pʳ_sea
        @test Φ[1, 1, bottom] ≈ FT(ERA5_gravitational_acceleration) * reference_height
        @test T[1, 1, bottom] == FT(290)
        @test u[1, 1, bottom] == FT(5)
        @test v[1, 1, bottom] == FT(-3)
        @test qᵛ[1, 1, bottom] == FT(0.01)
        @test iszero(qᶜˡ[1, 1, bottom])
        @test iszero(qʳ[1, 1, bottom])
        @test iszero(qᶜⁱ[1, 1, bottom])
        @test iszero(qˢ[1, 1, bottom])
        @test p[1, 1, second] == pressure_level_values[bottom]
        @test Φ[1, 1, second] ≈ FT(ERA5_gravitational_acceleration) * FT(100)
        @test T[1, 1, second] == FT(270)

        # Elevated column: the anchor plus 1000, 975, and 950 hPa are reconstructed;
        # shifted 925 hPa remains the first reanalysis level above it.
        fourth = third + 1
        @test all(p[2, 1, bottom:fourth] .≈ pʳ_land)
        @test all(Φ[2, 1, bottom:fourth] .≈
                  FT(ERA5_gravitational_acceleration) * FT(510))
        @test all(T[2, 1, bottom:fourth] .== FT(290))
        @test all(iszero, qᶜˡ[2, 1, bottom:fourth])
        @test p[2, 1, top] == pressure_level_values[top - 1]
        @test Φ[2, 1, top] ≈ FT(ERA5_gravitational_acceleration) * FT(700)
        @test T[2, 1, top] == FT(270)
        @test all(diff(vec(p[1, 1, :])) .< 0)
        @test all(diff(vec(Φ[1, 1, :])) .> 0)
        @test all(diff(vec(p[2, 1, :])) .<= 0)
        @test all(diff(vec(Φ[2, 1, :])) .>= 0)

        if arch isa CPU
            reconstructed_vertical = PressureLevelVerticalDiscretization(geopotential;
                gravitational_acceleration = ERA5_gravitational_acceleration,
                surface_geopotential)
            reconstructed_grid = LatitudeLongitudeGrid(arch, FT;
                size = (Nx, Ny, Nz), longitude = (0, 1), latitude = (0, 1),
                z = reconstructed_vertical, topology = (Bounded, Bounded, Bounded))
            # A query just above the repeated 510 m surface plateau must interpolate from its last
            # reconstructed slot toward the retained 700 m / 925 hPa level, not snap to k=1.
            @test column_fractional_z_index(FT(520), FT(2), FT(1), reconstructed_grid) ≈
                  FT(4 + 10 / 190)
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

    @testset "TimeSeriesInterpolation-backed Φ heights follow the clock" begin
        # The whole point of the FTS-backed vertical: `rnode` must return
        # different per-cell heights as the shared clock advances.
        Nx, Ny, Nz = 2, 2, 3
        Φ_grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz),
                                       longitude=(0, 1), latitude=(0, 1), z=(0, 1))
        Φ_fts = FieldTimeSeries{Center, Center, Center}(Φ_grid, [0.0, 10.0])
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            Φ_fts[1][i, j, k] = 1000.0 * k * g     # t=0:  heights {1, 2, 3} km
            Φ_fts[2][i, j, k] = 2000.0 * k * g     # t=10: heights {2, 4, 6} km
        end

        clock = Clock(time = 0.0)
        tsi   = TimeSeriesInterpolation(Φ_fts, Φ_fts.grid; clock)
        plvd  = PressureLevelVerticalDiscretization(tsi; gravitational_acceleration = g)
        grid  = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz),
                                      longitude=(0, 1), latitude=(0, 1), z=plvd)

        rnode = Oceananigans.Grids.rnode
        ℓ = (Center(), Center(), Center())

        @test rnode(1, 1, 2, grid, ℓ...) ≈ 2000.0    # k=2 at t=0 → 2 km
        clock.time = 10.0
        @test rnode(1, 1, 2, grid, ℓ...) ≈ 4000.0    # same grid, later snapshot → 4 km
        clock.time = 5.0
        @test rnode(1, 1, 2, grid, ℓ...) ≈ 3000.0    # linear-in-time between snapshots
    end
end
