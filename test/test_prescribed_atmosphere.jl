include("runtests_setup.jl")

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: parent
using Oceananigans.Grids: halo_size

function assert_tripolar_velocity_zipper(field, grid)
    set!(field, 1)
    fill_halo_regions!(field)

    Hx, Hy, Hz = halo_size(grid)
    data = parent(field)

    i = Hx + 1:size(data, 1) - Hx
    j_interior = Hy + size(grid, 2)
    j_halo = j_interior + 1
    k = 1

    north_halo = Array(@view data[i, j_halo, k])
    @test all(north_halo .== -1)
end

@testset "PrescribedAtmosphere tripolar velocity zipper sign" begin
    grid = TripolarGrid(CPU(); size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))
    atmosphere = PrescribedAtmosphere(grid, [0.0])

    for field in (atmosphere.velocities.u[1], atmosphere.velocities.v[1])
        assert_tripolar_velocity_zipper(field, grid)
    end
end

@testset "PrescribedAtmosphere set!" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch, size = 1, z = (-1, 0), topology = (Flat, Flat, Bounded))
        atmosphere = PrescribedAtmosphere(grid, [0.0])

        set!(atmosphere; u = 3, T = 305, q = 0.004, p = 101_325)
        @test only(Array(interior(atmosphere.velocities.u[1])))   == 3
        @test only(Array(interior(atmosphere.temperature[1])))    == 305
        @test only(Array(interior(atmosphere.specific_humidity[1]))) == 0.004
        @test only(Array(interior(atmosphere.pressure[1])))       == 101_325

        # An omitted keyword leaves that field untouched.
        set!(atmosphere; T = 300)
        @test only(Array(interior(atmosphere.temperature[1])))    == 300
        @test only(Array(interior(atmosphere.velocities.u[1])))   == 3
    end
end

@testset "Regridded prescribed atmosphere tripolar velocity zipper sign" begin
    # The exchanger builds its velocity state on the (tripolar) exchange grid with
    # north-fold BCs, independent of the atmosphere's source grid or data, so a plain
    # PrescribedAtmosphere on a lat-lon grid exercises the same path without any download.
    # Dataset-backed atmosphere constructors are exercised by the download tests.
    atmosphere_grid = LatitudeLongitudeGrid(CPU(); size = (36, 18, 1),
                                            longitude = (0, 360), latitude = (-80, 80),
                                            z = (-1, 0), halo = (3, 3, 3))
    atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0])

    exchange_grid = TripolarGrid(CPU(); size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))
    exchanger = NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere, exchange_grid)

    for field in (exchanger.state.u, exchanger.state.v)
        assert_tripolar_velocity_zipper(field, exchange_grid)
    end
end

@testset "Regional atmosphere fractional indices stay readable" begin
    # `initialize!` fills fractional indices over the exchange grid's halo as well as its
    # interior, so halo columns index into the atmosphere's own halo — which is where its
    # boundary conditions live. Interpolation reads `⌊f⌋` and `⌊f⌋ + 1`, so an index is
    # readable while it lies within `1 - H` and `N + H`.
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (2, 1, 1),
                                     longitude = (0, 20), latitude = (0, 10),
                                     z = (-1, 0),
                                     topology = (Bounded, Bounded, Bounded))

        InterfaceComputations = NumericalEarth.EarthSystemModels.InterfaceComputations
        atmosphere = PrescribedAtmosphere(grid, [0.0])
        exchanger = InterfaceComputations.ComponentExchanger(atmosphere, grid)
        InterfaceComputations.initialize!(exchanger, grid, atmosphere)

        Nx, Ny, _ = size(atmosphere.grid)
        Hx, Hy, _ = halo_size(atmosphere.grid)

        # The atmosphere shares the exchange grid, so interior cells map onto themselves.
        @test Array(interior(exchanger.regridder.i))[:, 1, 1] ≈ [1, 2]
        @test Array(interior(exchanger.regridder.j))[:, 1, 1] ≈ [1, 1]

        fi = Array(exchanger.regridder.i.data[0:Nx+1, 0:Ny+1, 1:1])
        fj = Array(exchanger.regridder.j.data[0:Nx+1, 0:Ny+1, 1:1])
        @test all(f -> 1 - Hx <= f < Nx + Hx, fi)
        @test all(f -> 1 - Hy <= f < Ny + Hy, fj)

        # Halo columns are not clamped onto the interior, so the read reaches the
        # atmosphere's halo and hence its boundary conditions — in latitude too, where the
        # atmosphere is a single cell deep.
        @test fi[end, 1, 1] > Nx
        @test fj[1, end, 1] > Ny

        # Interior columns are never clamped, so an exchange grid the atmosphere does not
        # cover keeps its out-of-range index rather than silently reading the nearest cell.
        wide_grid = LatitudeLongitudeGrid(arch; size = (4, 1, 1),
                                          longitude = (-60, 80), latitude = (0, 10),
                                          z = (-1, 0),
                                          topology = (Bounded, Bounded, Bounded))

        wide_exchanger = InterfaceComputations.ComponentExchanger(atmosphere, wide_grid)
        InterfaceComputations.initialize!(wide_exchanger, wide_grid, atmosphere)
        wide_fi = Array(interior(wide_exchanger.regridder.i))[:, 1, 1]
        @test wide_fi[1] < 1 - Hx
        @test wide_fi[end] > Nx + Hx
    end
end

@testset "Prescribed atmosphere tracers at any horizontal location" begin
    for arch in test_architectures
        atmosphere_grid = LatitudeLongitudeGrid(arch; size = (8, 8, 1), longitude = (0, 90), latitude = (-40, 40), z = (0, 1))
        grid = LatitudeLongitudeGrid(arch; size = (10, 12, 1), longitude = (10, 80), latitude = (-30, 30), z = (-100, 0), halo = (6, 6, 3))

        # Tracers have their own time axis, twice as long as the atmosphere's
        function linear_tracer(LX, LY, f)
            c = FieldTimeSeries{LX, LY, Nothing}(atmosphere_grid, [0, 7200.0])
            ℓλ = LX === Nothing ? Center() : LX()
            ℓφ = LY === Nothing ? Center() : LY()
            λ = λnodes(atmosphere_grid, ℓλ, ℓφ, Center())
            φ = φnodes(atmosphere_grid, ℓλ, ℓφ, Center())
            c₁ = LX === Nothing ? f.(φ') : LY === Nothing ? f.(λ) : f.(λ, φ')
            for n in 1:2
                set!(c[n], n .* c₁)
            end
            return c
        end

        tracers = (cᶠᶜ = linear_tracer(Face, Center, (λ, φ) -> λ + 2φ),
                   cᶠᶠ = linear_tracer(Face, Face, (λ, φ) -> λ - φ),
                   cˣ  = linear_tracer(Face, Nothing, λ -> 3λ),
                   cʸ  = linear_tracer(Nothing, Center, φ -> φ + 50),
                   cʸᶠ = linear_tracer(Nothing, Face, φ -> 2φ))

        atmosphere = PrescribedAtmosphere(atmosphere_grid, [0, 3600.0]; tracers)
        ocean = ocean_simulation(grid, closure = nothing)
        model = OceanOnlyModel(ocean; atmosphere, radiation = nothing)
        time_step!(model, 60)

        # Linear fields are interpolated exactly
        state = model.interfaces.exchanger.atmosphere.state
        λ = Array(λnodes(grid, Center(), Center(), Center()))
        φ = Array(φnodes(grid, Center(), Center(), Center()))
        factor = 1 + 60 / 7200
        @test Array(interior(state.cᶠᶜ, :, :, 1)) ≈ factor .* (λ .+ 2φ')
        @test Array(interior(state.cᶠᶠ, :, :, 1)) ≈ factor .* (λ .- φ')
        @test Array(interior(state.cˣ,  :, :, 1)) ≈ factor .* (3λ .+ 0φ')
        @test Array(interior(state.cʸ,  :, :, 1)) ≈ factor .* (0λ .+ φ' .+ 50)
        @test Array(interior(state.cʸᶠ, :, :, 1)) ≈ factor .* (0λ .+ 2φ')
    end
end
