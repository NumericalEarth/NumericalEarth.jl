include("runtests_setup.jl")

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.OutputReaders: Cyclical
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: days
using NumericalEarth.Lands: DryLand
using NumericalEarth.EarthSystemModels: thermodynamics_parameters
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger,
                                                              StateExchanger,
                                                              correct_state!,
                                                              saturation_specific_humidity
using Thermodynamics: Thermodynamics

const InterfaceComputations = NumericalEarth.EarthSystemModels.InterfaceComputations

# Exchange grid and the reference near-surface state (T in K, p in Pa, q in kg kg⁻¹).
function offset_test_setup(arch; FT = Float64)
    grid = LatitudeLongitudeGrid(arch, FT; size = (3, 4, 1),
                                 longitude = (90, 180), latitude = (-40, 40),
                                 z = (-1, 0), halo = (3, 3, 3))

    atmosphere = PrescribedAtmosphere(grid, [zero(FT)])
    radiation  = PrescribedRadiation(grid)
    return grid, atmosphere, radiation
end

function fill_reference_state!(exchanger, ℂ)
    T = exchanger.state.T
    p = exchanger.state.p
    q = exchanger.state.q
    set!(T, (λ, φ) -> 300 - 0.3 * abs(φ) + 0.01 * λ)
    set!(p, (λ, φ) -> 101_325 - 20 * φ)
    fill_halo_regions!(T)
    fill_halo_regions!(p)

    # 70% relative humidity
    Tᵢ = Array(interior(T))
    pᵢ = Array(interior(p))
    qᵢ = 0.7 .* saturation_specific_humidity.(Ref(ℂ), Tᵢ, pᵢ, Ref(Thermodynamics.Liquid()))
    set!(q, qᵢ)
    fill_halo_regions!(q)
    return nothing
end

snapshot(state) = (T = Array(interior(state.T)), q = Array(interior(state.q)), p = Array(interior(state.p)))

relative_humidity(ℂ, T, p, q) = q ./ saturation_specific_humidity.(Ref(ℂ), T, p, Ref(Thermodynamics.Liquid()))

# Apply `AtmosphereTemperatureOffset(ΔT)` and check the update against `expected_δT`
# (a horizontal array or a number) at the current atmosphere clock time.
function check_atmosphere_offset(exchanger, ℂ, grid, expected_δT)
    fill_reference_state!(exchanger, ℂ)
    before = snapshot(exchanger.state)
    correct_state!(exchanger, grid)
    after = snapshot(exchanger.state)

    δT = expected_δT .* ones(size(before.T))
    @test after.T ≈ before.T .+ δT
    @test after.p == before.p
    @test relative_humidity(ℂ, after.T, after.p, after.q) ≈ relative_humidity(ℂ, before.T, before.p, before.q) rtol=1e-6
    return nothing
end

@testset "linear_ramp" begin
    ramp(t) = linear_ramp(t; from = 1, to = -3, start = 10, duration = 20)
    @test ramp(-5) == 1
    @test ramp(10) == 1
    @test ramp(20) ≈ -1
    @test ramp(30) ≈ -3
    @test ramp(1e6) == -3
    @test linear_ramp(5; from = 0, to = 2, start = 5, duration = 0) == 0
    @test linear_ramp(6; from = 0, to = 2, start = 5, duration = 0) == 2

    corrections = temperature_offset_corrections(-2)
    @test corrections.atmosphere isa AtmosphereTemperatureOffset
    @test corrections.radiation isa DownwellingLongwaveOffset
    @test corrections.radiation.sensitivity == 0.7 / 0.133
    @test temperature_offset_corrections(-2; longwave_sensitivity = 4).radiation.sensitivity == 4
end

@testset "AtmosphereTemperatureOffset input forms" begin
    for arch in test_architectures
        grid, atmosphere, _ = offset_test_setup(arch)
        ℂ = thermodynamics_parameters(atmosphere)
        Nx, Ny, _ = size(grid)
        λ = Array(λnodes(grid, Center()))
        φ = Array(φnodes(grid, Center()))

        build(ΔT) = ComponentExchanger(atmosphere, grid; correction = AtmosphereTemperatureOffset(ΔT))

        @testset "Number" begin
            exchanger = build(-1.5)
            @test exchanger.correction.offset === -1.5
            @test exchanger.correction.clock === atmosphere.clock
            check_atmosphere_offset(exchanger, ℂ, grid, -1.5)
        end

        @testset "Zero offset is a no-op" begin
            for ΔT in (0, t -> 0)
                exchanger = build(ΔT)
                fill_reference_state!(exchanger, ℂ)
                before = snapshot(exchanger.state)
                correct_state!(exchanger, grid)
                after = snapshot(exchanger.state)
                @test after.T == before.T
                @test after.q == before.q
            end
        end

        @testset "Function of time" begin
            ΔT(t) = linear_ramp(t; from = 0, to = -4, start = 1days, duration = 2days)
            exchanger = build(ΔT)
            for (t, expected) in ((0, 0), (2days, -2), (5days, -4))
                atmosphere.clock.time = t
                check_atmosphere_offset(exchanger, ℂ, grid, expected)
            end
            atmosphere.clock.time = 0
        end

        @testset "Function of space and time" begin
            ΔT(λ, φ, t) = 0.01 * λ - 0.02 * φ + t / 1days
            exchanger = build(ΔT)
            atmosphere.clock.time = 3days
            expected = [ΔT(λ[i], φ[j], 3days) for i in 1:Nx, j in 1:Ny]
            check_atmosphere_offset(exchanger, ℂ, grid, expected)
            atmosphere.clock.time = 0
        end

        @testset "Static pattern" begin
            pattern = Field{Center, Center, Nothing}(grid)
            set!(pattern, (λ, φ) -> 0.05 * φ)
            expected = [0.05 * φ[j] for i in 1:Nx, j in 1:Ny]
            check_atmosphere_offset(build(pattern), ℂ, grid, expected)

            array = reshape(collect(1.0:Nx*Ny), Nx, Ny) ./ 10
            check_atmosphere_offset(build(array), ℂ, grid, array)

            # A two-argument function is a static pattern set via `set!`
            check_atmosphere_offset(build((λ, φ) -> 0.05 * φ), ℂ, grid, expected)
        end

        @testset "Cyclical FieldTimeSeries on a coarser grid" begin
            coarse_grid = LatitudeLongitudeGrid(arch; size = (4, 3, 1),
                                                longitude = (0, 360), latitude = (-60, 60),
                                                z = (-1, 0), halo = (3, 3, 3))
            times = [0, 1days, 2days]
            fts = FieldTimeSeries{Center, Center, Nothing}(coarse_grid, times; time_indexing = Cyclical())

            # Linear in λ and φ, so bilinear interpolation onto exchange-grid
            # centers inside the coarse cell-center range is exact.
            amplitude = (0.0, -1.0, -3.0)
            for n in 1:3
                set!(fts[n], (λ, φ) -> amplitude[n] + 0.02 * φ + 0.004 * λ)
            end
            fill_halo_regions!(fts)

            exchanger = build(fts)
            spatial = [0.02 * φ[j] + 0.004 * λ[i] for i in 1:Nx, j in 1:Ny]

            # Cyclical period = 3 days: between records, across the wrap
            # (record 3 → record 1), and after one full period.
            for (t, a) in ((0.5days, -0.5), (2.5days, -1.5), (3.25days, -0.25))
                atmosphere.clock.time = t
                check_atmosphere_offset(exchanger, ℂ, grid, a .+ spatial)
            end
            atmosphere.clock.time = 0
        end
    end
end

@testset "Space-time offset on a tripolar exchange grid" begin
    for arch in test_architectures
        atmosphere_grid = LatitudeLongitudeGrid(arch; size = (36, 18, 1),
                                                longitude = (0, 360), latitude = (-80, 80),
                                                z = (-1, 0), halo = (3, 3, 3))
        atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0])
        ℂ = thermodynamics_parameters(atmosphere)

        grid = TripolarGrid(arch; size = (32, 16, 1), z = (-1, 0), halo = (3, 3, 3))
        ΔT(λ, φ, t) = 0.02 * φ + 0.001 * λ
        exchanger = ComponentExchanger(atmosphere, grid; correction = AtmosphereTemperatureOffset(ΔT))

        λ = Array(λnodes(grid, Center(), Center(), Center()))
        φ = Array(φnodes(grid, Center(), Center(), Center()))
        check_atmosphere_offset(exchanger, ℂ, grid, ΔT.(λ, φ, 0))
    end
end

@testset "DownwellingLongwaveOffset" begin
    for arch in test_architectures
        grid, _, radiation = offset_test_setup(arch)
        s = 0.7 / 0.133

        exchanger = ComponentExchanger(radiation, grid; correction = DownwellingLongwaveOffset(-10; sensitivity = s))
        @test exchanger.correction.clock === radiation.clock
        @test exchanger.correction.sensitivity == s

        ℐꜜˡʷ = exchanger.state.ℐꜜˡʷ
        set!(ℐꜜˡʷ, (λ, φ) -> 300 + φ) # 270 to 330 W m⁻²
        before = Array(interior(ℐꜜˡʷ))
        correct_state!(exchanger, grid)
        @test Array(interior(ℐꜜˡʷ)) ≈ before .- 10s

        # Clipping at zero
        set!(ℐꜜˡʷ, 20)
        correct_state!(exchanger, grid)
        @test all(Array(interior(ℐꜜˡʷ)) .== 0)

        # Same input forms as the atmosphere offset, evaluated against the radiation clock
        exchanger = ComponentExchanger(radiation, grid;
                                       correction = DownwellingLongwaveOffset(t -> -t / 1days; sensitivity = 2))
        radiation.clock.time = 2days
        set!(exchanger.state.ℐꜜˡʷ, 300)
        correct_state!(exchanger, grid)
        @test all(Array(interior(exchanger.state.ℐꜜˡʷ)) .≈ 296)
        radiation.clock.time = 0
    end
end

@testset "Radiation correction slot in the state exchanger" begin
    for arch in test_architectures
        grid, atmosphere, radiation = offset_test_setup(arch)

        exchanger = StateExchanger(grid, radiation, atmosphere, nothing, nothing, nothing)
        @test exchanger.radiation.correction === nothing
        @test exchanger.atmosphere.correction === nothing

        corrections = temperature_offset_corrections(-1)
        exchanger = StateExchanger(grid, radiation, atmosphere, nothing, nothing, nothing;
                                   atmosphere_correction = corrections.atmosphere,
                                   radiation_correction = corrections.radiation)
        @test exchanger.radiation.correction isa DownwellingLongwaveOffset
        @test exchanger.atmosphere.correction isa AtmosphereTemperatureOffset
    end
end

@testset "Temperature offset in a coupled AtmosphereLandModel" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))

        atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
        radiation  = PrescribedRadiation(grid)
        land       = SlabLand(grid; hydrology = DryLand(), energy = SlabEnergy(eltype(grid)))

        default_model = AtmosphereLandModel(atmosphere, land; radiation)
        @test default_model.interfaces.exchanger.radiation.correction === nothing

        ΔT = -2
        corrections = temperature_offset_corrections(ΔT)
        model = AtmosphereLandModel(atmosphere, land; radiation,
                                    exchanger_correction = corrections.atmosphere,
                                    radiation_correction = corrections.radiation)

        Tᵃ, pᵃ, qᵃ, ℐᵃ = 290, 101_325, 0.008, 350
        fill!(parent(model.atmosphere.velocities.u), 1)
        fill!(parent(model.atmosphere.velocities.v), 0)
        fill!(parent(model.atmosphere.temperature), Tᵃ)
        fill!(parent(model.atmosphere.pressure), pᵃ)
        fill!(parent(model.atmosphere.specific_humidity), qᵃ)
        set!(model.radiation; downwelling_longwave = ℐᵃ)
        fill!(model.land.temperature, Tᵃ)

        update_state!(model)

        ℂ = thermodynamics_parameters(model.atmosphere)
        phase = Thermodynamics.Liquid()
        state = model.interfaces.exchanger.atmosphere.state
        T = only(Array(interior(state.T)))
        q = only(Array(interior(state.q)))
        @test T ≈ Tᵃ + ΔT
        @test q / saturation_specific_humidity(ℂ, T, pᵃ, phase) ≈
              qᵃ / saturation_specific_humidity(ℂ, Tᵃ, pᵃ, phase) rtol=1e-6

        ℐꜜˡʷ = only(Array(interior(model.interfaces.exchanger.radiation.state.ℐꜜˡʷ)))
        @test ℐꜜˡʷ ≈ ℐᵃ + 0.7 / 0.133 * ΔT
    end
end
