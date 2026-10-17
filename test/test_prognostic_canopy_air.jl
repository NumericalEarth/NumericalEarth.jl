include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, CanopyAirSpaceDiagnostics, CanopyAirState,
    DiagnosticCanopyAir, PrognosticCanopyAir, advance_canopy_air, node_memory,
    AtmosphericThermodynamics
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters

prognostic_test_cas(FT; kw...) = CanopyAirSpace(FT;
    soil = bare_soil_humidity(FT),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05), kw...)

prognostic_test_model(arch, cas; shortwave = 600.0, longwave = 350.0,
                      wind = 3.0, Tair = 300.0, qair = 0.008,
                      Tland = 298.0, water = 45.0) =
    coupled_land_model(arch; shortwave, longwave, wind, Tair, qair, Tland, water,
                       atmosphere_land_interface_temperature = cas,
                       atmosphere_land_interface_specific_humidity = cas)

@testset "CanopyAirSpace defaults to a diagnostic node" begin
    cas = prognostic_test_cas(Float64)
    @test cas.storage isa DiagnosticCanopyAir

    # A diagnostic node is massless, so it carries no state across time steps and the
    # flux path stays the one the massless closure has always taken.
    model = prognostic_test_model(CPU(), cas)
    @test model.interfaces.atmosphere_land_interface.temperature.state === nothing
end

@testset "advance_canopy_air" begin
    x, x_eq, Σg, C = 300.0, 305.0, 50.0, 1e4
    τ = C / Σg

    # Exact exponential solution of the linearized node ODE.
    for Δt in (10.0, 300.0, 3600.0)
        @test advance_canopy_air(x, x_eq, Σg, C, Δt) ≈ x_eq + (x - x_eq) * exp(-Δt / τ)
    end

    # Δt = 0 (first update_state!) and C = 0 (massless) land on the equilibrium.
    @test advance_canopy_air(x, x_eq, Σg, C, 0.0) == x_eq
    @test advance_canopy_air(x, x_eq, Σg, 0.0, 300.0) == x_eq

    # Hull-bounded: the update is a convex blend of the old value and the equilibrium.
    for (a, b, g, c, dt) in ((280.0, 310.0, 1.0, 1e5, 60.0), (310.0, 280.0, 200.0, 1e2, 600.0))
        x⁺ = advance_canopy_air(a, b, g, c, dt)
        @test min(a, b) ≤ x⁺ ≤ max(a, b)
    end

    # Node memory, the start-of-step weight in the step mean: → 1 for Δt ≪ τ, → 0 for
    # Δt ≫ τ, and 0 for a massless node or before the first step.
    @test 0 < node_memory(Σg, C, 300.0) < 1
    @test node_memory(Σg, C, 1e6) ≈ τ / 1e6 rtol = 1e-3
    @test node_memory(Σg, C, 1e-4) ≈ 1 atol = 1e-4 / τ
    @test node_memory(Σg, 0.0, 300.0) == 0
    @test node_memory(Σg, C, 0.0) == 0
end

@testset "Prognostic canopy air space" begin
    for arch in test_architectures
        # --- initialization: before any time step the node lands on the equilibrium
        # of the first solve and the state fields are populated.
        mp = prognostic_test_model(arch, prognostic_test_cas(Float64; storage = PrognosticCanopyAir()))
        Ts = mp.interfaces.atmosphere_land_interface.temperature
        @test Ts isa CanopyAirSpaceDiagnostics
        @test Ts.state isa CanopyAirState
        T₀ = scalar(Ts.state.temperature)
        q₀ = scalar(Ts.state.specific_humidity)
        @test 285 < T₀ < 320
        @test 0 < q₀ < 0.05
        @test scalar(Ts.interface) == T₀

        # --- daytime equivalence: at windy daytime the node's relaxation time is much
        # shorter than the step, so the prognostic column matches the diagnostic one.
        md = prognostic_test_model(arch, prognostic_test_cas(Float64; storage = DiagnosticCanopyAir()))
        @test md.interfaces.atmosphere_land_interface.temperature.state === nothing
        for _ in 1:24
            time_step!(md, 300.0)
            time_step!(mp, 300.0)
        end
        fd = md.interfaces.atmosphere_land_interface.fluxes
        fp = mp.interfaces.atmosphere_land_interface.fluxes
        @test scalar(fp.latent_heat) ≈ scalar(fd.latent_heat) atol = 2
        @test scalar(fp.sensible_heat) ≈ scalar(fd.sensible_heat) atol = 2

        # --- exact step ledger: flux to the atmosphere = Kirchhoff supply − storage
        # tendency, with capacities Cᵀ = ρ cᵖ hᶜ and Cᵛ = ρ hᶜ. The shares are evaluated
        # at the same node the skins were solved against.
        T⁻ = scalar(Ts.state.temperature)
        q⁻ = scalar(Ts.state.specific_humidity)
        Δt = 300.0
        time_step!(mp, Δt)
        T⁺ = scalar(Ts.state.temperature)
        q⁺ = scalar(Ts.state.specific_humidity)
        ℂ  = AtmosphereThermodynamicsParameters(Float64)
        ρ  = AtmosphericThermodynamics.air_density(ℂ, 300.0, 101325.0, 0.008)
        cᵖ = AtmosphericThermodynamics.cp_m(ℂ, 0.008)
        ℒ  = AtmosphericThermodynamics.latent_heat_vapor(ℂ, 300.0)
        hᶜ = 10.0
        Sᵀ = ρ * cᵖ * hᶜ * (T⁺ - T⁻) / Δt
        Sᵛ = ℒ * ρ * hᶜ * (q⁺ - q⁻) / Δt
        Hˡᵉᵃᶠ  = scalar(Ts.canopy_sensible_heat)
        Hᵍ  = scalar(Ts.soil_sensible_heat)
        LEˡᵉᵃᶠ = scalar(Ts.canopy_latent_heat)
        LEᵍ = scalar(Ts.soil_latent_heat)
        @test Hˡᵉᵃᶠ + Hᵍ - scalar(fp.sensible_heat) ≈ Sᵀ atol = 1e-6
        @test LEˡᵉᵃᶠ + LEᵍ - scalar(fp.latent_heat) ≈ Sᵛ atol = 1e-6

        # --- the skins balance against the exported node: leaf Rₙ = H + LE at the stored
        # leaf temperature and node.
        ℂ  = AtmosphereThermodynamicsParameters(Float64)
        Tˡ = scalar(Ts.canopy); Tᵍ = scalar(Ts.soil_skin)
        σ  = 5.670374419e-8; LAI = 4.0
        εˡ = 0.98 * (1 - exp(-LAI)); εᵍ = 0.96
        LWꜜᵍ = (1 - εˡ) * 350.0 + εˡ * σ * Tˡ^4
        LWꜛᵍ = εᵍ * σ * Tᵍ^4 + (1 - εᵍ) * LWꜜᵍ
        Rˡ = (1 - 0.15) * (1 - exp(-0.5 * LAI)) * 600.0 + εˡ * (350.0 + LWꜛᵍ) - 2 * εˡ * σ * Tˡ^4
        @test Rˡ ≈ Hˡᵉᵃᶠ + LEˡᵉᵃᶠ atol = 1e-3

        # --- the node stays inside the hull of its sources.
        θᵃᵗ = 300.0
        @test min(scalar(Ts.canopy), scalar(Ts.soil_skin), θᵃᵗ) - 1 ≤ scalar(Ts.interface) ≤
              max(scalar(Ts.canopy), scalar(Ts.soil_skin), θᵃᵗ) + 1
    end

    # --- calm-dusk boundedness: the configuration where the diagnostic closure has no
    # steady state (hot wet soil under cooler dry air, near-calm wind, dusk radiation).
    # The prognostic node keeps every exit finite, supply-consistent through the
    # ledger, and free of kW-scale artifacts.
    for arch in test_architectures
        cas = prognostic_test_cas(Float64; storage = PrognosticCanopyAir())
        mp = prognostic_test_model(arch, cas; shortwave = 50.0, longwave = 350.0,
                                   wind = 0.2, Tair = 304.0, qair = 0.008,
                                   Tland = 310.0, water = 135.0)
        Ts = mp.interfaces.atmosphere_land_interface.temperature
        fp = mp.interfaces.atmosphere_land_interface.fluxes
        for _ in 1:48
            time_step!(mp, 300.0)
            LE = scalar(fp.latent_heat)
            @test isfinite(LE)
            @test abs(LE) < 2000
        end
        @test isfinite(scalar(Ts.state.temperature))
        @test isfinite(scalar(Ts.state.specific_humidity))
        @test 0 < scalar(Ts.state.specific_humidity) < 0.06
    end
end
