include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!, time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, DryLayerHumidity, StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity, ConstantTortuosity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, CanopyAirSpaceDiagnostics, CanopyAirState,
    DiagnosticCanopyAir, PrognosticCanopyAir, advance_canopy_air, step_mean_canopy_air,
    AtmosphericThermodynamics
using NumericalEarth.Atmospheres: PrescribedAtmosphere, AtmosphereThermodynamicsParameters
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties

prognostic_test_cas(FT; kw...) = CanopyAirSpace(FT;
    soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                    dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
        vapor_exchange  = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                      molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.05, porosity = 0.4),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05), kw...)

function prognostic_test_model(arch, cas; shortwave = 600.0, longwave = 350.0,
                               wind = 3.0, Tair = 300.0, qair = 0.008,
                               Tland = 298.0, water = 45.0)
    FT = Float64
    grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                 z = (-1, 0), topology = (Flat, Flat, Bounded))
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
    fill!(parent(atmosphere.temperature), Tair)
    fill!(parent(atmosphere.specific_humidity), qair)
    fill!(parent(atmosphere.velocities.u), wind)
    fill!(parent(atmosphere.pressure), 101325.0)
    land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
    set!(land; T = Tland)
    fill!(parent(land.water_storage), water)
    radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                    land_surface = SurfaceRadiationProperties(0.2, 0.95))
    fill!(parent(radiation.downwelling_shortwave), shortwave)
    fill!(parent(radiation.downwelling_longwave), longwave)
    update_state!(radiation)
    model = AtmosphereLandModel(atmosphere, land; radiation,
                atmosphere_land_interface_temperature = cas,
                atmosphere_land_interface_specific_humidity = cas)
    update_state!(model.land)
    update_state!(model)
    return model
end

@inline value1(f) = Array(interior(f))[1, 1, 1]

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

    # Step-mean: between the endpoints, → equilibrium for Δt ≫ τ, → the initial
    # value for Δt ≪ τ (the ⟨x⟩ = x₀ + O(Δt/τ) limit, accurate through expm1).
    x̄ = step_mean_canopy_air(x, x_eq, Σg, C, 300.0)
    @test min(x, x_eq) ≤ x̄ ≤ max(x, x_eq)
    @test step_mean_canopy_air(x, x_eq, Σg, C, 1e6) ≈ x_eq atol = (x_eq - x) * τ / 1e6 * 1.01
    @test step_mean_canopy_air(x, x_eq, Σg, C, 1e-4) ≈ x atol = (x_eq - x) * 1e-4 / τ
end

@testset "Prognostic canopy air space" begin
    for arch in test_architectures
        # --- initialization: before any time step the node lands on the equilibrium
        # of the first solve and the state fields are populated.
        mp = prognostic_test_model(arch, prognostic_test_cas(Float64; storage = PrognosticCanopyAir()))
        Ts = mp.interfaces.atmosphere_land_interface.temperature
        @test Ts isa CanopyAirSpaceDiagnostics
        @test Ts.state isa CanopyAirState
        T₀ = value1(Ts.state.temperature)
        q₀ = value1(Ts.state.specific_humidity)
        @test 285 < T₀ < 320
        @test 0 < q₀ < 0.05
        @test value1(Ts.interface) == T₀

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
        @test value1(fp.latent_heat) ≈ value1(fd.latent_heat) atol = 2
        @test value1(fp.sensible_heat) ≈ value1(fd.sensible_heat) atol = 2

        # --- exact step ledger: flux to the atmosphere = Kirchhoff supply − storage
        # tendency, with capacities C_T = ρ cᵖ h_c and C_q = ρ h_c.
        T⁻ = value1(Ts.state.temperature)
        q⁻ = value1(Ts.state.specific_humidity)
        Δt = 300.0
        time_step!(mp, Δt)
        T⁺ = value1(Ts.state.temperature)
        q⁺ = value1(Ts.state.specific_humidity)
        ℂ  = AtmosphereThermodynamicsParameters(Float64)
        ρ  = AtmosphericThermodynamics.air_density(ℂ, 300.0, 101325.0, 0.008)
        cᵖ = AtmosphericThermodynamics.cp_m(ℂ, 0.008)
        ℒ  = AtmosphericThermodynamics.latent_heat_vapor(ℂ, 300.0)
        h_c = 10.0
        S_T = ρ * cᵖ * h_c * (T⁺ - T⁻) / Δt
        S_q = ℒ * ρ * h_c * (q⁺ - q⁻) / Δt
        Hᵛ  = value1(Ts.canopy_sensible_heat)
        Hᵍ  = value1(Ts.soil_sensible_heat)
        LEᵛ = value1(Ts.canopy_latent_heat)
        LEᵍ = value1(Ts.soil_latent_heat)
        @test Hᵛ + Hᵍ - value1(fp.sensible_heat) ≈ S_T atol = 1e-6
        @test LEᵛ + LEᵍ - value1(fp.latent_heat) ≈ S_q atol = 1e-6

        # --- the node stays inside the hull of its sources.
        θᵃᵗ = 300.0
        @test min(value1(Ts.canopy), value1(Ts.soil_skin), θᵃᵗ) - 1 ≤ value1(Ts.interface) ≤
              max(value1(Ts.canopy), value1(Ts.soil_skin), θᵃᵗ) + 1
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
            LE = value1(fp.latent_heat)
            @test isfinite(LE)
            @test abs(LE) < 2000
        end
        @test isfinite(value1(Ts.state.temperature))
        @test isfinite(value1(Ts.state.specific_humidity))
        @test 0 < value1(Ts.state.specific_humidity) < 0.06
    end
end
