# Unit tests for the NEMO 3.6 TKE vertical-mixing closure
# (experiments/OMIPSimulations/src/NEMOTKE/). These exercise the math-helper
# functions directly with hand-derived NEMO Fortran reference values; no
# Oceananigans simulation, no data downloads.

using Test
using Oceananigans

# Activate the OMIPSimulations project so NEMOTKEVerticalDiffusivity is visible.
using OMIPSimulations
using OMIPSimulations: NEMOTKEVerticalDiffusivity, NEMOTKEParameters
using OMIPSimulations.NEMOTKE: surface_TKE,
                                friction_velocity_squared,
                                natural_length_scale,
                                nemo_eddy_coefficient,
                                viscosity_with_floors,
                                diffusivity_with_floors,
                                stokes_velocity,
                                langmuir_source,
                                wave_decay_length,
                                wave_penetration_source,
                                evd_overwrite,
                                evd_overwrite_momentum

@testset "NEMOTKE" begin

    @testset "Parameters: OMIP-2 ORCA1 defaults" begin
        p = NEMOTKEParameters()
        # Physics coefficients
        @test p.Cᴷ    == 0.1
        @test p.Cᴰ    == 0.7
        @test p.Cᵇ    == 3.75
        @test p.Cᴸ    == 0.15
        @test p.Cᶠ    == 1.0
        @test p.Cˢ    == 0.016
        @test p.κᶜⁿᵛ  == 100.0
        @test p.νᵇ    == 1.2e-4
        @test p.κᵇ    == 1.2e-5
        # Floors
        @test p.minimum_TKE           ≈ sqrt(2) * 1e-6
        @test p.minimum_surface_TKE   == 1e-4
        @test p.minimum_mixing_length == 0.04
        # Formulation flags
        @test p.mixing_length_formulation        == 2
        @test p.wave_penetration_formulation     == 1
        @test p.surface_length_scale_formulation == 1
        # Apply flags
        @test p.apply_langmuir_circulation        == true
        @test p.apply_wave_penetration            == true
        @test p.apply_enhanced_vertical_diffusion == true
        @test p.apply_evd_to_momentum             == true
        @test p.apply_prandtl_richardson          == false
    end

    @testset "Constructor: kwargs override defaults" begin
        clo = NEMOTKEVerticalDiffusivity(; Cᴸ = 0.2,
                                           apply_wave_penetration = false)
        @test clo.parameters.Cᴸ                     == 0.2
        @test clo.parameters.apply_wave_penetration == false
        # Other defaults still in place
        @test clo.parameters.Cᴷ                     == 0.1
        @test clo.parameters.apply_langmuir_circulation == true
    end

    @testset "Constructor: parameters + kwargs raises ArgumentError" begin
        @test_throws ArgumentError NEMOTKEVerticalDiffusivity(parameters = NEMOTKEParameters(),
                                                              Cᴸ = 0.2)
    end

    @testset "Surface BC: e(top) = max(rn_emin0, rn_ebb · u★²)" begin
        p = NEMOTKEParameters()
        # Above floor: |τ| = 1e-4 → u★² = 1e-4 → e_surf = 3.75e-4
        @test surface_TKE(1e-4, p) ≈ 3.75e-4 atol = 1e-12
        # Below floor: |τ| = 1e-7 → e_surf clipped to rn_emin0 = 1e-4
        @test surface_TKE(1e-7, p) ≈ 1e-4 atol = 1e-12
        # Zero stress: e_surf = rn_emin0
        @test surface_TKE(0.0, p) == p.minimum_surface_TKE
        # Friction-velocity helper just returns |τ| in kinematic units
        @test friction_velocity_squared(3e-3, 4e-3) ≈ sqrt(9e-6 + 16e-6) atol = 1e-12
    end

    @testset "Natural length scale: ℓ₀ = max(rn_mxl0, √(2e/N²))" begin
        p = NEMOTKEParameters()
        # Strong stratification: 2e/N² < rn_mxl0² → ℓ floored
        @test natural_length_scale(1e-8, 1e-2, p) == p.minimum_mixing_length
        # Weak stratification: 2e/N² ≫ rn_mxl0² → ℓ ≈ √(2e/N²)
        e_strong   = 1e-3
        N²_weak    = 1e-4
        @test natural_length_scale(e_strong, N²_weak, p) ≈ sqrt(2e-3 / 1e-4) atol = 1e-6
        # Negative N² is replaced by safe small value → unphysically large ℓ allowed
        @test natural_length_scale(1e-3, -1.0, p) > 1e3
    end

    @testset "Eddy coefficient: K = Cᴷ · ℓ · √e + floors" begin
        p = NEMOTKEParameters()
        ℓ, e = 5.0, 1e-3
        K_TKE = nemo_eddy_coefficient(e, ℓ, p)
        @test K_TKE ≈ 0.1 * 5.0 * sqrt(1e-3) atol = 1e-12
        # Floors apply when K_TKE below background.
        @test viscosity_with_floors(0.0, p, 1.0)   == p.νᵇ
        @test diffusivity_with_floors(0.0, p, 1.0) == p.κᵇ
        # Cap applies when K_TKE above maximum.
        @test viscosity_with_floors(10.0, p, 0.5)   == 0.5
        @test diffusivity_with_floors(10.0, p, 0.5) == 0.5
    end

    @testset "Dissipation balance: pure-dissipation decay" begin
        # No production, no buoyancy, no source: tridiagonal interior reduces to
        #   eⁿ⁺¹ = eⁿ / (1 + Δt · Cᴰ · √eⁿ / ℓ)
        # We verify the analytical decay matches step-by-step for a single column.
        p = NEMOTKEParameters()
        e = 1e-3
        ℓ = 5.0
        Δt = 60.0
        for _ in 1:10
            e = e / (1 + Δt * p.Cᴰ * sqrt(e) / ℓ)
        end
        # After 10 minutes of pure dissipation, e dropped substantially but stayed positive
        @test e < 1e-3
        @test e > p.minimum_TKE
    end

    @testset "EVD overwrite: triggered when N² ≤ -1e-12" begin
        p = NEMOTKEParameters()
        K_TKE = 1e-3
        # Strongly unstable → overwrite to κᶜⁿᵛ = 100
        @test evd_overwrite(K_TKE, -1e-10, p) == p.κᶜⁿᵛ
        # Stable → preserved
        @test evd_overwrite(K_TKE, 1e-6, p) == K_TKE
        # Threshold exactly -1e-12 → triggers (≤ test, not <)
        @test evd_overwrite(K_TKE, -1e-12, p) == p.κᶜⁿᵛ
        # Just above threshold → preserved
        @test evd_overwrite(K_TKE, -1e-13, p) == K_TKE
        # Momentum variant: same behaviour when apply_evd_to_momentum = true (default)
        @test evd_overwrite_momentum(K_TKE, -1e-10, p) == p.κᶜⁿᵛ
        # Momentum-EVD off but tracer-EVD on: momentum K preserved
        p_no_mom = NEMOTKEParameters(apply_evd_to_momentum = false)
        @test evd_overwrite_momentum(K_TKE, -1e-10, p_no_mom) == K_TKE
        @test evd_overwrite(K_TKE,           -1e-10, p_no_mom) == p_no_mom.κᶜⁿᵛ
        # Master switch off: nothing fires
        p_off = NEMOTKEParameters(apply_enhanced_vertical_diffusion = false)
        @test evd_overwrite(K_TKE, -1e-10, p_off) == K_TKE
    end

    @testset "Langmuir source: cubic-in-(Cᴸ·u_s), zero below h_LC" begin
        p = NEMOTKEParameters()
        τmag = 1e-4
        u_s  = stokes_velocity(τmag, p)
        @test u_s ≈ 0.016 * sqrt(1e-4) atol = 1e-12
        h_LC = 50.0
        # At z = h_LC/2, sin(π/2) = 1 → LC = (Cᴸ · u_s)³ / h_LC
        @test langmuir_source(25.0, h_LC, u_s, p) ≈ (0.15 * u_s)^3 / h_LC atol = 1e-20
        # Below the LC depth → zero source
        @test langmuir_source(75.0, h_LC, u_s, p) == 0.0
        # At z = 0 → sin(0) = 0 → zero source (boundary)
        @test langmuir_source(0.0, h_LC, u_s, p) == 0.0
    end

    @testset "Wave penetration: nn_etau=1 with nn_htau=1 lat-dep h_τ" begin
        p = NEMOTKEParameters()  # default nn_htau = 1 (lat-dep)
        # Equator: |sin(0)| = 0 → h_τ floored to 0.5 m
        @test wave_decay_length(0.0, p) == 0.5
        # 30° latitude: 45·|sin(30°)| = 22.5 m → within [0.5, 30] window
        @test wave_decay_length(30.0, p) ≈ 22.5 atol = 1e-12
        # 50° (OS Papa): 45·|sin(50°)| ≈ 34.5 m → clipped to 30
        @test wave_decay_length(50.0, p) == 30.0
        # nn_htau=0 path: constant 10 m
        p0 = NEMOTKEParameters(surface_length_scale_formulation = 0)
        @test wave_decay_length(50.0, p0) == 10.0

        # Source value: WP(z) = Cᶠ · e_surf · exp(-z/h_τ) · (1 − ℵ)
        e_surf = 1e-4
        h_τ    = 30.0
        ℵ      = 0.0
        @test wave_penetration_source(0.0, e_surf, h_τ, ℵ, p) ≈ 1.0 * 1e-4 * 1.0 atol = 1e-15
        @test wave_penetration_source(h_τ, e_surf, h_τ, ℵ, p) ≈ 1.0 * 1e-4 * exp(-1) atol = 1e-12
        # Sea-ice gating: full ice cover → zero source
        @test wave_penetration_source(0.0, e_surf, h_τ, 1.0, p) == 0.0
    end
end
