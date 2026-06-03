include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    compute_interface_humidity,
    AirLandInterfaceState,
    InterfaceFluxScales,
    InterfaceVelocities,
    saturation_specific_humidity
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

# Build a state that the formulation can read; the kernel signature mirrors
# `compute_interface_humidity(formulation, Tₛ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)`.
function _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮, pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
    FT = Float64
    ℂ  = AtmosphereThermodynamicsParameters(FT)
    Ψₐ = (T = Tᵃᵗ, p = pᵃᵗ, q = qᵃᵗ, u = 1.0, v = 0.0, z = 10.0, h_bℓ = 1000.0)
    Ψₛ = AirLandInterfaceState(InterfaceFluxScales(u★, 0.0, q★),
                               InterfaceVelocities(0.0, 0.0),
                               Tⁱⁿ, qⁱⁿ⁻, (saturation=𝒮,), (temperature=Tˡᵃ,))
    Ψᵢ = (T = Tˡᵃ,)
    ℙₐ = (thermodynamics_parameters = ℂ,
          surface_layer_height = 10.0,
          gravitational_acceleration = 9.81)
    return ℂ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ
end

@testset "EvaporationFrontHumidity wet branch (𝒮 ≥ 𝒮ᶜ)" begin
    q = EvaporationFrontHumidity(;
        evaporation_front_depth = StorageBasedEvaporationFrontDepth(
            maximum_front_depth = 0.05, critical_saturation = 0.5, front_depth_exponent = 2.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_front_depth = 1e-4, molecular_diffusivity = 2.5e-5),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    # 𝒮 = 0.5 ⇒ δᵛ = 0 ⇒ wet ⇒ qⁱⁿ = qᵛ⁺(Tⁱⁿ).
    Tⁱⁿ = 300.0
    pᵃᵗ = 1.0e5
    ℂ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ = _make_call_args(q; Tˡᵃ=290.0, Tⁱⁿ=Tⁱⁿ, 𝒮=0.5,
                                          pᵃᵗ=pᵃᵗ, qᵃᵗ=1.0e-2, Tᵃᵗ=295.0,
                                          u★=0.3, q★=-2.0e-4, qⁱⁿ⁻=0.005)
    qⁱⁿ★ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)
    qˢᵃᵗ = saturation_specific_humidity(ℂ, Tⁱⁿ, pᵃᵗ, AtmosphericThermodynamics.Liquid())
    @test isapprox(qⁱⁿ★, qˢᵃᵗ; atol = 1e-15)
end

@testset "EvaporationFrontHumidity vapor divider" begin
    q = EvaporationFrontHumidity(;
        evaporation_front_depth = StorageBasedEvaporationFrontDepth(
            maximum_front_depth = 0.05, critical_saturation = 0.5, front_depth_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_front_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity_model = ConstantTortuosity()),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    # Fully dry: 𝒮 = 0 ⇒ δᵛ = δᵛ_max = 0.05, χ = 0.5 ⇒ Tᵉ = (Tⁱⁿ+Tˡᵃ)/2.
    Tˡᵃ = 290.0; Tⁱⁿ = 300.0
    pᵃᵗ = 1.0e5; qᵃᵗ = 1.0e-2; Tᵃᵗ = 295.0
    u★ = 0.3;   q★ = -2.0e-4; qⁱⁿ⁻ = 0.005
    ℂ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ = _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮 = 0.0,
                                          pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
    qⁱⁿ★ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)

    Tᵉ  = (Tⁱⁿ + Tˡᵃ) / 2
    qᵉ  = saturation_specific_humidity(ℂ, Tᵉ, pᵃᵗ, AtmosphericThermodynamics.Liquid())
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    wᵈ  = 2.5e-5 / 0.05
    Gᵉ  = ρᵃᵗ * wᵈ
    Δq  = qⁱⁿ⁻ - qᵃᵗ
    Jᵃ  = -ρᵃᵗ * u★ * q★
    expected = (Gᵉ * qᵉ * Δq + Jᵃ * qᵃᵗ) / (Gᵉ * Δq + Jᵃ)
    @test isapprox(qⁱⁿ★, expected; atol = 1e-15)
end

@testset "EvaporationFrontHumidity Tᵉ interpolation" begin
    # δᵛ controls χ = clip(δᵛ/ℓᵀ, 0, 1) ⇒ Tᵉ = Tⁱⁿ + χ(Tˡᵃ - Tⁱⁿ).
    # We don't directly expose Tᵉ, but the source humidity is qᵛ⁺(Tᵉ), so the
    # vapor balance pins Tᵉ implicitly. Use cases where 𝒮 → δᵛ is known.
    # 𝒮ᶜ = 0.5, η = 1, δᵛ_max = 0.05, ℓᵀ = 0.10:
    # 𝒮 = 0  → δᵛ = 0.05, χ = 0.5
    # 𝒮 = 0.25 → δᵛ = 0.025, χ = 0.25
    # 𝒮 = 0.5 → δᵛ = 0, wet branch.
    Tˡᵃ = 290.0; Tⁱⁿ = 310.0; pᵃᵗ = 1.0e5
    qᵃᵗ = 1.0e-2; Tᵃᵗ = 295.0; u★ = 0.3; q★ = -2.0e-4; qⁱⁿ⁻ = 0.005

    q = EvaporationFrontHumidity(;
        evaporation_front_depth = StorageBasedEvaporationFrontDepth(
            maximum_front_depth = 0.05, critical_saturation = 0.5, front_depth_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_front_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity_model = ConstantTortuosity()),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    ℂ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ = _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮 = 0.0,
                                          pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
    qⁱⁿ★_dry = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)

    ℂ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ = _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮 = 0.5,
                                          pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
    qⁱⁿ★_wet = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)

    # Wet (𝒮 ≥ 𝒮ᶜ): qⁱⁿ = qᵛ⁺(Tⁱⁿ); dry: qⁱⁿ < qᵛ⁺(Tⁱⁿ) because the source is colder.
    qˢᵃᵗ_Tⁱⁿ = saturation_specific_humidity(ℂ, Tⁱⁿ, pᵃᵗ, AtmosphericThermodynamics.Liquid())
    @test qⁱⁿ★_wet ≈ qˢᵃᵗ_Tⁱⁿ
    @test qⁱⁿ★_dry < qⁱⁿ★_wet
end

@testset "EvaporationFrontHumidity Gᵉ → 0 ⇒ qⁱⁿ → qᵃᵗ" begin
    # Gᵉ → 0 by setting Dᵛ₀ very small. Atmospheric flux drives qⁱⁿ toward qᵃᵗ.
    q = EvaporationFrontHumidity(;
        evaporation_front_depth = StorageBasedEvaporationFrontDepth(
            maximum_front_depth = 0.05, critical_saturation = 0.5, front_depth_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_front_depth = 1e-4, molecular_diffusivity = 1e-14,
            tortuosity_model = ConstantTortuosity()),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    Tⁱⁿ = 300.0; pᵃᵗ = 1.0e5; qᵃᵗ = 1.0e-2
    ℂ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ = _make_call_args(q; Tˡᵃ=290.0, Tⁱⁿ=Tⁱⁿ, 𝒮 = 0.0,
                                          pᵃᵗ=pᵃᵗ, qᵃᵗ=qᵃᵗ, Tᵃᵗ=295.0,
                                          u★=0.3, q★=-2.0e-4, qⁱⁿ⁻=0.005)
    qⁱⁿ★ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)
    @test isapprox(qⁱⁿ★, qᵃᵗ; atol = 1e-6)
end
