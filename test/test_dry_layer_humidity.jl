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
# `compute_interface_humidity(formulation, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)` (radiation
# state Ψᵣ unused by the dry-layer soil model, hence `nothing`).
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
    return ℂ, Ψₛ, Ψₐ, Ψᵢ, nothing, ℙₐ
end

@testset "DryLayerHumidity wet branch (𝒮 ≥ 𝒮ᶜ)" begin
    # wet_transition_width = 0 pins the sharp switch so the wet limit is exact.
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            wet_transition_width = 0),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    # 𝒮 = 0.5 ⇒ δᵛ = 0 ⇒ wet ⇒ qⁱⁿ = qᵛ⁺(Tⁱⁿ).
    Tⁱⁿ = 300.0
    pᵃᵗ = 1.0e5
    ℂ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(q; Tˡᵃ=290.0, Tⁱⁿ=Tⁱⁿ, 𝒮=0.5,
                                          pᵃᵗ=pᵃᵗ, qᵃᵗ=1.0e-2, Tᵃᵗ=295.0,
                                          u★=0.3, q★=-2.0e-4, qⁱⁿ⁻=0.005)
    qⁱⁿ★ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    qˢᵃᵗ = saturation_specific_humidity(ℂ, Tⁱⁿ, pᵃᵗ, AtmosphericThermodynamics.Liquid())
    @test isapprox(qⁱⁿ★, qˢᵃᵗ; atol = 1e-15)
end

@testset "DryLayerHumidity vapor divider" begin
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    # Fully dry: 𝒮 = 0 ⇒ δᵛ = δᵛ_max = 0.05, χ = 0.5 ⇒ Tᵉ = (Tⁱⁿ+Tˡᵃ)/2.
    Tˡᵃ = 290.0; Tⁱⁿ = 300.0
    pᵃᵗ = 1.0e5; qᵃᵗ = 1.0e-2; Tᵃᵗ = 295.0
    u★ = 0.3;   q★ = -2.0e-4; qⁱⁿ⁻ = 0.005
    ℂ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮 = 0.0,
                                          pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
    qⁱⁿ★ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

    Tᵉ  = (Tⁱⁿ + Tˡᵃ) / 2
    qᵉ  = saturation_specific_humidity(ℂ, Tᵉ, pᵃᵗ, AtmosphericThermodynamics.Liquid())
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    wᵈ  = 2.5e-5 / 0.05
    Gᵉ  = ρᵃᵗ * wᵈ
    Δq  = qⁱⁿ⁻ - qᵃᵗ
    Jᵃ  = -ρᵃᵗ * u★ * q★
    Gᵃ  = Jᵃ / Δq
    # Series (conductance-divider) form: qⁱⁿ is the conductance-weighted mean of qᵉ and qᵃᵗ.
    expected = (Gᵉ * qᵉ + Gᵃ * qᵃᵗ) / (Gᵉ + Gᵃ)
    @test isapprox(qⁱⁿ★, expected; atol = 1e-15)
end

@testset "DryLayerHumidity Tᵉ interpolation" begin
    # δᵛ controls χ = clip(δᵛ/ℓᵀ, 0, 1) ⇒ Tᵉ = Tⁱⁿ + χ(Tˡᵃ - Tⁱⁿ).
    # We don't directly expose Tᵉ, but the source humidity is qᵛ⁺(Tᵉ), so the
    # vapor balance pins Tᵉ implicitly. Use cases where 𝒮 → δᵛ is known.
    # 𝒮ᶜ = 0.5, η = 1, δᵛ_max = 0.05, ℓᵀ = 0.10:
    # 𝒮 = 0  → δᵛ = 0.05, χ = 0.5
    # 𝒮 = 0.25 → δᵛ = 0.025, χ = 0.25
    # 𝒮 = 0.5 → δᵛ = 0, wet branch.
    Tˡᵃ = 290.0; Tⁱⁿ = 310.0; pᵃᵗ = 1.0e5
    qᵃᵗ = 1.0e-2; Tᵃᵗ = 295.0; u★ = 0.3; q★ = -2.0e-4; qⁱⁿ⁻ = 0.005

    # wet_transition_width = 0 pins the sharp switch so the wet limit is exact.
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity = ConstantTortuosity(), wet_transition_width = 0),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    ℂ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮 = 0.0,
                                          pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
    qᵈʳʸ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

    ℂ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮 = 0.5,
                                          pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
    qʷᵉᵗ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

    # Wet (𝒮 ≥ 𝒮ᶜ): qⁱⁿ = qᵛ⁺(Tⁱⁿ); dry: qⁱⁿ < qᵛ⁺(Tⁱⁿ) because the source is colder.
    qˢᵃᵗ = saturation_specific_humidity(ℂ, Tⁱⁿ, pᵃᵗ, AtmosphericThermodynamics.Liquid())
    @test qʷᵉᵗ ≈ qˢᵃᵗ
    @test qᵈʳʸ < qʷᵉᵗ
end

@testset "DryLayerHumidity Gᵉ → 0 ⇒ qⁱⁿ → qᵃᵗ" begin
    # Gᵉ → 0 by setting Dᵛ₀ very small. Atmospheric flux drives qⁱⁿ toward qᵃᵗ.
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 1e-14,
            tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    Tⁱⁿ = 300.0; pᵃᵗ = 1.0e5; qᵃᵗ = 1.0e-2
    ℂ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(q; Tˡᵃ=290.0, Tⁱⁿ=Tⁱⁿ, 𝒮 = 0.0,
                                          pᵃᵗ=pᵃᵗ, qᵃᵗ=qᵃᵗ, Tᵃᵗ=295.0,
                                          u★=0.3, q★=-2.0e-4, qⁱⁿ⁻=0.005)
    qⁱⁿ★ = compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    @test isapprox(qⁱⁿ★, qᵃᵗ; atol = 1e-6)
end

@testset "DryLayerHumidity wet-transition blend" begin
    # η = 1 ⇒ δᵛ = δᵛ_max (1 - 𝒮/𝒮ᶜ), so 𝒮 dials δᵛ directly.
    make_q(wet_transition_width) = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(;
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity = ConstantTortuosity(), wet_transition_width),
        thermal_exchange_depth = 0.10, porosity = 0.4)

    sharp = make_q(0)
    blend = make_q(1e-2)

    Tˡᵃ = 290.0; Tⁱⁿ = 300.0; pᵃᵗ = 1.0e5
    qᵃᵗ = 1.0e-2; Tᵃᵗ = 295.0; u★ = 0.3; q★ = -2.0e-4; qⁱⁿ⁻ = 0.005
    function humidity(q, 𝒮)
        _, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ = _make_call_args(q; Tˡᵃ, Tⁱⁿ, 𝒮, pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)
        return compute_interface_humidity(q, Tⁱⁿ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    end

    ℂ = AtmosphereThermodynamicsParameters(Float64)
    qˢᵃᵗ = saturation_specific_humidity(ℂ, Tⁱⁿ, pᵃᵗ, AtmosphericThermodynamics.Liquid())

    # The default width is 5 δᵛ_min.
    default_exchange = DryLayerVaporPistonVelocity(
        minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5)
    @test default_exchange.wet_transition_width ≈ 5e-4

    # Sharp switch at 𝒮 ≥ 𝒮ᶜ: exact wet branch (logistic tail underflows).
    @test isapprox(humidity(sharp, 0.5), qˢᵃᵗ; atol = 1e-15)

    # Blended wet side (δᵛ = 0): the logistic tail admixes < 1 % of the series
    # solution, which is itself within a few percent of qˢᵃᵗ there.
    @test isapprox(humidity(blend, 0.5), qˢᵃᵗ; rtol = 1e-3)

    # Deep into the dry side (δᵛ = 0.05 ≫ δᵛ_min + δᵛʷ) the series solution
    # survives to rounding.
    @test isapprox(humidity(blend, 0.0), humidity(sharp, 0.0); rtol = 1e-14)

    # At the logistic center δᵛ = δᵛ_min + δᵛʷ/2 the weight is exactly 1/2, so
    # the blended humidity is the midpoint of the wet and series values.
    δᵛ  = 1e-4 + 1e-2 / 2
    𝒮ᵐ  = 0.5 * (1 - δᵛ / 0.05)
    qᵈʳʸ = humidity(sharp, 𝒮ᵐ)
    @test isapprox(humidity(blend, 𝒮ᵐ), (qˢᵃᵗ + qᵈʳʸ) / 2; rtol = 1e-12)
end
