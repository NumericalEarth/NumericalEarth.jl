include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    compute_interface_humidity,
    AirLandInterfaceState,
    InterfaceFluxScales,
    InterfaceVelocities,
    saturation_specific_humidity, SoilVaporDiffusivity
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

const ℂ = AtmosphereThermodynamicsParameters(Float64)
qᵛ⁺(T) = saturation_specific_humidity(ℂ, T, 1.0e5, AtmosphericThermodynamics.Liquid())

# `compute_interface_humidity(q, Tᵍʳᵒ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)`; the dry-layer soil model ignores the radiation state Ψᵣ.
function humidity(q, 𝒮; Tˡᵃ = 290.0, Tᵍʳᵒ = 300.0, pᵃᵗ = 1.0e5, qᵃᵗ = 1.0e-2, Tᵃᵗ = 295.0, u★ = 0.3, q★ = 2.0e-4, qⁱⁿ⁻ = 0.005)
    Ψₐ = (T = Tᵃᵗ, p = pᵃᵗ, q = qᵃᵗ, u = 1.0, v = 0.0, z = 10.0, h_bℓ = 1000.0)
    χq = q★ / (qᵃᵗ - qⁱⁿ⁻)   # transfer coefficient consistent with q★ = χq Δq
    Ψₛ = AirLandInterfaceState(InterfaceFluxScales(u★, 0.0, q★, 0.0, χq), InterfaceVelocities(0.0, 0.0), Tᵍʳᵒ, qⁱⁿ⁻)
    Ψˡᵃ = (T = Tˡᵃ, 𝒮 = 𝒮, effective_vapor_diffusivity = q.vapor_exchange.molecular_diffusivity,
           thermal_exchange_depth = 0.10)
    ℙₐ = (thermodynamics_parameters = ℂ, surface_layer_height = 10.0, gravitational_acceleration = 9.81)
    return compute_interface_humidity(q, Tᵍʳᵒ, Ψₛ, Ψₐ, Ψˡᵃ, nothing, ℙₐ)
end

@testset "Soil vapor diffusion follows porosity and residual water" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = 2, x = (0, 2), topology = (Bounded, Flat, Flat))
        ν, θʳ = reshape([0.35, 0.55], 2, 1, 1), reshape([0.03, 0.10], 2, 1, 1)
        porosity, residual = set!(CenterField(grid), ν), set!(CenterField(grid), θʳ)
        hydrology = VariablySaturatedHydrology(; slab_depth = 1, porosity, residual_liquid_fraction = residual, storage_height = 0.1,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 2, pore_size_uniformity = 1.5),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5))
        land = SlabLand(grid; hydrology)
        formulation = DryLayerHumidity(;
            dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5),
            vapor_exchange = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
                                                         tortuosity = PowerLawTortuosity()))
        diffusivity = Field(SoilVaporDiffusivity(formulation.vapor_exchange)(land))
        for 𝒮 in (0.2, 0.6)
            θ = @. θʳ + 𝒮 * (ν - θʳ)
            set!(land; M = 1000 .* θ)
            compute!(diffusivity)
            @test Array(interior(diffusivity)) ≈ @. 2.5e-5 * (ν - θ)^(10/3) / ν^2
        end
    end
end

@testset "DryLayerHumidity wet branch (𝒮 ≥ 𝒮ᶜ)" begin
    # wet_transition_width = 0 pins the sharp switch so the wet limit is exact.
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            wet_transition_width = 0))

    # 𝒮 = 0.5 ⇒ δᵛ = 0 ⇒ wet ⇒ qⁱⁿ = qᵛ⁺(Tᵍʳᵒ).
    @test isapprox(humidity(q, 0.5), qᵛ⁺(300.0); atol = 1e-15)
end

@testset "DryLayerHumidity vapor divider" begin
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity = ConstantTortuosity()))

    # Fully dry: 𝒮 = 0 ⇒ δᵛ = δᵛ_max = 0.05, χ = 0.5 ⇒ Tᵉ = (Tᵍʳᵒ+Tˡᵃ)/2.
    Tˡᵃ = 290.0; Tᵍʳᵒ = 300.0
    pᵃᵗ = 1.0e5; qᵃᵗ = 1.0e-2; Tᵃᵗ = 295.0
    u★ = 0.3;   q★ = 2.0e-4; qⁱⁿ⁻ = 0.005
    qⁱⁿ★ = humidity(q, 0.0; Tˡᵃ, Tᵍʳᵒ, pᵃᵗ, qᵃᵗ, Tᵃᵗ, u★, q★, qⁱⁿ⁻)

    Tᵉ  = (Tᵍʳᵒ + Tˡᵃ) / 2
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
    # δᵛ controls χ = clip(δᵛ/ℓᵀ, 0, 1) ⇒ Tᵉ = Tᵍʳᵒ + χ(Tˡᵃ - Tᵍʳᵒ).
    # We don't directly expose Tᵉ, but the source humidity is qᵛ⁺(Tᵉ), so the
    # vapor balance pins Tᵉ implicitly. Use cases where 𝒮 → δᵛ is known.
    # 𝒮ᶜ = 0.5, η = 1, δᵛ_max = 0.05, ℓᵀ = 0.10:
    # 𝒮 = 0  → δᵛ = 0.05, χ = 0.5
    # 𝒮 = 0.25 → δᵛ = 0.025, χ = 0.25
    # 𝒮 = 0.5 → δᵛ = 0, wet branch.

    # wet_transition_width = 0 pins the sharp switch so the wet limit is exact.
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity = ConstantTortuosity(), wet_transition_width = 0))

    qᵈʳʸ = humidity(q, 0.0; Tᵍʳᵒ = 310.0)
    qʷᵉᵗ = humidity(q, 0.5; Tᵍʳᵒ = 310.0)

    # Wet (𝒮 ≥ 𝒮ᶜ): qⁱⁿ = qᵛ⁺(Tᵍʳᵒ); dry: qⁱⁿ < qᵛ⁺(Tᵍʳᵒ) because the source is colder.
    @test qʷᵉᵗ ≈ qᵛ⁺(310.0)
    @test qᵈʳʸ < qʷᵉᵗ
end

@testset "DryLayerHumidity Gᵉ → 0 ⇒ qⁱⁿ → qᵃᵗ" begin
    # Gᵉ → 0 by setting Dᵛ₀ very small. Atmospheric flux drives qⁱⁿ toward qᵃᵗ.
    q = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 1e-14,
            tortuosity = ConstantTortuosity()))
    @test isapprox(humidity(q, 0.0), 1.0e-2; atol = 1e-6)
end

@testset "DryLayerHumidity wet-transition blend" begin
    # η = 1 ⇒ δᵛ = δᵛ_max (1 - 𝒮/𝒮ᶜ), so 𝒮 dials δᵛ directly.
    make_q(wet_transition_width) = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(
            maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 0.5, dry_layer_exponent = 1.0),
        vapor_exchange = DryLayerVaporPistonVelocity(;
            minimum_dry_layer_depth = 1e-4, molecular_diffusivity = 2.5e-5,
            tortuosity = ConstantTortuosity(), wet_transition_width))

    sharp = make_q(0)
    blend = make_q(1e-2)
    qˢᵃᵗ = qᵛ⁺(300.0)

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
