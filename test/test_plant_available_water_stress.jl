include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    PlantAvailableWaterStress, CriticalSaturation, evaporation_efficiency,
    van_genuchten_saturation, van_genuchten_texture_parameters

using NumericalEarth: VanGenuchtenRetention

β(model, 𝒮) = evaporation_efficiency(model, (saturation = 𝒮,))

@testset "PlantAvailableWaterStress" begin
    for FT in (Float32, Float64)
        α = FT(1)
        # Spanning coarse sand-like to clay-like retention shapes.
        for n in (FT(1.2), FT(1.5), FT(3))
            stress = PlantAvailableWaterStress(FT; inverse_air_entry_head = α, pore_size_uniformity = n)
            𝒮ᶠᶜ = van_genuchten_saturation(α * stress.field_capacity_head, n)
            𝒮ʷᵖ = van_genuchten_saturation(α * stress.wilting_point_head, n)
            @test 0 < 𝒮ʷᵖ < 𝒮ᶠᶜ < 1

            # Exact endpoints and clamped tails.
            @test β(stress, 𝒮ᶠᶜ) == 1
            @test β(stress, 𝒮ʷᵖ) == 0
            @test β(stress, one(FT)) == 1
            @test β(stress, zero(FT)) == 0

            # Monotone and bounded on a saturation sweep.
            values = [β(stress, 𝒮) for 𝒮 in range(zero(FT), one(FT); length = 101)]
            @test issorted(values)
            @test all(v -> 0 <= v <= 1, values)

            # Linear in the interior with slope 1/(𝒮ᶠᶜ − 𝒮ʷᵖ).
            𝒮₁ = (2𝒮ʷᵖ + 𝒮ᶠᶜ) / 3
            𝒮₂ = (𝒮ʷᵖ + 2𝒮ᶠᶜ) / 3
            slope = (β(stress, 𝒮₂) - β(stress, 𝒮₁)) / (𝒮₂ - 𝒮₁)
            @test slope ≈ 1 / (𝒮ᶠᶜ - 𝒮ʷᵖ) rtol = sqrt(eps(FT))
        end
    end

    # The saturation form is the liquid-fraction form: with θ = θʳ + (ν − θʳ)𝒮 the ratio
    # (θ − θʷᵖ)/(θᶠᶜ − θʷᵖ) is invariant, so β never needs the porosity or the residual.
    let FT = Float64
        α, n = FT(3.6), FT(1.6)
        stress = PlantAvailableWaterStress(FT; inverse_air_entry_head = α, pore_size_uniformity = n)
        𝒮ᶠᶜ = van_genuchten_saturation(α * stress.field_capacity_head, n)
        𝒮ʷᵖ = van_genuchten_saturation(α * stress.wilting_point_head, n)
        for (ν, θʳ) in ((0.35, 0.0), (0.45, 0.05), (0.6, 0.12)), 𝒮 in (0.05, 0.3, 0.7)
            θ(s) = θʳ + (ν - θʳ) * s
            β_θ = clamp((θ(𝒮) - θ(𝒮ʷᵖ)) / (θ(𝒮ᶠᶜ) - θ(𝒮ʷᵖ)), 0, 1)
            @test β(stress, 𝒮) ≈ β_θ atol = 4eps(FT)
        end
    end

    # Contrast with the bare-soil model: stomata shut at the wilting point (a positive
    # saturation), where `CriticalSaturation` still evaporates; and full efficiency is not
    # reached until field capacity.
    let FT = Float64
        stress = PlantAvailableWaterStress(FT; inverse_air_entry_head = 1, pore_size_uniformity = 2)
        bare = CriticalSaturation(FT(0.5))
        𝒮ʷᵖ = van_genuchten_saturation(FT(150), FT(2))
        @test β(stress, 𝒮ʷᵖ) == 0
        @test β(bare, 𝒮ʷᵖ) > 0
        @test β(stress, 𝒮ʷᵖ / 2) == 0
        𝒮ᶠᶜ = van_genuchten_saturation(FT(1), FT(2))
        𝒮ᵐⁱᵈ = (0.5 + 𝒮ᶠᶜ) / 2
        @test β(stress, 𝒮ᵐⁱᵈ) < 1
        @test β(bare, 𝒮ᵐⁱᵈ) == 1
    end

    # β is smooth in the wilting-point head: centered differences at two step sizes agree.
    let FT = Float64
        β_of_ψ(ψ) = β(PlantAvailableWaterStress(FT; inverse_air_entry_head = 1,
                                                pore_size_uniformity = 2,
                                                wilting_point_head = ψ), FT(0.3))
        derivative(δ) = (β_of_ψ(150 + δ) - β_of_ψ(150 - δ)) / 2δ
        @test derivative(1e-2) ≈ derivative(1e-3) rtol = 1e-4
        @test derivative(1e-3) != 0
    end

    # Parameter sources: the hydrology's retention curve and a texture class build the
    # same closure as the explicit numbers they stand for.
    let FT = Float64
        explicit = PlantAvailableWaterStress(FT; inverse_air_entry_head = 1, pore_size_uniformity = 2)
        shared   = PlantAvailableWaterStress(FT; retention_curve = VanGenuchtenRetention(α = 1, n = 2))
        @test shared === explicit

        loam = van_genuchten_texture_parameters(:loam)
        @test loam == (inverse_air_entry_head = 3.6, pore_size_uniformity = 1.56)
        @test PlantAvailableWaterStress(FT; texture = :loam) ===
              PlantAvailableWaterStress(FT; inverse_air_entry_head = loam.inverse_air_entry_head,
                                        pore_size_uniformity = loam.pore_size_uniformity)
        @test_throws ArgumentError van_genuchten_texture_parameters(:peat)
    end

    # Constructor validation: exactly one parameter source, physical parameters.
    @test_throws ArgumentError PlantAvailableWaterStress()
    @test_throws ArgumentError PlantAvailableWaterStress(inverse_air_entry_head = 1)
    @test_throws ArgumentError PlantAvailableWaterStress(texture = :loam, pore_size_uniformity = 2)
    @test_throws ArgumentError PlantAvailableWaterStress(retention_curve = VanGenuchtenRetention(α = 1, n = 2),
                                                         texture = :loam)
    @test_throws ArgumentError PlantAvailableWaterStress(inverse_air_entry_head = 1, pore_size_uniformity = 1)
    @test_throws ArgumentError PlantAvailableWaterStress(inverse_air_entry_head = 1, pore_size_uniformity = 2,
                                                         field_capacity_head = 200)
end
