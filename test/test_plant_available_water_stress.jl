include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    PlantAvailableWaterStress, CriticalSaturation, CanopyConductanceHumidity,
    evaporation_efficiency, interface_hydrology_state, requires_retention_curve,
    atmosphere_land_interface

using NumericalEarth: VanGenuchtenRetention
using NumericalEarth.Lands: van_genuchten_saturation

# Route through the per-cell materialization, as the flux kernel does: the endpoints come
# off the land's retention curve, not off the closure.
land_state(𝒮, curve) = (saturation = 𝒮, retention_curve = curve)

# The moisture availability β the formulation derives from that state.
availability(model, 𝒮, curve) =
    evaporation_efficiency(model, interface_hydrology_state(1, 1, nothing, model,
                                                            land_state(𝒮, curve)))

@testset "PlantAvailableWaterStress" begin
    for FT in (Float32, Float64)
        α = FT(1)
        # Spanning coarse sand-like to clay-like retention shapes.
        for n in (FT(1.2), FT(1.5), FT(3))
            stress = PlantAvailableWaterStress(FT)
            curve  = VanGenuchtenRetention(FT; α, n)
            𝒮ᶠᶜ = van_genuchten_saturation(α * stress.field_capacity_head, n)
            𝒮ʷᵖ = van_genuchten_saturation(α * stress.wilting_point_head, n)
            @test 0 < 𝒮ʷᵖ < 𝒮ᶠᶜ < 1

            # Exact endpoints and clamped tails.
            @test availability(stress, 𝒮ᶠᶜ, curve) == 1
            @test availability(stress, 𝒮ʷᵖ, curve) == 0
            @test availability(stress, one(FT), curve) == 1
            @test availability(stress, zero(FT), curve) == 0

            # Monotone and bounded on a saturation sweep.
            sweep = range(zero(FT), one(FT); length = 101)
            values = [availability(stress, 𝒮, curve) for 𝒮 in sweep]
            @test issorted(values)
            @test all(v -> 0 <= v <= 1, values)

            # Linear in the interior with slope 1/(𝒮ᶠᶜ − 𝒮ʷᵖ).
            𝒮₁ = (2𝒮ʷᵖ + 𝒮ᶠᶜ) / 3
            𝒮₂ = (𝒮ʷᵖ + 2𝒮ᶠᶜ) / 3
            slope = (availability(stress, 𝒮₂, curve) -
                     availability(stress, 𝒮₁, curve)) / (𝒮₂ - 𝒮₁)
            @test slope ≈ 1 / (𝒮ᶠᶜ - 𝒮ʷᵖ) rtol = sqrt(eps(FT))
        end
    end

    # The saturation form is the liquid-fraction form: with θ = θʳ + (ν − θʳ)𝒮 the ratio
    # (θ − θʷᵖ)/(θᶠᶜ − θʷᵖ) is invariant, so β never needs the porosity or the residual.
    let FT = Float64
        α, n = FT(3.6), FT(1.6)
        stress = PlantAvailableWaterStress(FT)
        curve  = VanGenuchtenRetention(FT; α, n)
        𝒮ᶠᶜ = van_genuchten_saturation(α * stress.field_capacity_head, n)
        𝒮ʷᵖ = van_genuchten_saturation(α * stress.wilting_point_head, n)
        for (ν, θʳ) in ((0.35, 0.0), (0.45, 0.05), (0.6, 0.12)), 𝒮 in (0.05, 0.3, 0.7)
            θ(s) = θʳ + (ν - θʳ) * s
            β_θ = clamp((θ(𝒮) - θ(𝒮ʷᵖ)) / (θ(𝒮ᶠᶜ) - θ(𝒮ʷᵖ)), 0, 1)
            @test availability(stress, 𝒮, curve) ≈ β_θ atol = 4eps(FT)
        end
    end

    # Contrast with the bare-soil model: stomata shut at the wilting point (a positive
    # saturation), where `CriticalSaturation` still evaporates; and full efficiency is not
    # reached until field capacity.
    let FT = Float64
        stress = PlantAvailableWaterStress(FT)
        curve  = VanGenuchtenRetention(FT; α = 1, n = 2)
        bare = CriticalSaturation(FT(0.5))
        𝒮ʷᵖ = van_genuchten_saturation(FT(150), FT(2))
        @test availability(stress, 𝒮ʷᵖ, curve) == 0
        @test availability(bare, 𝒮ʷᵖ, curve) > 0
        @test availability(stress, 𝒮ʷᵖ / 2, curve) == 0
        𝒮ᶠᶜ = van_genuchten_saturation(FT(1), FT(2))
        𝒮ᵐⁱᵈ = (0.5 + 𝒮ᶠᶜ) / 2
        @test availability(stress, 𝒮ᵐⁱᵈ, curve) < 1
        @test availability(bare, 𝒮ᵐⁱᵈ, curve) == 1
    end

    # β is smooth in the wilting-point head: centered differences at two step sizes agree.
    let FT = Float64
        curve = VanGenuchtenRetention(FT; α = 1, n = 2)
        efficiency_of_ψ(ψ) =
            availability(PlantAvailableWaterStress(FT; wilting_point_head = ψ), FT(0.3), curve)
        derivative(δ) = (efficiency_of_ψ(150 + δ) - efficiency_of_ψ(150 - δ)) / 2δ
        @test derivative(1e-2) ≈ derivative(1e-3) rtol = 1e-4
        @test derivative(1e-3) != 0
    end

    # The endpoints follow the curve they are handed, not the closure: one stress object
    # reads two soils and lands on two different stress bands. `effective_saturation`
    # reads the curve at `(i, j)`, so per-cell `Field` parameters need nothing here.
    let FT = Float64
        stress = PlantAvailableWaterStress(FT)
        loam = VanGenuchtenRetention(FT; α = 3.6, n = 1.56)   # Carsel-Parrish means
        clay = VanGenuchtenRetention(FT; α = 0.8, n = 1.09)
        endpoints(curve) =
            interface_hydrology_state(1, 1, nothing, stress, land_state(FT(0.3), curve))

        @test endpoints(loam).field_capacity_saturation ≈ 0.46628 atol = 1e-5
        @test endpoints(loam).wilting_saturation        ≈ 0.02950 atol = 1e-5
        # A clay wilts wetter than a loam is ever field-capacity dry.
        @test endpoints(clay).wilting_saturation > endpoints(loam).field_capacity_saturation
        # So the same wetness is unstressed on one soil and wilted on the other.
        @test availability(stress, FT(0.6), loam) == 1
        @test availability(stress, FT(0.6), clay) == 0
    end

    # Head validation.
    @test_throws ArgumentError PlantAvailableWaterStress(field_capacity_head = 200)
    @test_throws ArgumentError PlantAvailableWaterStress(field_capacity_head = 0)

    # The requirement is visible to the interface through nested formulations, and a
    # hydrology carrying no retention curve is rejected where the interface is built.
    let FT = Float64
        stress = PlantAvailableWaterStress(FT)
        transpiring = CanopyConductanceHumidity(FT; leaf_area_index = 2,
                                                moisture_stress = stress)
        canopy = CanopyAirSpace(FT; soil = BulkHumidity(), canopy = transpiring)

        @test !requires_retention_curve(CriticalSaturation(FT(0.5)))
        @test requires_retention_curve(stress)
        @test requires_retention_curve(transpiring)
        @test requires_retention_curve(canopy)

        grid = RectilinearGrid(FT; size = (), topology = (Flat, Flat, Flat))
        atmosphere = PrescribedAtmosphere(grid, [0.0, 1.0]; surface_layer_height = 10)
        interface(land) = atmosphere_land_interface(grid, atmosphere, land;
                                                    temperature = canopy,
                                                    specific_humidity = canopy)

        bucket = SlabLand(grid; hydrology = BucketHydrology())
        @test_throws ArgumentError interface(bucket)

        soil = SlabLand(grid; hydrology = VariablySaturatedHydrology(FT;
            slab_depth = 0.1, porosity = 0.4, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(FT; α = 3.6, n = 1.56),
            hydraulic_conductivity = VanGenuchtenConductivity(FT;
                                                              K_saturated = 1e-6, n = 1.56)))
        @test !isnothing(interface(soil))
    end
end
