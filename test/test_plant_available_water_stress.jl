include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    PlantAvailableWaterStress, CriticalSaturation, CanopyConductanceHumidity,
    evaporation_efficiency, interface_hydrology_state,
    atmosphere_land_interface, surface_hydrology

using NumericalEarth: VanGenuchtenRetention
using NumericalEarth.Lands: van_genuchten_saturation, effective_saturation

land_state(𝒮, curve, stress) = (𝒮 = 𝒮, root_zone_saturation = 𝒮,
    field_capacity_saturation = effective_saturation(1, 1, 1, nothing, curve, stress.field_capacity_head),
    wilting_point_saturation = effective_saturation(1, 1, 1, nothing, curve, stress.wilting_point_head))

function availability(model, 𝒮, curve)
    stress = model isa PlantAvailableWaterStress ? model : PlantAvailableWaterStress(typeof(𝒮))
    return evaporation_efficiency(model, interface_hydrology_state(1, 1, nothing, model, land_state(𝒮, curve, stress)))
end

@testset "PlantAvailableWaterStress" begin
    for FT in (Float32, Float64), n in (FT(1.2), FT(1.5), FT(3))
        stress = PlantAvailableWaterStress(FT)
        curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 1, pore_size_uniformity = n)
        β(𝒮) = availability(stress, 𝒮, curve)
        𝒮ᶠᶜ, 𝒮ʷᵖ = van_genuchten_saturation(stress.field_capacity_head, n), van_genuchten_saturation(stress.wilting_point_head, n)
        @test 0 < 𝒮ʷᵖ < 𝒮ᶠᶜ < 1
        @test β(𝒮ᶠᶜ) == β(one(FT)) == 1
        @test β(𝒮ʷᵖ) == β(zero(FT)) == 0
        values = β.(range(zero(FT), one(FT); length = 101))
        @test issorted(values) && all(v -> 0 ≤ v ≤ 1, values)
        𝒮₁, 𝒮₂ = (2𝒮ʷᵖ + 𝒮ᶠᶜ) / 3, (𝒮ʷᵖ + 2𝒮ᶠᶜ) / 3
        @test (β(𝒮₂) - β(𝒮₁)) / (𝒮₂ - 𝒮₁) ≈ 1 / (𝒮ᶠᶜ - 𝒮ʷᵖ) rtol = sqrt(eps(FT))
    end

    FT = Float64
    stress = PlantAvailableWaterStress(FT)

    # β is the same in saturation and in liquid fraction θ = θʳ + (ν − θʳ)𝒮
    curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 3.6, pore_size_uniformity = 1.6)
    𝒮ᶠᶜ = van_genuchten_saturation(3.6 * stress.field_capacity_head, 1.6)
    𝒮ʷᵖ = van_genuchten_saturation(3.6 * stress.wilting_point_head, 1.6)
    for (ν, θʳ) in ((0.35, 0.0), (0.45, 0.05), (0.6, 0.12)), 𝒮 in (0.05, 0.3, 0.7)
        θ(s) = θʳ + (ν - θʳ) * s
        @test availability(stress, 𝒮, curve) ≈ clamp((θ(𝒮) - θ(𝒮ʷᵖ)) / (θ(𝒮ᶠᶜ) - θ(𝒮ʷᵖ)), 0, 1) atol = 4eps(FT)
    end

    # Stomata shut at the wilting point, where bare soil still evaporates, and open fully only at field capacity
    curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 1, pore_size_uniformity = 2)
    bare = CriticalSaturation(FT(0.5))
    𝒮ʷᵖ, 𝒮ᵐⁱᵈ = van_genuchten_saturation(FT(150), FT(2)), (0.5 + van_genuchten_saturation(FT(1), FT(2))) / 2
    @test availability(stress, 𝒮ʷᵖ, curve) == availability(stress, 𝒮ʷᵖ / 2, curve) == 0
    @test availability(bare, 𝒮ʷᵖ, curve) > 0
    @test availability(stress, 𝒮ᵐⁱᵈ, curve) < 1
    @test availability(bare, 𝒮ᵐⁱᵈ, curve) == 1

    # β is smooth in the wilting-point head
    efficiency(ψ) = availability(PlantAvailableWaterStress(FT; wilting_point_head = ψ), FT(0.3), curve)
    derivative(δ) = (efficiency(150 + δ) - efficiency(150 - δ)) / 2δ
    @test derivative(1e-2) ≈ derivative(1e-3) rtol = 1e-4
    @test derivative(1e-3) != 0

    # One stress object reads two soils onto two stress bands (Carsel-Parrish loam and clay)
    loam = VanGenuchtenRetention(FT; inverse_air_entry_head = 3.6, pore_size_uniformity = 1.56)
    clay = VanGenuchtenRetention(FT; inverse_air_entry_head = 0.8, pore_size_uniformity = 1.09)
    endpoints(curve) = interface_hydrology_state(1, 1, nothing, stress, land_state(FT(0.3), curve, stress))
    @test endpoints(loam).field_capacity_saturation ≈ 0.46628 atol = 1e-5
    @test endpoints(loam).wilting_point_saturation  ≈ 0.02950 atol = 1e-5
    @test endpoints(clay).wilting_point_saturation > endpoints(loam).field_capacity_saturation
    @test availability(stress, FT(0.6), loam) == 1
    @test availability(stress, FT(0.6), clay) == 0

    # A Field-valued curve puts each column on its own endpoints
    columns_grid = RectilinearGrid(FT; size = (1, 2), extent = (1, 1), topology = (Periodic, Periodic, Flat))
    inverse_air_entry_head = set!(Field{Center, Center, Nothing}(columns_grid), (x, y) -> y < 0.5 ? 3.6 : 0.8)
    pore_size_uniformity = set!(Field{Center, Center, Nothing}(columns_grid), (x, y) -> y < 0.5 ? 1.56 : 1.09)
    hydrology = VariablySaturatedHydrology(FT; slab_depth = 1, porosity = 0.4, storage_height = 0.1,
        retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head, pore_size_uniformity),
        hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity))
    column_endpoints = surface_hydrology(SlabLand(columns_grid; hydrology), stress, columns_grid)
    for (j, curve) in enumerate((loam, clay)), 𝒮 in (FT(0.35), FT(0.6))
        state = interface_hydrology_state(1, j, columns_grid, stress, merge((; root_zone_saturation = 𝒮), column_endpoints))
        @test evaporation_efficiency(stress, state) == availability(stress, 𝒮, curve)
    end

    @test_throws ArgumentError PlantAvailableWaterStress(field_capacity_head = 200)
    @test_throws ArgumentError PlantAvailableWaterStress(field_capacity_head = 0)

    # Nested formulations expose the requirement for a retention curve to the interface
    canopy = CanopyGroundSurface(FT; soil_humidity = BulkHumidity(),
                                 canopy_humidity = CanopyConductanceHumidity(FT; leaf_area_index = 2, moisture_stress = stress))
    point_grid = RectilinearGrid(FT; size = (), topology = (Flat, Flat, Flat))
    atmosphere = PrescribedAtmosphere(point_grid, [0.0, 1.0]; surface_layer_height = 10)
    interface(hydrology) = atmosphere_land_interface(point_grid, atmosphere, SlabLand(point_grid; hydrology);
                                                     temperature = canopy, specific_humidity = canopy)
    @test_throws ArgumentError interface(BucketHydrology())
    interface(VariablySaturatedHydrology(FT; slab_depth = 0.1, porosity = 0.4, storage_height = 1000, retention_curve = loam,
              hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity = 1.56)))
end
