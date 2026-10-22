include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Lands: viscosity_correction

# Darcy's law with van Genuchten–Mualem properties, evaluated in Float64 for the reference.
function reference_interlayer_flux(M₁, M₂, h₁, h₂, n, K₀)
    m = 1 - 1/n
    𝒮₁ = (Float64(M₁) / (1000h₁) - 0.05) / 0.35
    𝒮₂ = (Float64(M₂) / (1000h₂) - 0.05) / 0.35
    Π₁ = -expm1(-log(𝒮₁) / m)^(1/n)
    Π₂ = -expm1(-log(𝒮₂) / m)^(1/n)
    𝒮̄ = (𝒮₁ + 𝒮₂) / 2
    K = K₀ * sqrt(𝒮̄) * expm1(m * log1p(-𝒮̄^(1/m)))^2
    ℓ = (h₁ + h₂) / 2
    return 1000K * (Π₂ - Π₁ - ℓ) / ℓ
end

@testset "Nonlinear wetting of dry soil" begin
    for arch in test_architectures, FT in (Float32, Float64), n in (1.05, 1.2, 2.0),
        (M₁, M₂) in ((0.0, 157.5), (15.0, 157.5), (75.0, 35.0))
        grid = RectilinearGrid(arch, FT; size = 1, z = (-1, 0), topology = (Flat, Flat, Bounded))
        hydrology = VariablySaturatedHydrology(FT; slab_depth = (0.3, 0.7), porosity = 0.4,
            residual_liquid_fraction = 0.05, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 1, pore_size_uniformity = n),
            hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-5,
                                                               pore_size_uniformity = n, water_viscosity = nothing))
        land = SlabLand(grid; hydrology)
        set!(land; T = 293, M = M₁, water_storage_2 = M₂)
        time_step!(land, 60)
        upper = scalar(land.water_storage)
        lower = scalar(land.prognostic.water_storage_2)
        flux = scalar(land.diagnostics.interlayer_liquid_flux_1)
        tolerance = FT === Float32 ? 5e-3 : 1e-9
        @test 15 < upper < 120
        @test upper + lower ≈ M₁ + M₂ rtol = 8eps(FT)
        @test upper - M₁ ≈ 60flux rtol = tolerance
        @test flux ≈ reference_interlayer_flux(upper, lower, 0.3, 0.7, n, 1e-5) rtol = tolerance
    end
end

@testset "Coupled nonlinear Darcy balance across three layers" begin
    for arch in test_architectures, FT in (Float32, Float64)
        grid = RectilinearGrid(arch, FT; size = 1, z = (-1, 0), topology = (Flat, Flat, Bounded))
        hydrology = VariablySaturatedHydrology(FT; slab_depth = (0.3, 0.3, 0.4), porosity = 0.4,
            residual_liquid_fraction = 0.05, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 1, pore_size_uniformity = 2),
            hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-3,
                                                               pore_size_uniformity = 2, water_viscosity = nothing))
        land = SlabLand(grid; hydrology)
        set!(land; T = 293, M = 15, water_storage_2 = 100, water_storage_3 = 40)
        time_step!(land, 600)
        M₁, M₂, M₃ = scalar(land.water_storage), scalar(land.prognostic.water_storage_2), scalar(land.prognostic.water_storage_3)
        J₁, J₂ = scalar(land.diagnostics.interlayer_liquid_flux_1), scalar(land.diagnostics.interlayer_liquid_flux_2)
        tolerance = FT === Float32 ? 1e-4 : 1e-10
        @test M₁ + M₂ + M₃ ≈ 155 rtol = 8eps(FT)
        @test M₁ - 15 ≈ 600J₁ rtol = tolerance
        @test M₂ - 100 ≈ 600 * (J₂ - J₁) rtol = tolerance
        @test M₃ - 40 ≈ -600J₂ rtol = tolerance
        @test J₁ ≈ reference_interlayer_flux(M₁, M₂, 0.3, 0.3, 2, 1e-3) rtol = tolerance
        @test J₂ ≈ reference_interlayer_flux(M₂, M₃, 0.3, 0.4, 2, 1e-3) rtol = tolerance
    end
end

@testset "VariablySaturatedHydrology diagnostics" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # hˡᵃ = 1 m, ρˡ = 1000, ν = 0.4, hˢˢ = 1000, θʳ = 0 ⇒ M⁺ = 400 kg/m².
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            residual_liquid_fraction = 0.0,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = NoRunoff(),
        )
        land = SlabLand(grid; hydrology)

        # Test cases: M = 200, 400, 401.
        set!(land; M = 200.0)
        @test only(Array(interior(land.saturation))) ≈ 0.5

        set!(land; M = 400.0)
        @test only(Array(interior(land.saturation))) ≈ 1.0

        set!(land; M = 401.0)   # over-saturated; saturation clamps at 1
        @test only(Array(interior(land.saturation))) ≈ 1.0
    end
end

@testset "VariablySaturatedHydrology conservation" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # No-flux: M(t) = M₀.
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = NoRunoff(),
        )
        land = SlabLand(grid; hydrology)
        set!(land; M = 200.0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)

        Δt = 100.0
        for _ in 1:10
            time_step!(land, Δt)
        end
        @test only(Array(interior(land.water_storage))) ≈ 200.0

        # Constant evaporation: dM/dt = -Jᵛ = -0.01.
        set!(land; M = 200.0)
        fill!(land.fluxes.vapor_flux, 0.01)
        time_step!(land, 1000.0)
        @test only(Array(interior(land.water_storage))) ≈ 190.0

        # Constant precip below infiltration capacity: dM/dt = +Pˡ.
        hydrology_capped = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = InfiltrationCapacityRunoff(infiltration_capacity = 7.0),
        )
        land_capped = SlabLand(grid; hydrology = hydrology_capped)
        set!(land_capped; M = 0.0)
        fill!(land_capped.fluxes.vapor_flux, 0.0)
        fill!(land_capped.fluxes.liquid_precipitation_flux, 5.0)  # below capacity 7
        time_step!(land_capped, 1.0)
        @test only(Array(interior(land_capped.water_storage))) ≈ 5.0
        @test only(Array(interior(land_capped.diagnostics.surface_runoff))) ≈ 0.0

        # Precip above capacity: surface runoff = Pˡ - capacity.
        set!(land_capped; M = 0.0)
        fill!(land_capped.fluxes.liquid_precipitation_flux, 10.0)  # above capacity 7
        time_step!(land_capped, 1.0)
        @test only(Array(interior(land_capped.water_storage))) ≈ 7.0
        @test only(Array(interior(land_capped.diagnostics.surface_runoff))) ≈ 3.0

        # A per-cell capacity field caps the same way.
        capacity = Field{Center, Center, Nothing}(grid)
        set!(capacity, 7)
        hydrology_field_capped = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1,
            porosity = 0.4,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = InfiltrationCapacityRunoff(infiltration_capacity = capacity),
        )
        land_field_capped = SlabLand(grid; hydrology = hydrology_field_capped)
        set!(land_field_capped; M = 0)
        fill!(land_field_capped.fluxes.vapor_flux, 0)
        fill!(land_field_capped.fluxes.liquid_precipitation_flux, 10)
        time_step!(land_field_capped, 1)
        @test maximum(land_field_capped.water_storage) ≈ 7
        @test maximum(land_field_capped.diagnostics.surface_runoff) ≈ 3

        # Free drainage: dM/dt = -ρˡ K_b. At full saturation K = K_sat Θ(T), where Θ is
        # the viscosity correction, so the rate carries the slab temperature.
        hydrology_drain = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
            deep_liquid_flux = FreeDrainageFlux(),
            runoff = NoRunoff(),
        )
        land_drain = SlabLand(grid; hydrology = hydrology_drain)
        set!(land_drain; T = 303.0, M = 400.0)  # fully saturated, warmer than the reference
        fill!(land_drain.fluxes.vapor_flux, 0)
        fill!(land_drain.fluxes.liquid_precipitation_flux, 0)
        viscosity = hydrology_drain.hydraulic_conductivity.water_viscosity
        Θ = viscosity_correction(viscosity, 303.0)
        time_step!(land_drain, 100)
        expected = 400 - 100 * 1000 * 1e-6 * Θ
        @test only(Array(interior(land_drain.water_storage))) ≈ expected atol = 1e-5

        # Rain on a column at pore capacity runs off: M stays at M⁺ less one step
        # of drainage, and the runoff is the rain minus the drainage ρˡ K₀ = 1e-3.
        hydrology_saturating = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1,
            porosity = 0.4,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2),
            deep_liquid_flux = FreeDrainageFlux(),
            runoff = InfiltrationCapacityRunoff(infiltration_capacity = 7),
        )
        land_saturating = SlabLand(grid; hydrology = hydrology_saturating)
        set!(land_saturating; T = 293, M = 399)
        fill!(land_saturating.fluxes.vapor_flux, 0)
        fill!(land_saturating.fluxes.liquid_precipitation_flux, 5)  # below capacity 7
        for _ in 1:20
            time_step!(land_saturating, 1)
        end
        M = maximum(land_saturating.water_storage)
        @test M ≤ 400
        @test M ≈ 400 - 1e-3 atol = 1e-2
        @test maximum(land_saturating.diagnostics.surface_runoff) ≈ 5 - 1e-3 atol = 1e-2
    end
end

@testset "VariablySaturatedHydrology time-varying deep head" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        Πᵈ = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 200.0])
        interior(Πᵈ[1]) .= -1.0
        interior(Πᵈ[2]) .= -3.0

        storage(deep_pressure_head) = begin
            hydrology = VariablySaturatedHydrology(eltype(grid);
                slab_depth = 1.0,
                porosity = 0.4,
                storage_height = 1000,
                retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
                hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
                deep_liquid_flux = DarcyDeepLiquidFlux(exchange_length = 0.5),
                deep_pressure_head,
                runoff = NoRunoff(),
            )
            land = SlabLand(grid; hydrology)
            set!(land; M = 300.0)
            land.clock.time = 90.0
            time_step!(land, 10.0)
            only(Array(interior(land.water_storage)))
        end

        # The clock ticks before the hydrology step, so the series is read at t = 100 s,
        # where it interpolates to −2 m: the step must match the constant head −2 m.
        @test storage(Πᵈ) ≈ storage(-2.0)
        @test storage(Πᵈ) != storage(-1.0)
    end
end

@testset "VariablySaturatedHydrology surface water store" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        ponded_hydrology(drainage_time_scale; infiltration_capacity = 1e-3) =
            VariablySaturatedHydrology(eltype(grid);
                slab_depth = 1.0,
                porosity = 0.4,
                storage_height = 1000,
                retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
                hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
                runoff = InfiltrationCapacityRunoff(; infiltration_capacity, drainage_time_scale))

        value(field) = only(Array(interior(field)))

        # Rain at five times the capacity: infiltration runs at the cap, the excess ponds,
        # and rain = infiltration + storage + runoff to machine precision.
        land = SlabLand(grid; hydrology = ponded_hydrology(3600))
        set!(land; M = 100.0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 5e-3)
        Δt = 60.0
        ∫R = 0.0
        for _ in 1:200
            time_step!(land, Δt)
            ∫R += value(land.diagnostics.surface_runoff) * Δt
        end
        ΔM = value(land.water_storage) - 100
        S  = value(land.prognostic.surface_water_storage)
        @test ΔM ≈ 200 * Δt * 1e-3
        @test S > 0
        @test 200 * Δt * 5e-3 ≈ ΔM + S + ∫R rtol=1e-12

        # Zero capacity and no rain: the store drains as S₀ e^{−t/τ}.
        land = SlabLand(grid; hydrology = ponded_hydrology(1800; infiltration_capacity = 0))
        set!(land; M = 100.0, surface_water_storage = 10.0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)
        for _ in 1:12
            time_step!(land, 150.0)
        end
        @test value(land.prognostic.surface_water_storage) ≈ 10 * exp(-1)
        @test value(land.water_storage) == 100

        # Under a canopy interception store, over-capacity throughfall still ponds.
        hydrology = InterceptingHydrology(eltype(grid); soil = ponded_hydrology(3600), leaf_area_index = 3.0)
        land = SlabLand(grid; hydrology)
        set!(land; M = 100.0)
        for _ in 1:20
            fill!(land.fluxes.liquid_precipitation_flux, 5e-3)
            time_step!(land, Δt)
        end
        @test value(land.prognostic.canopy_water_storage) > 0
        @test value(land.prognostic.surface_water_storage) > 0
    end
end

@testset "Layered VariablySaturatedHydrology" begin
    for arch in test_architectures
        FT = Float64
        grid = RectilinearGrid(arch, FT; size = 1, x = (0, 1), y = (0, 1), z = (-1, 0),
                               topology = (Flat, Flat, Bounded))
        value(field) = only(Array(interior(field)))

        ν, θʳ, h₁, h₂ = 0.4, 0.05, 0.3, 0.7
        ℓ = (h₁ + h₂) / 2
        retention = VanGenuchtenRetention(FT; inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0)
        head(𝒮) = NumericalEarth.Lands.pressure_head(1, 1, grid, retention, 𝒮)
        effective_saturation(M, h) = (M / (1000h) - θʳ) / (ν - θʳ)

        layered(; matching_point_conductivity = 1e-5,
                  hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity, pore_size_uniformity = 2.0),
                  deep_liquid_flux = NoDeepLiquidFlux(), root_fraction = nothing) =
            VariablySaturatedHydrology(FT; slab_depth = (h₁, h₂), porosity = ν,
                                       residual_liquid_fraction = θʳ, storage_height = 1000,
                                       retention_curve = retention, hydraulic_conductivity,
                                       deep_liquid_flux, root_fraction, runoff = NoRunoff())
        column(; kw...) = SlabLand(grid; hydrology = layered(; kw...))
        quiet!(land) = foreach(flux -> fill!(flux, 0), land.fluxes)

        land = column()
        @test :interlayer_liquid_flux_1 in keys(land.diagnostics)
        @test value(land.prognostic.water_storage_2) ≈ 1000h₂ * (ν + θʳ) / 2   # half effective saturation
        @test_throws ArgumentError layered(root_fraction = (1.0,))
        @test_throws ArgumentError layered(root_fraction = (0.7, 0.7))

        # A rain pulse, steady evaporation and free drainage: the column's budget closes.
        land = column(deep_liquid_flux = FreeDrainageFlux(FT))
        set!(land; T = 293.0, M = 75.0, water_storage_2 = 105.0)
        Δt, Jᵛ = 600.0, 1e-5
        fill!(land.fluxes.vapor_flux, Jᵛ)
        ∫P = ∫Jᵈ = 0.0
        for n in 1:600
            P = n ≤ 100 ? 2e-4 : 0.0
            fill!(land.fluxes.liquid_precipitation_flux, P)
            time_step!(land, Δt)
            ∫P  += P * Δt
            ∫Jᵈ += value(land.diagnostics.deep_liquid_flux) * Δt
        end
        ΔM = value(land.water_storage) - 75 + value(land.prognostic.water_storage_2) - 105
        @test ∫P ≈ ΔM + 600Δt * Jᵛ - ∫Jᵈ rtol=1e-12
        @test ∫Jᵈ < 0

        # A wet slab over a drier layer with a closed bottom relaxes to equal hydraulic head, Π₂ − Π₁ = ℓ.
        land = column()
        set!(land; T = 293.0, M = 75.0, water_storage_2 = 105.0)
        quiet!(land)
        for _ in 1:8640
            time_step!(land, 600.0)
        end
        M₁, M₂ = value(land.water_storage), value(land.prognostic.water_storage_2)
        @test M₂ > 105
        @test M₁ + M₂ ≈ 180 rtol=1e-12
        @test head(effective_saturation(M₂, h₂)) - head(effective_saturation(M₁, h₁)) ≈ ℓ atol=1e-3

        # The implicit exchange survives sandy conductivity at ten-minute steps, where the explicit one collapses.
        land = column(matching_point_conductivity = 1e-3)
        set!(land; T = 293.0, M = 75.0, water_storage_2 = 105.0)
        quiet!(land)
        for _ in 1:50
            time_step!(land, 600.0)
        end
        M₁, M₂ = value(land.water_storage), value(land.prognostic.water_storage_2)
        @test M₁ > 1000h₁ * θʳ
        @test M₁ + M₂ ≈ 180 rtol=1e-12
        @test head(effective_saturation(M₂, h₂)) - head(effective_saturation(M₁, h₁)) ≈ ℓ atol=1e-2

        # A top layer at residual wets from the moist layer below, without overshooting its pore volume.
        land = column()
        set!(land; T = 293.0, M = 1000h₁ * θʳ, water_storage_2 = 1000h₂ * (ν + θʳ) / 2)
        quiet!(land)
        for _ in 1:50
            time_step!(land, 600.0)
        end
        M₁, M₂ = value(land.water_storage), value(land.prognostic.water_storage_2)
        @test 1000h₁ * θʳ < M₁ < 1000h₁ * ν
        @test M₁ + M₂ ≈ 1000h₁ * θʳ + 1000h₂ * (ν + θʳ) / 2 rtol=1e-12

        # An impermeable layer passes nothing: the exchange sees the half-layers' conductivities in series.
        land = column(hydraulic_conductivity = (VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-5, pore_size_uniformity = 2.0),
                                                VanGenuchtenConductivity(FT; matching_point_conductivity = 0, pore_size_uniformity = 2.0)))
        set!(land; T = 293.0, M = 1000ν * h₁, water_storage_2 = 1000ν * h₂)
        quiet!(land)
        time_step!(land, 1.0)
        @test value(land.diagnostics.interlayer_liquid_flux_1) == 0
        @test value(land.prognostic.water_storage_2) == 1000ν * h₂

        # Rain fills the top layer to its pore volume and runs off; the layer below builds positive head.
        land = column()
        set!(land; T = 293.0, M = 75.0, water_storage_2 = 1000ν * h₂)
        fill!(land.fluxes.liquid_precipitation_flux, 2e-4)
        for _ in 1:1000
            time_step!(land, 600.0)
        end
        quiet!(land)
        for _ in 1:1000
            time_step!(land, 600.0)
        end
        M₁, M₂ = value(land.water_storage), value(land.prognostic.water_storage_2)
        Π₁ = (M₁ / (1000h₁) - ν) * 1000
        Π₂ = (M₂ / (1000h₂) - ν) * 1000
        @test M₁ ≈ 1000ν * h₁ && M₂ > 1000ν * h₂
        @test Π₂ - Π₁ ≈ ℓ atol=1e-3

        # Roots draw the transpiration from each layer in proportion to rₖ 𝒮ₖ and the soil
        # evaporation from the top layer; the surface reads the top layer's 𝒮₁ and the canopy
        # stress the root-weighted Σ rₖ 𝒮ₖ.
        land = column(matching_point_conductivity = 1e-15, root_fraction = (0.5, 0.5))
        set!(land; T = 293.0, M = 30.0, water_storage_2 = 245.0)
        𝒮₁, 𝒮₂ = effective_saturation(30.0, h₁), effective_saturation(245.0, h₂)
        @test value(land.saturation) ≈ 𝒮₁
        @test value(land.diagnostics.root_zone_saturation) ≈ (𝒮₁ + 𝒮₂) / 2
        @test value(land.diagnostics.column_water_storage) ≈ 275.0
        quiet!(land)
        fill!(land.fluxes.vapor_flux, 1e-5)
        fill!(land.fluxes.transpiration, 1e-5)
        time_step!(land, 600.0)
        @test 30 - value(land.water_storage) ≈ 600 * 1e-5 * 𝒮₁ / (𝒮₁ + 𝒮₂) rtol=1e-6
        @test 245 - value(land.prognostic.water_storage_2) ≈ 600 * 1e-5 * 𝒮₂ / (𝒮₁ + 𝒮₂) rtol=1e-6

        set!(land; M = 30.0, water_storage_2 = 245.0)
        fill!(land.fluxes.transpiration, 0)
        time_step!(land, 600.0)
        @test 30 - value(land.water_storage) ≈ 600 * 1e-5 rtol=1e-6
        @test value(land.prognostic.water_storage_2) ≈ 245 rtol=1e-12

        # The whole column's water sets the thermal inertia of its single temperature.
        land = column(matching_point_conductivity = 1e-15)
        set!(land; T = 293.0, M = 30.0, water_storage_2 = 245.0)
        quiet!(land)
        fill!(land.fluxes.surface_energy_flux, 100.0)
        time_step!(land, 600.0)
        cᵈʳʸ, cˡ = land.energy.dry_heat_capacity, land.energy.liquid_heat_capacity
        @test value(land.temperature) ≈ 293 - 100 * 600 / (cᵈʳʸ + cˡ * 275) rtol = 1e-10

        # Per-layer conductivity: the bottom layer drains with its own curve.
        land = column(deep_liquid_flux = FreeDrainageFlux(FT),
                      hydraulic_conductivity = (VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-5, pore_size_uniformity = 2.0),
                                                VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0)))
        set!(land; T = 293.0, M = 75.0, water_storage_2 = 1000ν * h₂)
        quiet!(land)
        time_step!(land, 1.0)
        @test value(land.diagnostics.deep_liquid_flux) ≈ -1e-3 rtol=1e-6

        # A Darcy exchange with a prescribed head is implicit too: one slab, a step far beyond the
        # exchange time scale, and the storage still relaxes to Π = Πᵈ − ℓ without overshooting.
        soil = VariablySaturatedHydrology(FT; slab_depth = h₁, porosity = ν, residual_liquid_fraction = θʳ,
                                          storage_height = 1000, retention_curve = retention,
                                          hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-3, pore_size_uniformity = 2.0),
                                          deep_liquid_flux = DarcyDeepLiquidFlux(FT; exchange_length = ℓ),
                                          deep_pressure_head = -1.0, runoff = NoRunoff())
        land = SlabLand(grid; hydrology = soil)
        set!(land; T = 293.0, M = 75.0)
        quiet!(land)
        Ms = [value(land.water_storage)]
        for _ in 1:20
            time_step!(land, 3600.0)
            push!(Ms, value(land.water_storage))
        end
        Mᵉ = 1000h₁ * (θʳ + (ν - θʳ) * (1 + (1 + ℓ)^2)^(-1/2))
        @test all(Ms .≥ Mᵉ - 1e-3)
        @test Ms[end] ≈ Mᵉ atol=1e-3

        # Under a canopy interception store the deep layer keeps its declaration and initial state.
        land = SlabLand(grid; hydrology = InterceptingHydrology(FT; soil = layered(), leaf_area_index = 3.0))
        @test :water_storage_2 in keys(land.prognostic) && :canopy_water_storage in keys(land.prognostic)
        @test value(land.prognostic.water_storage_2) ≈ 1000h₂ * (ν + θʳ) / 2
    end
end
