include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Lands: viscosity_correction

function reference_interlayer_flux(M₁, M₂, h₁, h₂, n, K₀)
    m = 1 - 1/n
    𝒮₁, 𝒮₂ = (Float64(M₁) / (1000h₁) - 0.05) / 0.35, (Float64(M₂) / (1000h₂) - 0.05) / 0.35
    Π(𝒮) = -expm1(-log(𝒮) / m)^(1/n)
    𝒮̄, ℓ = (𝒮₁ + 𝒮₂) / 2, (h₁ + h₂) / 2
    K = K₀ * sqrt(𝒮̄) * expm1(m * log1p(-𝒮̄^(1/m)))^2
    return 1000K * (Π(𝒮₂) - Π(𝒮₁) - ℓ) / ℓ
end

van_genuchten_conductivity(FT, K₀, n; kw...) = VanGenuchtenConductivity(FT; matching_point_conductivity = K₀, pore_size_uniformity = n, kw...)

van_genuchten_hydrology(FT, slab_depth; n = 2, K₀ = 1e-5, kw...) =
    VariablySaturatedHydrology(FT; slab_depth, porosity = 0.4, residual_liquid_fraction = 0.05, storage_height = 1000,
                               retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 1, pore_size_uniformity = n),
                               hydraulic_conductivity = van_genuchten_conductivity(FT, K₀, n), kw...)

function van_genuchten_column(arch, FT, slab_depth, n, K₀)
    grid = RectilinearGrid(arch, FT; size = 1, z = (-1, 0), topology = (Flat, Flat, Bounded))
    conductivity = van_genuchten_conductivity(FT, K₀, n; water_viscosity = nothing)
    return SlabLand(grid; hydrology = van_genuchten_hydrology(FT, slab_depth; n, hydraulic_conductivity = conductivity))
end

@testset "Nonlinear Darcy exchange between van Genuchten layers" begin
    for arch in test_architectures, FT in (Float32, Float64)
        for n in (1.05, 1.2, 2.0), (M₁, M₂) in ((0.0, 157.5), (15.0, 157.5), (75.0, 35.0))
            land = van_genuchten_column(arch, FT, (0.3, 0.7), n, 1e-5)
            set!(land; T = 293, M = M₁, water_storage_2 = M₂)
            time_step!(land, 60)
            upper, lower = scalar(land.water_storage), scalar(land.prognostic.water_storage_2)
            J = scalar(land.diagnostics.interlayer_liquid_flux_1)
            tolerance = FT === Float32 ? 5e-3 : 1e-9
            @test 15 < upper < 120
            @test upper + lower ≈ M₁ + M₂ rtol = 8eps(FT)
            @test upper - M₁ ≈ 60J rtol = tolerance
            @test J ≈ reference_interlayer_flux(upper, lower, 0.3, 0.7, n, 1e-5) rtol = tolerance
        end

        # Dry soil near the residual fraction holds its water
        land = van_genuchten_column(arch, FT, 0.3, 1.2, 1e-5)
        set!(land; T = 293, M = FT(15.00002))
        time_step!(land, 60)
        @test scalar(land.water_storage) == FT(15.00002)

        land = van_genuchten_column(arch, FT, (0.3, 0.3, 0.4), 2, 1e-3)
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

quiet!(land) = foreach(flux -> fill!(flux, 0), land.fluxes)
advance!(land, N, Δt) = foreach(_ -> time_step!(land, Δt), 1:N)

@testset "VariablySaturatedHydrology surface water store" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = 1, x = (0, 1), y = (0, 1), z = (-1, 0), topology = (Flat, Flat, Bounded))
        ponded(drainage_time_scale; infiltration_capacity = 1e-3) =
            van_genuchten_hydrology(eltype(grid), 1.0; K₀ = 1e-6, residual_liquid_fraction = 0,
                                    runoff = InfiltrationCapacityRunoff(; infiltration_capacity, drainage_time_scale))

        # Rain at five times the infiltration capacity
        land = SlabLand(grid; hydrology = ponded(3600))
        set!(land; M = 100.0)
        quiet!(land)
        fill!(land.fluxes.liquid_precipitation_flux, 5e-3)
        ∫R = 0.0
        for _ in 1:200
            time_step!(land, 60.0)
            ∫R += scalar(land.diagnostics.surface_runoff) * 60
        end
        ΔM, S = scalar(land.water_storage) - 100, scalar(land.prognostic.surface_water_storage)
        @test ΔM ≈ 200 * 60 * 1e-3
        @test S > 0
        @test 200 * 60 * 5e-3 ≈ ΔM + S + ∫R rtol=1e-12

        # Zero capacity and no rain: S₀ e^{−t/τ}
        land = SlabLand(grid; hydrology = ponded(1800; infiltration_capacity = 0))
        set!(land; M = 100.0, surface_water_storage = 10.0)
        quiet!(land)
        advance!(land, 12, 150.0)
        @test scalar(land.prognostic.surface_water_storage) ≈ 10 * exp(-1)
        @test scalar(land.water_storage) == 100

        land = SlabLand(grid; hydrology = InterceptingHydrology(eltype(grid); soil = ponded(3600), leaf_area_index = 3.0))
        set!(land; M = 100.0)
        for _ in 1:20
            fill!(land.fluxes.liquid_precipitation_flux, 5e-3)
            time_step!(land, 60.0)
        end
        @test scalar(land.prognostic.canopy_water_storage) > 0
        @test scalar(land.prognostic.surface_water_storage) > 0
    end
end

@testset "Layered VariablySaturatedHydrology" begin
    for arch in test_architectures
        FT = Float64
        grid = RectilinearGrid(arch, FT; size = 1, x = (0, 1), y = (0, 1), z = (-1, 0), topology = (Flat, Flat, Bounded))
        ν, θʳ, h₁, h₂ = 0.4, 0.05, 0.3, 0.7
        ℓ = (h₁ + h₂) / 2
        half_saturated = 1000h₂ * (ν + θʳ) / 2
        conductivity(K₀) = van_genuchten_conductivity(FT, K₀, 2)
        effective_saturation(M, h) = (M / (1000h) - θʳ) / (ν - θʳ)
        head(M, h) = NumericalEarth.Lands.pressure_head(1, 1, grid, VanGenuchtenRetention(FT; inverse_air_entry_head = 1, pore_size_uniformity = 2), effective_saturation(M, h))

        layered(; kw...) = van_genuchten_hydrology(FT, (h₁, h₂); kw...)
        function column(M₁, M₂; kw...)
            land = SlabLand(grid; hydrology = layered(; kw...))
            set!(land; T = 293.0, M = M₁, water_storage_2 = M₂)
            quiet!(land)
            return land
        end
        storages(land) = (scalar(land.water_storage), scalar(land.prognostic.water_storage_2))

        land = SlabLand(grid; hydrology = layered())
        @test :interlayer_liquid_flux_1 in keys(land.diagnostics)
        @test scalar(land.prognostic.water_storage_2) ≈ half_saturated
        @test_throws ArgumentError layered(root_fraction = (1.0,))
        @test_throws ArgumentError layered(root_fraction = (0.7, 0.7))

        # Budget under a rain pulse, steady evaporation and free drainage
        land = column(75.0, 105.0; deep_liquid_flux = FreeDrainageFlux(FT))
        Δt, Jᵛ = 600.0, 1e-5
        fill!(land.fluxes.vapor_flux, Jᵛ)
        ∫P = ∫Jᵈ = 0.0
        for n in 1:600
            P = n ≤ 100 ? 2e-4 : 0.0
            fill!(land.fluxes.liquid_precipitation_flux, P)
            time_step!(land, Δt)
            ∫P  += P * Δt
            ∫Jᵈ += scalar(land.diagnostics.deep_liquid_flux) * Δt
        end
        @test ∫P ≈ sum(storages(land)) - 180 + 600Δt * Jᵛ - ∫Jᵈ rtol=1e-12
        @test ∫Jᵈ < 0

        # A closed bottom relaxes to equal hydraulic head, Π₂ − Π₁ = ℓ, also with sandy conductivity
        for (K₀, N, tolerance) in ((1e-5, 8640, 1e-3), (1e-3, 50, 1e-2))
            land = column(75.0, 105.0; K₀)
            advance!(land, N, 600.0)
            M₁, M₂ = storages(land)
            @test M₂ > 105 && M₁ > 1000h₁ * θʳ
            @test M₁ + M₂ ≈ 180 rtol=1e-12
            @test head(M₂, h₂) - head(M₁, h₁) ≈ ℓ atol=tolerance
        end

        # A top layer at residual wets from below without overshooting its pore volume
        land = column(1000h₁ * θʳ, half_saturated)
        advance!(land, 50, 600.0)
        M₁, M₂ = storages(land)
        @test 1000h₁ * θʳ < M₁ < 1000h₁ * ν
        @test M₁ + M₂ ≈ 1000h₁ * θʳ + half_saturated rtol=1e-12

        land = column(1000ν * h₁, 1000ν * h₂; hydraulic_conductivity = (conductivity(1e-5), conductivity(0)))
        time_step!(land, 1.0)
        @test scalar(land.diagnostics.interlayer_liquid_flux_1) == 0
        @test scalar(land.prognostic.water_storage_2) == 1000ν * h₂

        # Rain fills the top layer and runs off; above saturation Π = M/h − 1000ν
        land = column(75.0, 1000ν * h₂)
        fill!(land.fluxes.liquid_precipitation_flux, 2e-4)
        advance!(land, 1000, 600.0)
        quiet!(land)
        advance!(land, 1000, 600.0)
        M₁, M₂ = storages(land)
        @test M₁ ≈ 1000ν * h₁ && M₂ > 1000ν * h₂
        @test M₂ / h₂ - M₁ / h₁ ≈ ℓ atol=1e-3

        # Transpiration draws from each layer ∝ rₖ 𝒮ₖ, soil evaporation from the top layer
        land = column(30.0, 245.0; K₀ = 1e-15, root_fraction = (0.5, 0.5))
        𝒮₁, 𝒮₂ = effective_saturation(30.0, h₁), effective_saturation(245.0, h₂)
        @test scalar(land.saturation) ≈ 𝒮₁
        @test scalar(land.diagnostics.root_zone_saturation) ≈ (𝒮₁ + 𝒮₂) / 2
        @test scalar(land.diagnostics.column_water_storage) ≈ 275.0
        fill!(land.fluxes.vapor_flux, 1e-5)
        fill!(land.fluxes.transpiration, 1e-5)
        time_step!(land, 600.0)
        @test 30 - scalar(land.water_storage) ≈ 600 * 1e-5 * 𝒮₁ / (𝒮₁ + 𝒮₂) rtol=1e-6
        @test 245 - scalar(land.prognostic.water_storage_2) ≈ 600 * 1e-5 * 𝒮₂ / (𝒮₁ + 𝒮₂) rtol=1e-6
        set!(land; M = 30.0, water_storage_2 = 245.0)
        fill!(land.fluxes.transpiration, 0)
        time_step!(land, 600.0)
        @test 30 - scalar(land.water_storage) ≈ 600 * 1e-5 rtol=1e-6
        @test scalar(land.prognostic.water_storage_2) ≈ 245 rtol=1e-12

        # The whole column's water sets the thermal inertia
        land = column(30.0, 245.0; K₀ = 1e-15)
        fill!(land.fluxes.surface_energy_flux, 100.0)
        time_step!(land, 600.0)
        cᵈʳʸ, cˡ = land.energy.dry_heat_capacity, land.energy.liquid_heat_capacity
        @test scalar(land.temperature) ≈ 293 - 100 * 600 / (cᵈʳʸ + cˡ * 275) rtol = 1e-10

        # The bottom layer drains with its own conductivity curve
        land = column(75.0, 1000ν * h₂; deep_liquid_flux = FreeDrainageFlux(FT), hydraulic_conductivity = (conductivity(1e-5), conductivity(1e-6)))
        time_step!(land, 1.0)
        @test scalar(land.diagnostics.deep_liquid_flux) ≈ -1e-3 rtol=1e-6

        # Implicit exchange with a prescribed head relaxes to Π = Πᵈ − ℓ without overshooting
        land = SlabLand(grid; hydrology = van_genuchten_hydrology(FT, h₁; K₀ = 1e-3, deep_pressure_head = -1.0,
                                                                 deep_liquid_flux = DarcyDeepLiquidFlux(FT; exchange_length = ℓ)))
        set!(land; T = 293.0, M = 75.0)
        quiet!(land)
        Ms = [(time_step!(land, 3600.0); scalar(land.water_storage)) for _ in 1:20]
        Mᵉ = 1000h₁ * (θʳ + (ν - θʳ) * (1 + (1 + ℓ)^2)^(-1/2))
        @test all(Ms .≥ Mᵉ - 1e-3)
        @test Ms[end] ≈ Mᵉ atol=1e-3

        land = SlabLand(grid; hydrology = InterceptingHydrology(FT; soil = layered(), leaf_area_index = 3.0))
        @test (:water_storage_2, :canopy_water_storage) ⊆ keys(land.prognostic)
        @test scalar(land.prognostic.water_storage_2) ≈ half_saturated
    end
end
