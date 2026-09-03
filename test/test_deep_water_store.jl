include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Lands: SlabLand, SlabEnergy, VariablySaturatedHydrology,
    InterceptingHydrology, DeepWaterStore,
    VanGenuchtenRetention, VanGenuchtenConductivity,
    DarcyDeepLiquidFlux, FreeDrainageFlux, NoDeepLiquidFlux, NoRunoff, pressure_head

scalar(field) = Array(interior(field))[1, 1, 1]

store_test_grid(arch, FT) = RectilinearGrid(arch, FT; size = 1, x = (0, 1), y = (0, 1),
                                            z = (-1, 0), topology = (Flat, Flat, Bounded))

const ν, θʳ, hˡᵃ, hᵈ, ℓ = 0.4, 0.05, 0.3, 0.7, 0.5

soil_hydrology(grid, FT) = VariablySaturatedHydrology(FT;
    slab_depth = hˡᵃ, porosity = ν, residual_liquid_fraction = θʳ, storage_height = 0.1,
    retention_curve = VanGenuchtenRetention(FT; α = 1.0, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(FT; K_saturated = 1e-5, n = 2.0),
    deep_liquid_flux = DarcyDeepLiquidFlux(FT; exchange_length = ℓ),
    deep_pressure_head = CenterField(grid),
    runoff = NoRunoff())

function store_land(arch, FT; drainage = FreeDrainageFlux(FT))
    grid = store_test_grid(arch, FT)
    hydrology = DeepWaterStore(FT; soil = soil_hydrology(grid, FT), thickness = hᵈ, drainage)
    return SlabLand(grid; energy = SlabEnergy(FT), hydrology)
end

set_rain!(land, rain) = fill!(parent(land.fluxes.liquid_precipitation_flux), rain)
head(hydrology, 𝒮) = pressure_head(hydrology.retention_curve, 𝒮)

@testset "DeepWaterStore declarations & container" begin
    for arch in test_architectures
        land = store_land(arch, Float64)
        @test :deep_water_storage in keys(land.prognostic)
        @test :deep_drainage_flux in keys(land.diagnostics)
        @test :deep_liquid_flux in keys(land.diagnostics)   # from the wrapped soil
        @test :vapor_flux in keys(land.fluxes)
    end
end

@testset "Closed water budget across slab and store" begin
    for arch in test_architectures
        FT = Float64
        land = store_land(arch, FT)
        set!(land; T = 290.0, M = 75.0, deep_water_storage = 105.0)
        Δt, Jᵛ = 600.0, 1e-5
        fill!(parent(land.fluxes.vapor_flux), Jᵛ)

        M₀, Mᵈ₀ = scalar(land.water_storage), scalar(land.prognostic.deep_water_storage)
        ∫P = 0.0; ∫Jᵛ = 0.0; ∫Jᵈ = 0.0
        for n in 1:600
            Pⁿ = n ≤ 100 ? 2e-4 : 0.0
            set_rain!(land, Pⁿ)
            time_step!(land, Δt)
            ∫P  += Pⁿ * Δt
            ∫Jᵛ += Jᵛ * Δt
            ∫Jᵈ += scalar(land.diagnostics.deep_drainage_flux) * Δt
            @test scalar(land.water_storage) > 0
        end
        ΔM  = scalar(land.water_storage) - M₀
        ΔMᵈ = scalar(land.prognostic.deep_water_storage) - Mᵈ₀
        @test ∫P ≈ ΔM + ΔMᵈ + ∫Jᵛ - ∫Jᵈ rtol=1e-12
        @test ∫Jᵈ < 0
    end
end

@testset "Slab and store reach hydrostatic equilibrium" begin
    for arch in test_architectures
        FT = Float64
        land = store_land(arch, FT; drainage = NoDeepLiquidFlux())
        set!(land; T = 290.0, M = 75.0, deep_water_storage = 105.0)   # θ = 0.25 over a drier θᵈ = 0.15
        Δt = 600.0
        for _ in 1:8640   # 60 days
            set_rain!(land, 0.0)
            time_step!(land, Δt)
        end
        soil = land.hydrology.soil
        Mᵈ  = scalar(land.prognostic.deep_water_storage)
        𝒮ᵈ  = (Mᵈ / (1000hᵈ) - θʳ) / (ν - θʳ)
        Π   = head(soil, scalar(land.saturation))
        Πᵈ  = head(soil, 𝒮ᵈ)
        @test Mᵈ > 105.0                                          # the store took water from the wetter slab
        @test scalar(land.water_storage) + Mᵈ ≈ 180.0 rtol=1e-12  # and nothing left the column
        @test Πᵈ - Π ≈ ℓ atol=1e-3                                # equal hydraulic head: Πᵈ = Π + ℓ
    end
end

@testset "Composition with the canopy store" begin
    for arch in test_architectures
        FT = Float64
        grid = store_test_grid(arch, FT)
        hydrology = InterceptingHydrology(FT;
            soil = DeepWaterStore(FT; soil = soil_hydrology(grid, FT), thickness = hᵈ),
            leaf_area_index = 3.0)
        land = SlabLand(grid; energy = SlabEnergy(FT), hydrology)
        @test :deep_water_storage in keys(land.prognostic)
        @test :canopy_water_storage in keys(land.prognostic)

        set!(land; T = 290.0, M = 75.0, deep_water_storage = 105.0)
        for _ in 1:20
            set_rain!(land, 2e-4)
            time_step!(land, 600.0)
        end
        @test isfinite(scalar(land.prognostic.deep_water_storage))
        @test scalar(land.prognostic.deep_water_storage) > 105.0
    end
end
