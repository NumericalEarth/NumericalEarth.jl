include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!
using Oceananigans.TimeSteppers: update_state!, time_step!
using Thermodynamics
using NumericalEarth.Lands: SaturatedSurface
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    JarvisConductance,
    CanopyConductanceHumidity, BulkHumidity,
    AbsorbedPhotosyntheticallyActiveRadiation, absorbed_par_value,
    stomatal_conductance

@testset "Standalone canopy transpiration draws from the soil" begin
    for arch in test_architectures, land_temperature in (300.0, 280.0)
        hydrology = VariablySaturatedHydrology(; slab_depth = 1, porosity = 0.4,
            residual_liquid_fraction = 0.05, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 0, pore_size_uniformity = 2))
        model = coupled_land_model(arch; hydrology, land_temperature, water = 200.0,
            air_temperature = 290.0, air_specific_humidity = 0.01, wind = 5.0,
            atmosphere_land_interface_specific_humidity = CanopyConductanceHumidity(leaf_area_index = 3))
        land = model.land
        set!(land; M = 200)
        update_state!(model)
        vapor = scalar(land.fluxes.vapor_flux)
        time_step!(model, 60)
        @test (land_temperature - 290) * vapor > 0
        @test scalar(land.water_storage) ≈ 200 - 60 * vapor
    end
end

@testset "Jarvis stomatal conductance" begin
    for FT in (Float32, Float64)
        jarv = JarvisConductance(FT)
        Tˡᵉᵃ = FT(298.15)
        gs = @inferred stomatal_conductance(jarv, FT(2e-4), FT(1000), Tˡᵉᵃ, FT(1))
        @test gs isa FT && gs > 0

        gₛ(APAR, VPD, T = Tˡᵉᵃ, β = 1) = stomatal_conductance(jarv, FT(APAR), FT(VPD), FT(T), FT(β))
        T₀ = jarv.optimal_temperature
        @test gₛ(1e-3, 1000) > gₛ(1e-5, 1000)
        @test gₛ(1e6, 1000) ≈ gₛ(1e5, 1000) rtol=1e-3
        @test gₛ(1e-3, 3000) < gₛ(1e-3, 1000)
        @test gₛ(1e-3, 1000, T₀) > max(gₛ(1e-3, 1000, T₀ + 15), gₛ(1e-3, 1000, T₀ - 15))
        @test gₛ(1e-3, 1000, Tˡᵉᵃ, 0.5) ≈ gₛ(1e-3, 1000) / 2
        @test gₛ(1e-3, 1000, Tˡᵉᵃ, 0) == 0
    end
end

@testset "Absorbed PAR (prescribed vs interactive)" begin
    for FT in (Float32, Float64)
        λˡᵉᵃᶠ = FT(2)
        @test absorbed_par_value(FT(4e-4), nothing, λˡᵉᵃᶠ, nothing) == FT(4e-4)

        inter = AbsorbedPhotosyntheticallyActiveRadiation(FT)
        APAR(ℐꜜˢʷ) = absorbed_par_value(inter, (; ℐꜜˢʷ = -FT(ℐꜜˢʷ)), λˡᵉᵃᶠ, nothing)
        @test APAR(0) == 0
        @test 1e-4 < APAR(500) < 1e-3   # same order as the prescribed default
        @test APAR(1000) ≈ 2 * APAR(500)
        @test (@inferred absorbed_par_value(inter, (; ℐꜜˢʷ = -FT(500)), λˡᵉᵃᶠ, nothing)) isa FT
    end
end

@testset "CanopyConductanceHumidity coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        function latent_heat(q_formulation)
            model = coupled_land_model(arch; air_temperature = 290.0, air_specific_humidity = 0.006,
                                       wind = 5.0, land_temperature = 300.0, water = nothing,
                                       hydrology = SaturatedSurface(), radiation = nothing,
                                       atmosphere_land_interface_specific_humidity = q_formulation)
            return scalar(model.interfaces.atmosphere_land_interface.fluxes.latent_heat)
        end
        canopy_latent_heat(; kw...) = latent_heat(CanopyConductanceHumidity(FT; kw...))

        LE_bulk = latent_heat(BulkHumidity(Thermodynamics.Liquid()))
        LE_canopy = canopy_latent_heat()
        @test 0 < LE_canopy < LE_bulk
        @test abs(canopy_latent_heat(leaf_area_index = 4.0)) > abs(canopy_latent_heat(leaf_area_index = 0.5))
        @test abs(canopy_latent_heat(moisture_stress = 0.3)) < LE_canopy
        @test 0 < canopy_latent_heat(conductance = JarvisConductance(FT)) < LE_bulk
        # With radiation off the interactive canopy sees no light, so LE ≈ 0 up to roundoff.
        @test -1e-10 ≤ canopy_latent_heat(absorbed_par = AbsorbedPhotosyntheticallyActiveRadiation(FT)) < LE_canopy
    end
end
