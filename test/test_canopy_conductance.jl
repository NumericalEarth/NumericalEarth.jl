include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!
using Oceananigans.TimeSteppers: update_state!, time_step!
using Thermodynamics
using NumericalEarth.Lands: SaturatedSurface
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    RubiscoLightLimitedPhotosynthesis, MedlynConductance, JarvisConductance,
    CanopyConductanceHumidity, BulkHumidity,
    InteractiveAbsorbedPAR, absorbed_par_value,
    net_assimilation, stomatal_conductance, temperature_response

@testset "FvCB–Medlyn canopy physics" begin
    for FT in (Float32, Float64)
        photo, cond = RubiscoLightLimitedPhotosynthesis(FT), MedlynConductance(FT)
        Tˡᵉᵃᶠ, p, ca = FT(298), FT(101325), FT(40)
        A(ci, APAR) = net_assimilation(photo, FT(ci), FT(APAR), Tˡᵉᵃᶠ, p, FT(1))
        A_dim, A_light, A_sat = A(28, 1e-4), A(28, 1e-3), A(28, 1e-2)
        @test A_dim < A_light ≤ A_sat
        @test A_sat - A_light < A_light - A_dim   # saturating, not linear
        @test A(35, 1e-3) > A(15, 1e-3)

        gs_ref, An_ref, ci_ref = @inferred stomatal_conductance(cond, photo, FT(1e-3), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(1))
        gₛ(APAR, VPD, β) = stomatal_conductance(cond, photo, FT(APAR), FT(VPD), Tˡᵉᵃᶠ, ca, p, FT(β))[1]
        @test gs_ref isa FT && gs_ref > cond.g0
        @test An_ref > 0
        @test photo.Γ★25 < ci_ref < ca
        @test ci_ref ≈ ca * cond.g1 / (cond.g1 + sqrt(FT(1000)))
        @test gₛ(1e-3, 3000, 1) < gs_ref
        @test gₛ(1e-3, 1000, 0.3) < gs_ref
        @test gₛ(0, 1000, 1) ≈ cond.g0   # dark: Aₙ = −Rᵈ < 0 is floored, leaving g₀
    end
end

@testset "Standalone canopy transpiration draws from the root zone" begin
    for arch in test_architectures, land_temperature in (300.0, 280.0)
        hydrology = VariablySaturatedHydrology(; slab_depth = (0.3, 0.7), porosity = 0.4,
            residual_liquid_fraction = 0.05, storage_height = 1000, root_fraction = (0.5, 0.5),
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 0, pore_size_uniformity = 2))
        model = coupled_land_model(arch; hydrology, land_temperature, water = 30.0,
            air_temperature = 290.0, air_specific_humidity = 0.01, wind = 5.0,
            atmosphere_land_interface_specific_humidity = CanopyConductanceHumidity(leaf_area_index = 3))
        land = model.land
        set!(land; M = 30, water_storage_2 = 245)
        update_state!(model)
        time_step!(model, 60)

        vapor, transpiration = scalar(land.fluxes.vapor_flux), scalar(land.fluxes.transpiration)
        @test (land_temperature - 290) * vapor > 0
        @test transpiration == max(0, vapor)
        # Effective saturations are 1/7 and 6/7; equal root fractions give the same uptake shares.
        @test scalar(land.water_storage) ≈ 30 - 60 * (vapor - transpiration + transpiration / 7)
        @test scalar(land.prognostic.water_storage_2) ≈ 245 - 60 * transpiration * 6 / 7
    end
end

@testset "Peaked Arrhenius + Heskel respiration" begin
    for FT in (Float32, Float64)
        Vcmax = PeakedArrheniusParameters(FT; activation_energy = 71513, entropy = 649, deactivation_energy = 200000)
        Jmax = PeakedArrheniusParameters(FT; activation_energy = 49884, entropy = 646, deactivation_energy = 200000)
        @test temperature_response(FT(298.15), Vcmax) ≈ 1
        @test temperature_response(FT(298.15), HeskelParameters(FT)) ≈ 1

        Ts = FT(273):FT(1):FT(323)
        vc = [temperature_response(T, Vcmax) for T in Ts]
        jm = [temperature_response(T, Jmax) for T in Ts]
        @test vc[end] < maximum(vc)
        @test jm[end] < maximum(jm)
        @test Ts[argmax(vc)] ≥ Ts[argmax(jm)]
        @test all(diff([temperature_response(T, HeskelParameters(FT)) for T in FT(273):FT(1):FT(318)]) .> 0)

        photo = RubiscoLightLimitedPhotosynthesis(FT)
        Aₙ(T) = net_assimilation(photo, FT(28), FT(8e-4), T, FT(101325), FT(1))
        @test Aₙ(FT(318)) < maximum(Aₙ, FT(273):FT(1):FT(318))
        @test (@inferred net_assimilation(photo, FT(28), FT(8e-4), FT(298), FT(101325), FT(1))) isa FT
        dAdT(T) = (Aₙ(T + FT(0.05)) - Aₙ(T - FT(0.05))) / FT(0.1)
        @test dAdT(FT(288)) > 0
        @test dAdT(FT(313)) < 0
    end
end

@testset "Jarvis stomatal conductance" begin
    for FT in (Float32, Float64)
        jarv = JarvisConductance(FT)
        Tˡᵉᵃᶠ, ca, p = FT(298.15), FT(40), FT(101325)
        gs, An, ci = @inferred stomatal_conductance(jarv, nothing, FT(2e-4), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(1))
        @test gs isa FT && gs > 0
        @test An == 0 && ci == 0

        gₛ(APAR, VPD, T = Tˡᵉᵃᶠ, β = 1) = stomatal_conductance(jarv, nothing, FT(APAR), FT(VPD), FT(T), ca, p, FT(β))[1]
        T₀ = jarv.optimal_temperature
        @test gₛ(1e-3, 1000) > gₛ(1e-5, 1000)
        @test gₛ(1e6, 1000) ≈ gₛ(1e5, 1000) rtol=1e-3
        @test gₛ(1e-3, 3000) < gₛ(1e-3, 1000)
        @test gₛ(1e-3, 1000, T₀) > max(gₛ(1e-3, 1000, T₀ + 15), gₛ(1e-3, 1000, T₀ - 15))
        @test gₛ(1e-3, 1000, Tˡᵉᵃᶠ, 0.5) ≈ gₛ(1e-3, 1000) / 2
        @test gₛ(1e-3, 1000, Tˡᵉᵃᶠ, 0) == 0
    end
end

@testset "Absorbed PAR (prescribed vs interactive)" begin
    for FT in (Float32, Float64)
        LAI = FT(2)
        @test absorbed_par_value(FT(4e-4), nothing, LAI, nothing) == FT(4e-4)

        inter = InteractiveAbsorbedPAR(FT)
        APAR(ℐꜜˢʷ) = absorbed_par_value(inter, (; ℐꜜˢʷ = -FT(ℐꜜˢʷ)), LAI, nothing)
        @test APAR(0) == 0
        @test 1e-4 < APAR(500) < 1e-3   # same order as the prescribed default
        @test APAR(1000) ≈ 2 * APAR(500)
        @test (@inferred absorbed_par_value(inter, (; ℐꜜˢʷ = -FT(500)), LAI, nothing)) isa FT
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
        @test -1e-10 ≤ canopy_latent_heat(absorbed_par = InteractiveAbsorbedPAR(FT)) < LE_canopy
    end
end
