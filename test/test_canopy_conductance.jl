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
#####
##### Photosynthesis + conductance physics (pure functions — no grid needed).
#####

@testset "FvCB–Medlyn canopy physics" begin
    for FT in (Float32, Float64)
        photo = RubiscoLightLimitedPhotosynthesis(FT)
        cond  = MedlynConductance(FT)

        Tˡᵉᵃᶠ   = FT(298)      # leaf temperature (K)
        p    = FT(101325)   # air pressure (Pa)
        ca   = FT(40)       # CO₂ partial pressure (Pa)
        ci   = FT(28)       # intercellular CO₂ (Pa)
        β    = FT(1)

        # Net assimilation rises with absorbed PAR and saturates (light limitation).
        A_dim   = net_assimilation(photo, ci, FT(1e-4), Tˡᵉᵃᶠ, p, β)
        A_light = net_assimilation(photo, ci, FT(1e-3), Tˡᵉᵃᶠ, p, β)
        A_sat   = net_assimilation(photo, ci, FT(1e-2), Tˡᵉᵃᶠ, p, β)
        @test A_light > A_dim
        @test A_sat ≥ A_light
        @test A_sat - A_light < A_light - A_dim   # saturating, not linear

        # ... and rises with intercellular CO₂.
        @test net_assimilation(photo, FT(35), FT(1e-3), Tˡᵉᵃᶠ, p, β) >
              net_assimilation(photo, FT(15), FT(1e-3), Tˡᵉᵃᶠ, p, β)

        # Coupled solve: light drives conductance up, VPD and moisture stress down.
        gs_ref, An_ref, ci_ref = stomatal_conductance(cond, photo, FT(1e-3), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(1))
        gs_dry, _, _           = stomatal_conductance(cond, photo, FT(1e-3), FT(3000), Tˡᵉᵃᶠ, ca, p, FT(1))
        gs_str, _, _           = stomatal_conductance(cond, photo, FT(1e-3), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(0.3))
        gs_drk, _, _           = stomatal_conductance(cond, photo, FT(0),    FT(1000), Tˡᵉᵃᶠ, ca, p, FT(1))

        @test gs_ref > cond.g0
        @test An_ref > 0
        @test photo.Γ★25 < ci_ref < ca          # intercellular CO₂ in physical band
        @test gs_dry < gs_ref                    # higher VPD closes stomata
        @test gs_str < gs_ref                    # moisture stress closes stomata
        @test gs_drk ≈ cond.g0                   # dark: Aₙ = −Rᵈ < 0 is floored, leaving g₀
        @test eltype(gs_ref) == FT

        # The intercellular CO₂ is the closed-form Medlyn optimality ratio.
        @test ci_ref ≈ ca * cond.g1 / (cond.g1 + sqrt(FT(1000)))
        @inferred stomatal_conductance(cond, photo, FT(1e-3), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(1))
    end
end

@testset "Standalone canopy transpiration draws from the root zone" begin
    for arch in test_architectures, land_temperature in (300.0, 280.0)
        hydrology = VariablySaturatedHydrology(; slab_depth = (0.3, 0.7), porosity = 0.4,
            residual_liquid_fraction = 0.05, storage_height = 1000, root_fraction = (0.5, 0.5),
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 0,
                                                               pore_size_uniformity = 2))
        model = coupled_land_model(arch; hydrology, land_temperature, water = 30.0,
            air_temperature = 290.0, air_specific_humidity = 0.01, wind = 5.0,
            atmosphere_land_interface_specific_humidity = CanopyConductanceHumidity(leaf_area_index = 3))
        land = model.land
        set!(land; M = 30, water_storage_2 = 245)
        update_state!(model)
        time_step!(model, 60)

        vapor = scalar(land.fluxes.vapor_flux)
        transpiration = scalar(land.fluxes.transpiration)
        @test (land_temperature - 290) * vapor > 0
        @test transpiration == max(0, vapor)
        # Effective saturations are 1/7 and 6/7; equal root fractions give the same uptake shares.
        @test scalar(land.water_storage) ≈ 30 - 60 * (vapor - transpiration + transpiration / 7)
        @test scalar(land.prognostic.water_storage_2) ≈ 245 - 60 * transpiration * 6 / 7
    end
end

#####
##### Peaked Arrhenius (Vcmax/Jmax rolloff) + Heskel respiration.
#####

@testset "Peaked Arrhenius + Heskel respiration" begin
    for FT in (Float32, Float64)
        T₂₅ = FT(298.15)

        # Normalization: both scalings are 1 at 25 °C.
        @test temperature_response(T₂₅, PeakedArrheniusParameters(FT; activation_energy = 71513, entropy = 649, deactivation_energy = 200000)) ≈ 1
        @test temperature_response(T₂₅, HeskelParameters(FT)) ≈ 1

        # Vcmax/Jmax rise then fall — interior optimum with high-T rolloff.
        Ts = FT(273):FT(1):FT(323)
        vc = [temperature_response(T, PeakedArrheniusParameters(FT; activation_energy = 71513, entropy = 649, deactivation_energy = 200000)) for T in Ts]
        jm = [temperature_response(T, PeakedArrheniusParameters(FT; activation_energy = 49884, entropy = 646, deactivation_energy = 200000)) for T in Ts]
        @test vc[end] < maximum(vc)                       # rolls off by 50 °C
        @test jm[end] < maximum(jm)
        @test Ts[argmax(vc)] ≥ Ts[argmax(jm)]             # Vcmax optimum ≥ Jmax optimum

        # Rd strictly increases over 0–45 °C (guards the Celsius-vs-Kelvin trap).
        rd = [temperature_response(T, HeskelParameters(FT)) for T in FT(273):FT(1):FT(318)]
        @test all(diff(rd) .> 0)

        photo_peak = RubiscoLightLimitedPhotosynthesis(FT)

        # Aₙ(Tˡᵉᵃᶠ) reaches an interior maximum and rolls off at high leaf temperature.
        Tl = FT(273):FT(1):FT(318)
        An_peak = [net_assimilation(photo_peak, FT(28), FT(8e-4), T, FT(101325), FT(1)) for T in Tl]
        @test An_peak[end] < maximum(An_peak)

        # Type stability.
        @test eltype(net_assimilation(photo_peak, FT(28), FT(8e-4), FT(298), FT(101325), FT(1))) == FT
        @inferred net_assimilation(photo_peak, FT(28), FT(8e-4), FT(298), FT(101325), FT(1))

        # Smooth: ∂Aₙ/∂Tₗ (finite difference) is finite and changes sign at the
        # optimum — the continuous, AD-friendly signal the peaked form provides.
        dAdT(T) = (net_assimilation(photo_peak, FT(28), FT(8e-4), T + FT(0.05), FT(101325), FT(1)) -
                   net_assimilation(photo_peak, FT(28), FT(8e-4), T - FT(0.05), FT(101325), FT(1))) / FT(0.1)
        @test dAdT(FT(288)) > 0    # below optimum: assimilation increasing
        @test dAdT(FT(313)) < 0    # above optimum: assimilation decreasing
    end
end

#####
##### Jarvis empirical conductance (closed-form, no photosynthesis).
#####

@testset "Jarvis stomatal conductance" begin
    for FT in (Float32, Float64)
        jarv = JarvisConductance(FT)
        Tˡᵉᵃᶠ = FT(298.15); ca = FT(40); p = FT(101325)

        # Runs with no photosynthesis model (no Farquhar call, no iteration).
        gs, An, ci = stomatal_conductance(jarv, nothing, FT(2e-4), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(1))
        @test gs > 0
        @test An == 0 && ci == 0
        @test eltype(gs) == FT

        # gₛ rises and saturates with APAR, falls with VPD, and peaks at optimal_temperature.
        gₛ(APAR, VPD, T) = stomatal_conductance(jarv, nothing, APAR, VPD, T, ca, p, FT(1))[1]
        gs_dim = gₛ(FT(1e-5), FT(1000), Tˡᵉᵃᶠ)
        gs_lit = gₛ(FT(1e-3), FT(1000), Tˡᵉᵃᶠ)
        gs_dry = gₛ(FT(1e-3), FT(3000), Tˡᵉᵃᶠ)
        @test gs_lit > gs_dim
        @test gₛ(FT(1e6), FT(1000), Tˡᵉᵃᶠ) ≈ gₛ(FT(1e5), FT(1000), Tˡᵉᵃᶠ) rtol=1e-3
        @test gs_dry < gs_lit
        optimum = jarv.optimal_temperature
        @test gₛ(FT(1e-3), FT(1000), optimum) > gₛ(FT(1e-3), FT(1000), optimum + FT(15))
        @test gₛ(FT(1e-3), FT(1000), optimum) > gₛ(FT(1e-3), FT(1000), optimum - FT(15))

        # Scales linearly with the moisture-stress factor β and vanishes at β = 0.
        gs_half = stomatal_conductance(jarv, nothing, FT(1e-3), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(0.5))[1]
        @test gs_half ≈ gs_lit / 2
        @test stomatal_conductance(jarv, nothing, FT(1e-3), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(0))[1] == 0

        @inferred stomatal_conductance(jarv, nothing, FT(1e-3), FT(1000), Tˡᵉᵃᶠ, ca, p, FT(1))
    end
end

#####
##### Absorbed PAR: prescribed (offline) vs live from the downwelling shortwave.
#####

@testset "Absorbed PAR (prescribed vs interactive)" begin
    for FT in (Float32, Float64)
        LAI = FT(2)

        # A prescribed value ignores the radiation state.
        @test absorbed_par_value(FT(4e-4), nothing, LAI, nothing) == FT(4e-4)

        inter = InteractiveAbsorbedPAR(FT)
        rad(downwelling_shortwave) = (; ℐꜜˢʷ = - FT(downwelling_shortwave))

        # Night: no shortwave ⇒ no absorbed PAR.
        @test absorbed_par_value(inter, rad(0), LAI, nothing) == 0

        # APAR is linear in the downwelling shortwave, and per-leaf O(1e-4) at a
        # representative midday flux — same order as the prescribed default (4e-4).
        apar_noon = absorbed_par_value(inter, rad(500), LAI, nothing)
        @test apar_noon > 0
        @test 1e-4 < apar_noon < 1e-3
        @test absorbed_par_value(inter, rad(1000), LAI, nothing) ≈ 2 * apar_noon

        @test eltype(absorbed_par_value(inter, rad(500), LAI, nothing)) == FT
        @inferred absorbed_par_value(inter, rad(500), LAI, nothing)
    end
end

#####
##### Coupled single-column: the canopy resistance raises the Bowen ratio.
#####

@testset "CanopyConductanceHumidity coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))

        # Warm, wet surface under drier air → upward evaporation.
        function latent_heat(q_formulation)
            model = coupled_land_model(arch; grid, air_temperature = 290.0, air_specific_humidity = 0.006,
                                       wind = 5.0, land_temperature = 300.0, water = nothing,
                                       hydrology = SaturatedSurface(), radiation = nothing,
                                       atmosphere_land_interface_specific_humidity = q_formulation)
            return scalar(model.interfaces.atmosphere_land_interface.fluxes.latent_heat)
        end

        # Saturated bare surface (no resistance) evaporates most; the canopy
        # stomatal resistance in series reduces the latent-heat flux magnitude.
        LE_bulk   = latent_heat(BulkHumidity(Thermodynamics.Liquid()))
        LE_canopy = latent_heat(CanopyConductanceHumidity(FT))
        @test abs(LE_canopy) < abs(LE_bulk)
        @test LE_canopy > 0   # upward evaporation → positive (evaporative cooling) latent flux

        # More leaf area → larger canopy conductance → stronger latent flux.
        LE_lo = latent_heat(CanopyConductanceHumidity(FT; leaf_area_index = 0.5))
        LE_hi = latent_heat(CanopyConductanceHumidity(FT; leaf_area_index = 4.0))
        @test abs(LE_hi) > abs(LE_lo)

        # Moisture stress throttles transpiration (constant β here; the
        # saturation-dependent CriticalSaturation path is covered above).
        LE_stressed = latent_heat(CanopyConductanceHumidity(FT; moisture_stress = 0.3))
        @test abs(LE_stressed) < abs(LE_canopy)

        # Jarvis conductance runs end-to-end (no photosynthesis model) and, like
        # Medlyn, resists evaporation relative to the unresisted bulk surface.
        LE_jarvis = latent_heat(CanopyConductanceHumidity(FT; conductance = JarvisConductance(FT)))
        @test 0 < LE_jarvis < LE_bulk

        # Live PAR is threaded to the humidity solve: with radiation off (ℐꜜˢʷ = 0)
        # the interactive canopy sees no light, so its conductance collapses toward
        # the cuticular minimum — far below the prescribed-PAR canopy. This exercises
        # the radiation state reaching `absorbed_par_value` inside the coupled solve.
        LE_dark = latent_heat(CanopyConductanceHumidity(FT; absorbed_par = InteractiveAbsorbedPAR(FT)))
        @test -1e-10 ≤ LE_dark < LE_canopy   # ≈ 0 up to roundoff
    end
end
