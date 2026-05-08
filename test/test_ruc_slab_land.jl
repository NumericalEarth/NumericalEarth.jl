using Test
using NumericalEarth
using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: time_step!, update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations: SimilarityScales

const InterfaceComputations = NumericalEarth.EarthSystemModels.InterfaceComputations

scalar(field) = Array(interior(field))[1]

function expected_jarvis_resistance(rg, qa, Ta, θ, lai, r_smin, p_hPa, p)
    e = Ta > 273.15 ?
        6.1121 * exp(17.502 * (Ta - 273.15) / (Ta - 32.18)) :
        6.1115 * exp(22.452 * (Ta - 273.15) / (Ta - 0.61))
    qsat_air = 0.622 * e / (p_hPa - (1 - 0.622) * e)

    F1 = max((rg / p.rg_lim) / (1 + rg / p.rg_lim), 1e-3)
    F2 = clamp(1 / (1 + max(0, qsat_air - qa) / p.vpd_lim), 1e-3, 1)
    F3 = θ ≥ p.theta_fc ? 1.0 :
         θ ≤ p.theta_wilt ? 1e-3 :
         clamp((θ - p.theta_wilt) / (p.theta_fc - p.theta_wilt), 1e-3, 1)
    F4 = clamp(1 - 0.0016 * (p.T_opt - Ta)^2, 1e-3, 1)

    return clamp(r_smin / (lai * F1 * F2 * F3 * F4), r_smin, p.r_smax)
end

function ruc_test_grid()
    return RectilinearGrid(CPU();
                           size = (1, 1),
                           halo = (1, 1),
                           x = (0, 1),
                           y = (0, 1),
                           topology = (Bounded, Bounded, Flat))
end

function ruc_test_land(; T = 273.15, θ = 0.30, vegfrac = 0.0, lai = 0.0,
                       parameters = RucSlabLandParameters(Float64))
    land = RucSlabLand(ruc_test_grid(); parameters)
    set!(land; T, Tc = T, θ, vegfrac, lai)

    fill!(land.forcings.air_temperature, T)
    fill!(land.forcings.air_humidity, 0)
    fill!(land.forcings.solar_irradiance, 100)
    fill!(land.forcings.surface_pressure, 1013.25)

    return land
end

@testset "RUC slab land surface processes" begin
    @testset "canopy interception uses RUC fixed saturation" begin
        land = ruc_test_land(T = 280, θ = 0.30, vegfrac = 0.8, lai = 5.0)

        fill!(land.forcings.rainfall_rate, 0.01)
        time_step!(land, 1.0)

        @test scalar(land.vegetation.canopy_capacity) ≈ 5e-4
        @test scalar(land.canopy.cst) ≈ 5e-4
    end

    @testset "mavail and bare-soil evaporation use RUC soilres" begin
        land = ruc_test_land(T = 298, θ = 0.10, vegfrac = 0.5, lai = 2.0)
        update_state!(land)

        p = land.parameters
        expected_mavail = (0.10 - p.theta_air_dry) / (p.theta_fc - p.theta_air_dry)
        fc = max(p.theta_air_dry, 0.5 * p.theta_fc)
        fex_fc = max(0.01, min(1.0, 0.10 / fc))
        expected_soilres = 0.25 * (1 - cos(pi * fex_fc))^2

        @test scalar(land.vegetation.mavail) ≈ expected_mavail

        fill!(land.forcings.moisture_flux, 0.1)
        time_step!(land, 10.0)

        expected_θ = 0.10 - 0.1 * (1 - 0.5) * expected_soilres * 10.0 /
                           (1000 * p.soil_depth)
        @test scalar(land.soil_moisture) ≈ expected_θ
    end

    @testset "snow melt is capped and retained following RUC" begin
        land = ruc_test_land(T = 274.15, θ = 0.20)
        set!(land; snwe = 0.10, snhei = 0.40, rhosn = 250.0, swl = 0.0)
        update_state!(land)

        time_step!(land, 60.0)

        melt_swe = 60.0 * 5.6e-8 * 2.0 * 1.0
        retained_fraction = min(0.18, max(0.08, 0.10 / 0.10 * 0.13))
        retained = retained_fraction * melt_swe
        overflow = melt_swe - retained

        @test scalar(land.snow.swl) ≈ retained
        @test scalar(land.snow.swl_overflow) ≈ overflow
        @test scalar(land.soil_moisture) ≈ 0.20 + overflow / land.parameters.soil_depth

        solid_swe = 0.10 - melt_swe
        total_pack_swe = solid_swe + retained
        expected_rhosn = (250.0 * solid_swe + 1000.0 * retained) / total_pack_swe
        @test scalar(land.snow.snhei) ≈ total_pack_swe * 1000.0 / expected_rhosn
    end

    @testset "fresh snow density follows WRF air-temperature formula" begin
        land = ruc_test_land(T = 260.0, θ = 0.30)
        fill!(land.forcings.air_temperature, 280.0)
        fill!(land.forcings.snowfall_rate, 1e-6)

        time_step!(land, 1.0)

        expected_ρ_new = min(125.0, 1000.0 / max(8.0, 17.0 * tanh((276.65 - 280.0) * 0.15)))
        @test scalar(land.snow.rhonewsn) ≈ expected_ρ_new
        @test scalar(land.snow.rhosn) ≈ expected_ρ_new
    end

    @testset "current-step fresh snow activates RUC melt limiter" begin
        land = ruc_test_land(T = 274.15, θ = 0.20)
        set!(land; snwe = 0.10, snhei = 0.25, rhosn = 400.0, swl = 0.0)
        fill!(land.forcings.air_temperature, 263.15)
        fill!(land.forcings.snowfall_rate, 1e-6)
        update_state!(land)

        time_step!(land, 60.0)

        new_swe = 1e-6 * 60.0
        capped_melt = 60.0 * 5.6e-8 * 2.0 * 1.0
        @test scalar(land.snow.snwe) ≈ 0.10 + new_swe - capped_melt
    end

    @testset "fresh-snow albedo diagnostics reset without current snowfall" begin
        land = ruc_test_land(T = 260.0, θ = 0.30)
        fill!(land.forcings.air_temperature, 280.0)
        fill!(land.forcings.snowfall_rate, 3e-4)

        # Step 1 — first snowfall on a clean slate. `snowfallac` is zero
        # going in, so per Fortran:1641 `snowfracnewsn` is evaluated from
        # the **pre-increment** `snowfallac` (= 0) and the fresh-snow
        # albedo lock cannot latch yet.
        time_step!(land, 1.0)
        @test scalar(land.snow.keep_snow_albedo) == 0
        @test scalar(land.snow.rhonewsn) ≈ 125.0

        # Steps 2-3 — continued snowfall accumulates `snowfallac` until the
        # pre-increment value exceeds `snhei_crit_newsn = 0.0005·ρ_w/ρ_sn ≈
        # 4 mm`, at which point the fresh-snow-albedo lock activates.
        time_step!(land, 1.0)
        time_step!(land, 1.0)
        @test scalar(land.snow.keep_snow_albedo) == 1
        @test scalar(land.snow.rhonewsn) ≈ 125.0

        fill!(land.forcings.snowfall_rate, 0)
        time_step!(land, 1.0)

        @test scalar(land.snow.keep_snow_albedo) == 0
        @test scalar(land.snow.snowfracnewsn) == 0
        @test scalar(land.snow.rhonewsn) ≈ 100.0
    end

    @testset "WRF RUC snow emissivity and USGS table defaults" begin
        parameters = RucSlabLandParameters(Float64)
        @test parameters.emiss_snow ≈ 0.98

        registry = usgs_land_classifications(Float64)
        @test registry[2].z0 ≈ 0.20
        @test registry[2].lai ≈ 5.68
        @test registry[2].emissivity ≈ 0.92
        @test registry[24].z0 ≈ 0.011
        @test registry[24].emissivity ≈ 0.98
    end

    @testset "land classification applies valid ids and preserves invalid ids" begin
        land = ruc_test_land()
        registry = usgs_land_classifications(Float64)
        grassland = registry[7]

        apply_land_classifications!(land, fill(grassland.id, 1, 1), registry)

        @test scalar(land.vegetation.vegfrac) ≈ grassland.vegfrac
        @test scalar(land.vegetation.lai) ≈ grassland.lai
        @test scalar(land.vegetation.albedo_veg) ≈ grassland.albedo
        @test scalar(land.vegetation.emissivity_veg) ≈ grassland.emissivity
        @test scalar(land.vegetation.z0_veg) ≈ grassland.z0
        @test scalar(land.vegetation.r_smin) ≈ grassland.r_smin
        @test scalar(land.vegetation.is_urban) ≈ 0

        apply_land_classifications!(land, fill(999, 1, 1), registry)

        @test scalar(land.vegetation.vegfrac) ≈ grassland.vegfrac
        @test scalar(land.vegetation.lai) ≈ grassland.lai
        @test scalar(land.vegetation.albedo_veg) ≈ grassland.albedo
        @test scalar(land.vegetation.emissivity_veg) ≈ grassland.emissivity
        @test scalar(land.vegetation.z0_veg) ≈ grassland.z0
        @test scalar(land.vegetation.r_smin) ≈ grassland.r_smin
        @test scalar(land.vegetation.is_urban) ≈ 0
    end

    @testset "Jarvis resistance uses local surface pressure" begin
        land = ruc_test_land(T = 298.0, θ = 0.30, vegfrac = 0.8, lai = 3.0)
        fill!(land.forcings.air_temperature, 300.0)
        fill!(land.forcings.air_humidity, 0.005)
        fill!(land.forcings.solar_irradiance, 300.0)
        fill!(land.forcings.surface_pressure, 700.0)

        update_state!(land)

        p = land.parameters
        expected = expected_jarvis_resistance(300.0, 0.005, 300.0, 0.30, 3.0,
                                              p.r_smin, 700.0, p)
        @test scalar(land.vegetation.r_s) ≈ expected
    end

    @testset "land roughness enters atmosphere-land MOST fluxes" begin
        grid = RectilinearGrid(CPU();
                               size = (2, 1),
                               halo = (1, 1),
                               x = (0, 2),
                               y = (0, 1),
                               topology = (Bounded, Bounded, Flat))

        land = RucSlabLand(grid)
        set!(land; T = 293.15, Tc = 293.15, θ = 0.30, vegfrac = 1.0, lai = 1.0)
        interior(land.vegetation.z0_veg, :, :, 1) .= reshape([0.02, 0.80], 2, 1)
        update_state!(land)

        atmosphere = PrescribedAtmosphere(grid, [0.0])
        fill!(parent(atmosphere.velocities.u), 5.0)
        fill!(parent(atmosphere.velocities.v), 0.0)
        fill!(parent(atmosphere.tracers.T), 293.15)
        fill!(parent(atmosphere.tracers.q), 0.005)
        fill!(parent(atmosphere.pressure), 101325.0)
        update_state!(atmosphere)

        default_interfaces = InterfaceComputations.ComponentInterfaces(atmosphere, nothing, nothing; land)
        default_flux_formulation = default_interfaces.atmosphere_land_interface.flux_formulation
        @test default_flux_formulation.roughness_lengths.momentum isa LandRoughnessLength
        @test default_flux_formulation.roughness_lengths.temperature.multiplier ≈ 0.1
        @test default_flux_formulation.roughness_lengths.water_vapor.multiplier ≈ 0.1

        @inline zero_stability_function(ζ) = zero(ζ)
        stability_functions = SimilarityScales(zero_stability_function,
                                               zero_stability_function,
                                               zero_stability_function)

        flux_formulation = SimilarityTheoryFluxes(; momentum_roughness_length = LandRoughnessLength(Float64),
                                                     temperature_roughness_length = LandRoughnessLength(Float64; multiplier = 0.1),
                                                     water_vapor_roughness_length = LandRoughnessLength(Float64; multiplier = 0.1),
                                                     gustiness_parameter = 0,
                                                     minimum_gustiness = 0,
                                                     stability_functions,
                                                     solver_stop_criteria = InterfaceComputations.FixedIterations(1))

        interfaces = InterfaceComputations.ComponentInterfaces(atmosphere, nothing, nothing;
                                                              land,
                                                              atmosphere_land_fluxes = flux_formulation)
        model = AtmosphereLandModel(atmosphere, land; interfaces)

        κ = flux_formulation.von_karman_constant
        h = atmosphere.surface_layer_height
        U = 5.0
        u★ = model.interfaces.atmosphere_land_interface.fluxes.friction_velocity

        @test u★[1, 1, 1] ≈ κ * U / log(h / 0.02)
        @test u★[2, 1, 1] ≈ κ * U / log(h / 0.80)
        @test u★[1, 1, 1] < u★[2, 1, 1]
    end
end
