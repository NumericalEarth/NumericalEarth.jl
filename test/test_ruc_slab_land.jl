using Test
using NumericalEarth
using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: time_step!, update_state!

scalar(field) = Array(interior(field))[1]

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
end
