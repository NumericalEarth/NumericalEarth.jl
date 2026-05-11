using Test
using Oceananigans
using Oceananigans.Fields: interior, CenterField
using Oceananigans.TimeSteppers: time_step!, update_state!
using NumericalEarth
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketWithSnow,
                            PrescribedSurfaceProperties,
                            wetness

bw_grid() = RectilinearGrid(CPU();
                            size = (1, 1), halo = (1, 1),
                            x = (0, 1), y = (0, 1),
                            topology = (Bounded, Bounded, Flat))

bw_cell(field) = Array(interior(field))[1, 1, 1]

function bw_land(; T = 290.0, W = 100.0, SWE = 0.0,
                  vegfrac = 0.0, lai = 0.0,
                  W_max = 150.0, W_crit_frac = 0.75,
                  SWE_crit = 0.05,
                  ρcH_g = 1.0e6)
    grid = bw_grid()
    energy    = SlabEnergy(Float64; heat_capacity = ρcH_g)
    hydrology = BucketWithSnow(Float64;
                                W_max = W_max,
                                W_crit_frac = W_crit_frac,
                                SWE_crit = SWE_crit,
                                ground_heat_capacity = ρcH_g)
    surface   = PrescribedSurfaceProperties(grid;
                                            vegfrac = vegfrac, lai = lai)
    land = SlabLand(grid; energy, hydrology, surface)
    fill!(land.state.T, T)
    fill!(land.state.W, W)
    fill!(land.state.SWE, SWE)
    fill!(land.fluxes.air_temperature, T)
    fill!(land.fluxes.air_humidity, 0.0)
    fill!(land.fluxes.surface_pressure, 1013.25)
    fill!(land.fluxes.solar_irradiance, 0.0)
    return land
end

@testset "BucketWithSnow — bucket-only path" begin
    @testset "W mass balance with rain only" begin
        land = bw_land(W = 100.0)
        fill!(land.fluxes.rainfall_rate, 0.01)   # kg m⁻² s⁻¹
        time_step!(land, 10.0)
        @test bw_cell(land.state.W) ≈ 100.0 + 0.01 * 10.0
    end

    @testset "W saturates at W_max with overflow lost" begin
        land = bw_land(W = 149.5, W_max = 150.0)
        fill!(land.fluxes.rainfall_rate, 0.1)    # 1 kg/m² inflow over 10 s
        time_step!(land, 10.0)
        @test bw_cell(land.state.W) ≈ 150.0
    end

    @testset "wetness saturates at β = 1 when W ≥ W_crit" begin
        land = bw_land(W = 150.0, W_max = 150.0, W_crit_frac = 0.75)
        update_state!(land)
        @test bw_cell(wetness(land.hydrology, land.state, land.parameters)) ≈ 1.0
    end

    @testset "wetness is W / (W_crit · W_max) for W < W_crit, vegfrac=0" begin
        land = bw_land(W = 50.0, W_max = 150.0, W_crit_frac = 0.75)
        update_state!(land)
        @test bw_cell(wetness(land.hydrology, land.state, land.parameters)) ≈
              50.0 / (0.75 * 150.0)
    end
end

@testset "BucketWithSnow — snow ablation" begin
    @testset "energy-limited melt cools slab and feeds bucket" begin
        ρcH = 1.0e6
        land = bw_land(T = 280.0, W = 100.0, SWE = 0.10,
                        SWE_crit = 0.05, ρcH_g = ρcH)
        fill!(land.fluxes.rainfall_rate, 0.0)
        fill!(land.fluxes.snowfall_rate, 0.0)
        fill!(land.fluxes.evaporation, 0.0)
        fill!(land.fluxes.sublimation, 0.0)
        fill!(land.fluxes.transpiration, 0.0)

        L_f = 3.337e5
        ρ_w = 1000.0
        # Δmelt_max = ρcH * (280 − 273.15) / (L_f · ρ_w)
        Δmelt_expected = ρcH * (280.0 - 273.15) / (L_f * ρ_w)

        time_step!(land, 1.0)

        @test bw_cell(land.state.T) ≈ 273.15 atol = 1e-9
        @test bw_cell(land.state.SWE) ≈ 0.10 - Δmelt_expected
        @test bw_cell(land.state.W)   ≈ 100.0 + Δmelt_expected * ρ_w
    end

    @testset "sublimation drains SWE without changing T" begin
        land = bw_land(T = 270.0, W = 100.0, SWE = 0.05, ρcH_g = 1.0e6)
        fill!(land.fluxes.sublimation, 1e-4)   # 0.1 g m⁻² s⁻¹
        Δt = 60.0
        time_step!(land, Δt)
        ρ_w = 1000.0
        @test bw_cell(land.state.SWE) ≈ 0.05 - 1e-4 * Δt / ρ_w
        @test bw_cell(land.state.T) ≈ 270.0
    end
end

@testset "BucketWithSnow — β corner cases" begin
    @testset "vegfrac = 0, W = 0, SWE = 0 → β = 0" begin
        land = bw_land(W = 0.0, SWE = 0.0, vegfrac = 0.0)
        update_state!(land)
        @test bw_cell(land.state.moisture_availability) ≈ 0.0
    end

    @testset "vegfrac = 0, W = W_crit, SWE = 0 → β = 1" begin
        W_max = 150.0; W_crit_frac = 0.75
        land = bw_land(W = W_crit_frac * W_max,
                        SWE = 0.0, vegfrac = 0.0,
                        W_max = W_max, W_crit_frac = W_crit_frac)
        update_state!(land)
        @test bw_cell(land.state.moisture_availability) ≈ 1.0
    end

    @testset "SWE ≫ SWE_crit → β → 1" begin
        land = bw_land(W = 0.0, SWE = 0.5, SWE_crit = 0.05, vegfrac = 0.0)
        update_state!(land)
        @test bw_cell(land.state.moisture_availability) ≈ 1.0 atol = 1e-8
    end
end
