include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyConductanceHumidity, JarvisConductance
using NumericalEarth.Lands: SaturatedSurface
#####
##### Integration — a static spatial LAI map drives spatially varying transpiration.
#####

@testset "Field-valued LAI: spatial coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = (2, 1, 1),
                                     longitude = (0, 20), latitude = (0, 10),
                                     z = (-1, 0),
                                     topology = (Bounded, Bounded, Bounded))

        # Cell 1 (λ = 5): dense canopy; cell 2 (λ = 15): bare (LAI = 0).
        lai_field = Field{Center, Center, Nothing}(grid)
        set!(lai_field, (λ, φ) -> ifelse(λ < 10, 4.0, 0.0))

        canopy = CanopyConductanceHumidity(FT; leaf_area_index = lai_field, conductance = JarvisConductance(FT))
        model = coupled_land_model(arch; grid, Tair = 290.0, qair = 0.006, wind = 5.0, Tland = 300.0,
                                   water = nothing, hydrology = SaturatedSurface(), radiation = nothing,
                                   atmosphere_land_interface_specific_humidity = canopy)
        LE = Array(interior(model.interfaces.atmosphere_land_interface.fluxes.latent_heat))[:, 1, 1]

        # Dense-canopy cell transpires; the bare cell (g_c = LAI · gₛ = 0) has ~no latent flux.
        @test LE[1] > LE[2]
        @test isapprox(LE[2], 0; atol = 1e-6)
        @test all(isfinite, LE)
    end
end

#####
##### Integration — a monthly-style LAI time series drives time-varying transpiration.
#####

@testset "Field-valued LAI: temporal coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))

        times   = [0.0, 100.0]
        lai_fts = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        set!(lai_fts[1], 0.5)   # sparse at t = 0
        set!(lai_fts[2], 4.0)   # dense at t = 100

        function latent_heat_at(t)
            canopy = CanopyConductanceHumidity(FT; leaf_area_index = lai_fts, conductance = JarvisConductance(FT))
            model = coupled_land_model(arch; grid, time = t, Tair = 290.0, qair = 0.006, wind = 5.0, Tland = 300.0,
                                       water = nothing, hydrology = SaturatedSurface(), radiation = nothing,
                                       atmosphere_land_interface_specific_humidity = canopy)
            return scalar(model.interfaces.atmosphere_land_interface.fluxes.latent_heat)
        end

        LE_t0  = latent_heat_at(0.0)     # LAI = 0.5
        LE_mid = latent_heat_at(50.0)    # LAI = 2.25 (interpolated)
        LE_t1  = latent_heat_at(100.0)   # LAI = 4.0

        # Rising LAI ⇒ rising canopy conductance ⇒ rising latent flux.
        @test LE_t0 < LE_mid < LE_t1
        @test all(isfinite, (LE_t0, LE_mid, LE_t1))
    end
end
