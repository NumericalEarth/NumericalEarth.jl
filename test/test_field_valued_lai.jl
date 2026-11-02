include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyConductanceHumidity, JarvisConductance
using NumericalEarth.Lands: SaturatedSurface

function jarvis_latent_heat(arch, grid, leaf_area_index; time = nothing)
    canopy = CanopyConductanceHumidity(Float64; leaf_area_index, conductance = JarvisConductance(Float64))
    model = coupled_land_model(arch; grid, time, air_temperature = 290.0, air_specific_humidity = 0.006,
                               wind = 5.0, land_temperature = 300.0, water = nothing,
                               hydrology = SaturatedSurface(), radiation = nothing,
                               atmosphere_land_interface_state = (temperature = BulkTemperature(), specific_humidity = canopy))
    return Array(interior(model.interfaces.atmosphere_land_interface.fluxes.latent_heat))[:, 1, 1]
end

@testset "Field-valued λˡᵉᵃᶠ: spatial coupled fluxes" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (2, 1, 1), longitude = (0, 20), latitude = (0, 10),
                                     z = (-1, 0), topology = (Bounded, Bounded, Bounded))
        leaf_area_index = Field{Center, Center, Nothing}(grid)
        set!(leaf_area_index, (λ, φ) -> ifelse(λ < 10, 4.0, 0.0))
        LE = jarvis_latent_heat(arch, grid, leaf_area_index)
        @test LE[1] > LE[2]
        @test isapprox(LE[2], 0; atol = 1e-6)
        @test all(isfinite, LE)
    end
end

@testset "Field-valued λˡᵉᵃᶠ: temporal coupled fluxes" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = 1, latitude = 10, longitude = 10, z = (-1, 0), topology = (Flat, Flat, Bounded))
        leaf_area_index = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 100.0])
        set!(leaf_area_index[1], 0.5)
        set!(leaf_area_index[2], 4.0)
        LE₀, LE₅₀, LE₁₀₀ = (only(jarvis_latent_heat(arch, grid, leaf_area_index; time)) for time in (0.0, 50.0, 100.0))
        @test LE₀ < LE₅₀ < LE₁₀₀
        @test all(isfinite, (LE₀, LE₁₀₀))
    end
end
