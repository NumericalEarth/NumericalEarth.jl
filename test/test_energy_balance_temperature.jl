include("runtests_setup.jl")

using Oceananigans
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    SoilSkinTemperature, BulkTemperature, FractionalHumidity, CanopyInterface, SoilConductiveFlux,
    SimilarityTheoryFluxes, atmosphere_land_stability_functions
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

# Skin temperature, turbulent flux sum, and conductive flux Λᵍʳⁿᵈ(Tˡᵃ − Tᵍʳⁿᵈ) over a 300 K slab with Λᵍʳⁿᵈ = 1.5 / 0.05.
# With radiation off the balance is Λᵍʳⁿᵈ(Tˡᵃ − Tᵍʳⁿᵈ) = 𝒬ᵀ + 𝒬ᵛ.
function skin_column(arch, temperature, FT = Float64; air_temperature = 290.0, efficiency = 1.0)
    humidity = FractionalHumidity(AtmosphericThermodynamics.Liquid(); efficiency)
    model = coupled_land_model(arch, FT; air_temperature, land_temperature=300.0, air_specific_humidity=0.006, wind=5.0,
                               water=90.0, radiation=nothing, atmosphere_land_interface_state=(; temperature, specific_humidity=humidity))
    interface = model.interfaces.atmosphere_land_interface
    T = scalar(interface.temperature)
    return T, scalar(interface.fluxes.sensible_heat) + scalar(interface.fluxes.latent_heat), 1.5 / 0.05 * (300 - T)
end

@testset "EnergyBalanceTemperature" begin
    for arch in test_architectures
        skin(Λ) = SoilSkinTemperature(Λ, 0.05)
        @test skin_column(arch, BulkTemperature())[1] ≈ 300
        @test skin_column(arch, skin(1e7))[1] ≈ 300 atol=1e-2

        # An evaporating moist surface cools the skin; warm air over a nearly dry surface warms it.
        for (air_temperature, efficiency, warming) in ((290.0, 1.0, -1), (312.0, 0.05, 1))
            T, F, G = skin_column(arch, skin(1.5); air_temperature, efficiency)
            @test sign(T - 300) == warming
            @test F ≈ G rtol=1e-4
        end
    end

    for FT in (Float32, Float64)
        T, F, G = skin_column(CPU(), SoilSkinTemperature(FT(1.5), FT(0.05)), FT)
        @test T isa FT
        @test F ≈ G rtol=1e-3
    end
end

@testset "Massless skin closes its energy balance in calm air" begin
    for arch in test_architectures
        conductivity, thickness = 0.625, 0.05
        soil = CanopyInterface(Float64; soil_humidity = bare_soil_humidity(Float64),
                               ground_heat_flux = SoilConductiveFlux(conductivity, thickness))
        rough = SimilarityTheoryFluxes(Float64; stability_functions = atmosphere_land_stability_functions(Float64),
                                       momentum_roughness_length = 1.0, temperature_roughness_length = 0.1,
                                       water_vapor_roughness_length = 0.1)
        model = tiled_land_model(arch; vegetated = soil, cover_fraction = 0.0, interface_kw = (; bare_fluxes = rough),
                                 wind = 0.3, shortwave = 1000.0, air_temperature = 305.0, air_specific_humidity = 0.002,
                                 land_temperature = 295.0, water = 150.0)
        skin = model.interfaces.atmosphere_land_interface.bare
        radiation = model.interfaces.exchanger.radiation.state
        T = scalar(skin.temperature)

        σ = model.radiation.stefan_boltzmann_constant
        α = model.radiation.surface_properties.land.albedo
        ϵ = model.radiation.surface_properties.land.emissivity
        net_radiation = - (1 - α) * scalar(radiation.ℐꜜˢʷ) - ϵ * (scalar(radiation.ℐꜜˡʷ) + σ * T^4)
        ground_heat_flux = conductivity / thickness * (T - scalar(model.land.temperature))
        turbulent_heat_flux = scalar(skin.fluxes.sensible_heat) + scalar(skin.fluxes.latent_heat)
        @test net_radiation - ground_heat_flux - turbulent_heat_flux ≈ 0 atol=1e-2
    end
end
