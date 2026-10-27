include("runtests_setup.jl")

using Oceananigans
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    SoilSkinTemperature, BulkTemperature, FractionalHumidity,
    compute_interface_temperature,
    AirLandInterfaceState, InterfaceFluxScales, InterfaceVelocities
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

# Skin temperature, turbulent flux sum, and conductive flux Λᵍʳᵒ(Tˡᵃ − Tᵍʳᵒ) over a 300 K slab with Λᵍʳᵒ = 1.5 / 0.05.
# With radiation off the balance is Λᵍʳᵒ(Tˡᵃ − Tᵍʳᵒ) = 𝒬ᵀ + 𝒬ᵛ.
function skin_column(arch, temperature; air_temperature = 290.0, efficiency = 1.0)
    humidity = FractionalHumidity(AtmosphericThermodynamics.Liquid(); efficiency)
    model = coupled_land_model(arch; air_temperature, land_temperature=300.0, air_specific_humidity=0.006, wind=5.0,
                               water=90.0, radiation=nothing, atmosphere_land_interface_temperature=temperature,
                               atmosphere_land_interface_specific_humidity=humidity)
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
        Ψₛ = AirLandInterfaceState(InterfaceFluxScales(FT(0.3), FT(0.1), FT(-2e-4)), InterfaceVelocities(FT(0), FT(0)),
                                   FT(300), FT(8e-3))
        Ψₐ = (T=FT(290), p=FT(1e5), q=FT(6e-3), z=FT(10))
        Ψˡᵃ = (T=FT(300), 𝒮=FT(0.6), surface_storage=nothing)
        Ψᵣ = (σ=FT(0), α=FT(0), ϵ=FT(0), ℐꜜˢʷ=FT(0), ℐꜜˡʷ=FT(0))
        ℙₛ = (; state = (; specific_humidity = FractionalHumidity(AtmosphericThermodynamics.Liquid(); efficiency=FT(1))))
        ℙₐ = (thermodynamics_parameters=AtmosphereThermodynamicsParameters(FT), gravitational_acceleration=FT(9.81))
        skin = SoilSkinTemperature(FT(1.5), FT(0.05))
        @test @inferred(compute_interface_temperature(skin, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, (;))) isa FT
    end
end
