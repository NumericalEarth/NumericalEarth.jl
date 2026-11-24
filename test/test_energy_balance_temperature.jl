include("runtests_setup.jl")

using Oceananigans
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    SoilSkinTemperature, BulkTemperature, FractionalHumidity, CanopyInterface, SoilConductiveFlux,
    SimilarityTheoryFluxes, MomentumRoughnessLength, ScalarRoughnessLength, atmosphere_land_stability_functions
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

# Skin temperature, friction velocity, and sensible and latent heat fluxes of a land column.
function column_fluxes(arch, temperature; ℓ = 0.1, ℓˢ = ℓ / 10, efficiency = 0.5, solver_stop_criteria = nothing, kw...)
    fluxes = SimilarityTheoryFluxes(Float64; stability_functions = atmosphere_land_stability_functions(Float64),
                                    momentum_roughness_length = ℓ, temperature_roughness_length = ℓˢ,
                                    water_vapor_roughness_length = ℓˢ, solver_stop_criteria)
    humidity = FractionalHumidity(AtmosphericThermodynamics.Liquid(); efficiency)
    model = coupled_land_model(arch; atmosphere_land_fluxes = fluxes,
                               atmosphere_land_interface_state = (; temperature, specific_humidity = humidity), kw...)
    interface = model.interfaces.atmosphere_land_interface
    return scalar(interface.temperature), scalar(interface.fluxes.friction_velocity),
           scalar(interface.fluxes.sensible_heat), scalar(interface.fluxes.latent_heat)
end

@testset "Massless skin fluxes are the similarity fluxes at its temperature" begin
    # Still and windy sunny air, a smooth surface that convects past the unstable bound of the stability
    # functions, a rough surface on a clear night past the stable bound, still dry air whose evaporation
    # reverses the buoyancy flux as u★ rises, and roughness lengths that depend on u★.
    columns = ((; wind = 0.0), (; wind = 3.0), (; wind = 0.2, ℓ = 1e-5),
               (; wind = 2.0, ℓ = 1.0, shortwave = 0.0, longwave = 250.0),
               (; wind = 0.0, ℓ = 1.0, shortwave = 1000.0, air_temperature = 305.0, air_specific_humidity = 0.006, land_temperature = 315.0),
               (; wind = 5.0, ℓ = MomentumRoughnessLength(Float64), ℓˢ = ScalarRoughnessLength(Float64)))

    for arch in test_architectures, column in columns
        T, skin_fluxes... = column_fluxes(arch, SoilSkinTemperature(0.625, 0.05); column...)
        _, bulk_fluxes... = column_fluxes(arch, BulkTemperature(); column..., land_temperature = T)
        @test all(isapprox.(skin_fluxes, bulk_fluxes; rtol = 1e-5))
    end
end

@testset "Massless skin stays coupled where a decoupled state also balances" begin
    # A saturated rough surface under a clear night sky: the stable bound implies itself at 283.0 K, and
    # the coupled skin balances at 283.8 K.
    for arch in test_architectures
        T, _ = column_fluxes(arch, SoilSkinTemperature(0.625, 0.05); wind = 3.5, ℓ = 0.5, efficiency = 1.0, shortwave = 0.0,
                             longwave = 289.0, air_temperature = 290.0, air_specific_humidity = 0.00711, land_temperature = 287.0)
        @test T > 283.5
    end
end

@testset "Massless skin with FixedIterations" begin
    for arch in test_architectures, wind in (1.0, 3.0)
        T, _ = column_fluxes(arch, SoilSkinTemperature(0.625, 0.05); wind)
        Tⁿ, _ = column_fluxes(arch, SoilSkinTemperature(0.625, 0.05); wind, solver_stop_criteria = FixedIterations(8))
        @test Tⁿ ≈ T atol=0.5
    end
end

@testset "Massless skin closes its energy balance in calm air" begin
    conductivity, thickness = 0.625, 0.05

    # Rough surfaces under light wind: sunny over cool wet soil, hot and dry over drier soil, and at night.
    calm_columns = ((1.0, (; wind = 0.3,  shortwave = 1000.0, air_temperature = 305.0, air_specific_humidity = 0.002,
                             land_temperature = 295.0, water = 150.0)),
                    (1.5, (; wind = 0.04, shortwave = 200.0,  air_temperature = 318.0, air_specific_humidity = 0.012,
                             land_temperature = 328.0, water = 45.0)),
                    (1.5, (; wind = 1.0,  shortwave = 0.0,    air_temperature = 305.0, air_specific_humidity = 0.006,
                             land_temperature = 305.0, water = 150.0)))

    for arch in test_architectures, (ℓ, column) in calm_columns, wet_soil_resistance in (nothing, ExponentialSoilResistance(Float64))
        soil = CanopyInterface(Float64; soil_humidity = bare_soil_humidity(Float64; wet_soil_resistance),
                               ground_heat_flux = SoilConductiveFlux(conductivity, thickness))
        rough = SimilarityTheoryFluxes(Float64; stability_functions = atmosphere_land_stability_functions(Float64),
                                       momentum_roughness_length = ℓ, temperature_roughness_length = ℓ / 10,
                                       water_vapor_roughness_length = ℓ / 10)
        model = tiled_land_model(arch; vegetated = soil, cover_fraction = 0.0, interface_kw = (; bare_fluxes = rough), column...)
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
