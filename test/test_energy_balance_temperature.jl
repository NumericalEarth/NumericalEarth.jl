include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    EnergyBalanceTemperature, SoilSkinTemperature, DiagnosticSkin,
    BulkTemperature, SoilConductiveFlux, FractionalHumidity,
    compute_interface_temperature,
    AirLandInterfaceState, InterfaceFluxScales, InterfaceVelocities
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Atmospheres: PrescribedAtmosphere, AtmosphereThermodynamicsParameters
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

# Solve the coupled single-column interface and return the skin temperature, the turbulent
# flux sum, and the conductive flux Λᵍ(Tˡᵃ − Tᵍ). With radiation off the massless balance
# reduces to Λᵍ(Tˡᵃ − Tᵍ) = 𝒬ᵀ + 𝒬ᵛ, so the two agree at convergence.
function skin_column(arch; temperature, Tair, Tland, efficiency, conductivity=1.5, thickness=0.05)
    FT = Float64
    grid = LatitudeLongitudeGrid(arch, FT; size=1, latitude=10, longitude=10,
                                 z=(-1, 0), topology=(Flat, Flat, Bounded))
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height=10, boundary_layer_height=512)
    fill!(parent(atmosphere.temperature), Tair)
    fill!(parent(atmosphere.specific_humidity), 0.006)
    fill!(parent(atmosphere.velocities.u), 5.0)
    fill!(parent(atmosphere.pressure), 101325.0)
    land = SlabLand(grid; hydrology=BucketHydrology(FT; maximum_water_storage=150.0), energy=SlabEnergy(FT))
    set!(land; T=Tland)
    fill!(parent(land.water_storage), 90.0)   # 𝒮 = 0.6
    model = AtmosphereLandModel(atmosphere, land; radiation=nothing,
                                atmosphere_land_interface_temperature=temperature,
                                atmosphere_land_interface_specific_humidity=FractionalHumidity(AtmosphericThermodynamics.Liquid(); efficiency))
    update_state!(model.land)
    update_state!(model)
    ai = model.interfaces.atmosphere_land_interface
    Tin = Array(interior(ai.temperature))[1, 1, 1]
    QT  = Array(interior(ai.fluxes.sensible_heat))[1, 1, 1]
    QV  = Array(interior(ai.fluxes.latent_heat))[1, 1, 1]
    Λ   = conductivity / thickness
    return Tin, QT + QV, Λ * (Tland - Tin)
end

@testset "EnergyBalanceTemperature" begin
    for arch in test_architectures
        skin(Λ) = SoilSkinTemperature(Λ, 0.05; storage=DiagnosticSkin())

        # Bulk temperature: the skin equals the bulk slab temperature.
        Tin_bulk, _, _ = skin_column(arch; temperature=BulkTemperature(), Tair=290.0, Tland=300.0, efficiency=1.0)
        @test Tin_bulk ≈ 300.0

        # Λᵍ → ∞ recovers BulkTemperature (skin pinned to the bulk).
        Tin_stiff, _, _ = skin_column(arch; temperature=skin(1e7), Tair=290.0, Tland=300.0, efficiency=1.0)
        @test isapprox(Tin_stiff, 300.0; atol=1e-2)

        # Evaporating moist surface: skin cooler than the bulk; balance closes.
        Tin_c, F_c, G_c = skin_column(arch; temperature=skin(1.5), Tair=290.0, Tland=300.0, efficiency=1.0)
        @test Tin_c < 300.0
        @test isapprox(F_c, G_c; rtol=1e-4)

        # Warm air over a nearly-dry surface: downward sensible heat, skin warmer
        # than the bulk; balance closes.
        Tin_h, F_h, G_h = skin_column(arch; temperature=skin(1.5), Tair=312.0, Tland=300.0, efficiency=0.05)
        @test Tin_h > 300.0
        @test isapprox(F_h, G_h; rtol=1e-4)
    end

    for FT in (Float32, Float64)
        t = SoilSkinTemperature(FT(1.5), FT(0.05); storage=DiagnosticSkin())
        Ψₛ = AirLandInterfaceState(InterfaceFluxScales(FT(0.3), FT(0.1), FT(-2e-4)),
                                   InterfaceVelocities(FT(0), FT(0)),
                                   FT(300), FT(8e-3), (saturation=FT(0.6),), (temperature=FT(300),))
        Ψᵢ = (u=FT(0), v=FT(0), T=FT(300))
        Ψₐ = (T=FT(290), p=FT(1e5), q=FT(6e-3), z=FT(10))
        ℙₐ = (thermodynamics_parameters=AtmosphereThermodynamicsParameters(FT), gravitational_acceleration=FT(9.81))
        rad = (σ=FT(0), α=FT(0), ϵ=FT(0), ℐꜜˢʷ=FT(0), ℐꜜˡʷ=FT(0))
        q_form = FractionalHumidity(AtmosphericThermodynamics.Liquid(); efficiency=FT(1))
        ℙₛ = (specific_humidity_formulation=q_form,)
        T★ = compute_interface_temperature(t, Ψₛ, Ψₐ, Ψᵢ, rad, ℙₛ, ℙₐ, (;))
        @test eltype(T★) == FT
        @inferred compute_interface_temperature(t, Ψₛ, Ψₐ, Ψᵢ, rad, ℙₛ, ℙₐ, (;))
    end
end
