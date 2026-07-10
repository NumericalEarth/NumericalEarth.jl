include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    EnergyBalanceTemperature, SoilSkin, SoilSkinTemperature, SkinTemperature,
    BulkTemperature, SoilConductiveFlux, FractionalHumidity,
    balance_conductance, compute_interface_temperature,
    AirLandInterfaceState, InterfaceFluxScales, InterfaceVelocities
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Atmospheres: PrescribedAtmosphere, AtmosphereThermodynamicsParameters
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

# Interface (skin) temperature after solving the coupled single-column interface.
function interface_temperature(arch, temperature; Tair, Tland, efficiency)
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
    fill!(parent(land.water_storage), 90.0)
    model = AtmosphereLandModel(atmosphere, land; radiation=nothing,
                                atmosphere_land_interface_temperature=temperature,
                                atmosphere_land_interface_specific_humidity=FractionalHumidity(AtmosphericThermodynamics.Liquid(); efficiency))
    update_state!(model.land)
    update_state!(model)
    return Array(interior(model.interfaces.atmosphere_land_interface.temperature))[1, 1, 1]
end

@testset "EnergyBalanceTemperature (SoilSkin)" begin
    for arch in test_architectures
        # The SoilSkin instance is behaviorally identical to the equivalent
        # SkinTemperature(SoilConductiveFlux(...)) (Plan B), bit-for-bit.
        Tin_ebt   = interface_temperature(arch, SoilSkinTemperature(1.5, 0.05); Tair=290.0, Tland=300.0, efficiency=1.0)
        Tin_skinT = interface_temperature(arch, SkinTemperature(SoilConductiveFlux(1.5, 0.05); max_ΔT=50.0);
                                          Tair=290.0, Tland=300.0, efficiency=1.0)
        @test Tin_ebt == Tin_skinT

        # Λⁱⁿ → ∞ recovers BulkTemperature.
        @test isapprox(interface_temperature(arch, SoilSkinTemperature(1e7, 0.05); Tair=290.0, Tland=300.0, efficiency=1.0),
                       300.0; atol=1e-2)

        # Evaporating surface: skin cooler than the bulk.
        @test interface_temperature(arch, SoilSkinTemperature(1.5, 0.05); Tair=290.0, Tland=300.0, efficiency=1.0) < 300.0
    end

    # Conductance accessor and type stability.
    for FT in (Float32, Float64)
        @test balance_conductance(SoilSkin(), SoilConductiveFlux(FT(1.5), FT(0.05))) == FT(30)
        t = SoilSkinTemperature(FT(1.5), FT(0.05))
        Ψₛ = AirLandInterfaceState(InterfaceFluxScales(FT(0.3), FT(0.1), FT(-2e-4)),
                                   InterfaceVelocities(FT(0), FT(0)),
                                   FT(300), FT(8e-3), (saturation=FT(0.6),), (temperature=FT(300),))
        Ψᵢ = (u=FT(0), v=FT(0), T=FT(300))
        Ψₐ = (T=FT(290), p=FT(1e5), q=FT(6e-3), z=FT(10))
        ℙₐ = (thermodynamics_parameters=AtmosphereThermodynamicsParameters(FT), gravitational_acceleration=FT(9.81))
        rad = (σ=FT(0), α=FT(0), ϵ=FT(0), ℐꜜˢʷ=FT(0), ℐꜜˡʷ=FT(0))
        T★ = compute_interface_temperature(t, Ψₛ, Ψₐ, Ψᵢ, rad, (;), ℙₐ, (;))
        @test eltype(T★) == FT
        @inferred compute_interface_temperature(t, Ψₛ, Ψₐ, Ψᵢ, rad, (;), ℙₐ, (;))
    end
end
