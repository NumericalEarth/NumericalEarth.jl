include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, DryLayerHumidity, StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity, ConstantTortuosity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, SoilSkinTemperature, canopy_air_space_solve,
    AirLandInterfaceState, InterfaceFluxScales, InterfaceVelocities, AirLandRadiationState
using NumericalEarth.Atmospheres: PrescribedAtmosphere, AtmosphereThermodynamicsParameters
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties

build_canopy_air_space(FT) = CanopyAirSpace(FT;
    soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                    dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
        vapor_exchange  = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                      molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.05, porosity = 0.4),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 3.0, moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05))

# Coupled single-column model with the CanopyAirSpace in both interface slots.
function canopy_air_space_model(arch, cas; shortwave = 600.0)
    FT = Float64
    grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                 z = (-1, 0), topology = (Flat, Flat, Bounded))
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
    fill!(parent(atmosphere.temperature), 300.0)
    fill!(parent(atmosphere.specific_humidity), 0.008)
    fill!(parent(atmosphere.velocities.u), 3.0)
    fill!(parent(atmosphere.pressure), 101325.0)
    land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
    set!(land; T = 298.0)
    fill!(parent(land.water_storage), 45.0)   # 𝒮 = 0.3
    radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                    land_surface = SurfaceRadiationProperties(0.2, 0.95))
    fill!(parent(radiation.downwelling_shortwave), shortwave)
    fill!(parent(radiation.downwelling_longwave), 350.0)
    update_state!(radiation)
    model = AtmosphereLandModel(atmosphere, land; radiation,
                atmosphere_land_interface_temperature = cas,
                atmosphere_land_interface_specific_humidity = cas)
    update_state!(model.land)
    update_state!(model)
    return model
end

@testset "CanopyAirSpace" begin
    for arch in test_architectures
        cas = build_canopy_air_space(Float64)
        model = canopy_air_space_model(arch, cas)
        ali = model.interfaces.atmosphere_land_interface
        Ts = ali.temperature

        # The CAS interface carries its diagnostic temperatures as a NamedTuple of fields.
        @test Ts isa NamedTuple
        Tᵃᶜ = Array(interior(Ts.interface))[1, 1, 1]
        Tᵛ  = Array(interior(Ts.canopy))[1, 1, 1]
        Tⁱⁿ = Array(interior(Ts.soil_skin))[1, 1, 1]
        Tₑ  = Array(interior(Ts.effective))[1, 1, 1]
        Gᶜ  = Array(interior(Ts.ground_heat_flux))[1, 1, 1]
        𝒬ᵀ  = Array(interior(ali.fluxes.sensible_heat))[1, 1, 1]
        𝒬ᵛ  = Array(interior(ali.fluxes.latent_heat))[1, 1, 1]

        # Finite and physical.
        @test all(isfinite, (Tᵃᶜ, Tᵛ, Tⁱⁿ, Tₑ, Gᶜ, 𝒬ᵀ, 𝒬ᵛ))
        @test 285 < Tᵃᶜ < 320

        # Sunlit: the leaf is warmer than the shaded soil skin, and the node lies between
        # its coolest and warmest sources.
        @test Tⁱⁿ < Tᵛ
        θᵃᵗ = 300.0
        @test min(Tⁱⁿ, Tᵛ, θᵃᵗ) - 1 ≤ Tᵃᶜ ≤ max(Tⁱⁿ, Tᵛ, θᵃᵗ) + 1

        # Conservation: the slab is driven by the skin→bulk conduction, Es = −Gcond.
        Es = Array(interior(model.land.fluxes.surface_energy_flux))[1, 1, 1]
        @test Es ≈ -Gᶜ atol = 1e-6

        # Two-source flux shares: the leaf/ground sensible and latent shares are finite
        # and sum to the atmosphere-facing totals (node continuity).
        Hᵛ  = Array(interior(Ts.canopy_sensible_heat))[1, 1, 1]
        Hᵍ  = Array(interior(Ts.soil_sensible_heat))[1, 1, 1]
        LEᵛ = Array(interior(Ts.canopy_latent_heat))[1, 1, 1]
        LEᵍ = Array(interior(Ts.soil_latent_heat))[1, 1, 1]
        @test all(isfinite, (Hᵛ, Hᵍ, LEᵛ, LEᵍ))
        @test Hᵛ + Hᵍ ≈ 𝒬ᵀ rtol = 1e-2
        @test LEᵛ + LEᵍ ≈ 𝒬ᵛ rtol = 1e-2
        # Sunlit dense canopy: transpiration is the larger latent source.
        @test LEᵛ > LEᵍ

        # A brighter sun warms the leaf.
        model_dark = canopy_air_space_model(arch, cas; shortwave = 0.0)
        Tᵛ_dark = Array(interior(model_dark.interfaces.atmosphere_land_interface.temperature.canopy))[1, 1, 1]
        @test Tᵛ > Tᵛ_dark
    end

    # Non-CAS regression: an ordinary temperature closure keeps a plain-Field interface
    # temperature and adds a radiative contribution (no NamedTuple, no internalized radiation).
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
        fill!(parent(atmosphere.temperature), 290.0); fill!(parent(atmosphere.specific_humidity), 0.006)
        fill!(parent(atmosphere.velocities.u), 5.0); fill!(parent(atmosphere.pressure), 101325.0)
        land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
        set!(land; T = 300.0); fill!(parent(land.water_storage), 90.0)
        model = AtmosphereLandModel(atmosphere, land; radiation = nothing,
                    atmosphere_land_interface_temperature = SoilSkinTemperature(1.5, 0.05))
        update_state!(model.land); update_state!(model)
        T = model.interfaces.atmosphere_land_interface.temperature
        @test T isa Oceananigans.Fields.Field
        @test Array(interior(T))[1, 1, 1] < 300.0   # evaporating skin cooler than the bulk
    end

    # Type stability of the coupled solve (Float32 and Float64).
    for FT in (Float32, Float64)
        cas = build_canopy_air_space(FT)
        ℂ = AtmosphereThermodynamicsParameters(FT)
        ℙₐ = (thermodynamics_parameters = ℂ, gravitational_acceleration = FT(9.81))
        Ψₛ = AirLandInterfaceState(InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3)),
                                   InterfaceVelocities(FT(0), FT(0)), FT(300), FT(0.012),
                                   (saturation = FT(0.3),), (temperature = FT(298),), (leaf_area_index = FT(3),))
        Ψₐ = (z = FT(10), u = FT(3), v = FT(0), T = FT(300), p = FT(101325), q = FT(0.008), h_bℓ = FT(600))
        Ψᵢ = (u = FT(0), v = FT(0), T = FT(298))
        Ψᵣ = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
        @inferred canopy_air_space_solve(cas, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    end
end
