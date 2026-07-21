include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, CanopyInterception, DryLayerHumidity, StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity, ConstantTortuosity, PowerLawTortuosity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, SoilSkinTemperature, canopy_air_space_solve, dry_layer_terms,
    compute_interface_temperature, compute_interface_humidity, interface_temperature_and_humidity,
    saturation_specific_humidity, default_dry_air_molar_mass, AtmosphericThermodynamics,
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

# The wet-canopy vapor mass conductance is g_wet = ρᵃᵗ·LAI·gᵇ. A molar-mass factor (Mᵈ ≈ 0.029)
# in place of the air density (ρᵃᵗ ≈ 1.2) would make it ~40× too small — smaller than the dry
# stomatal conductance, so a wet leaf would evaporate *slower* than a dry one.
@testset "Wet-canopy vapor conductance scales with air density" begin
    FT = Float64
    ℂ  = AtmosphereThermodynamicsParameters(FT)
    ℙₐ = (thermodynamics_parameters = ℂ, gravitational_acceleration = FT(9.81))
    LAI = 3.0; gᵇ = 0.02; c = 0.1
    cas = CanopyAirSpace(FT;
        soil = DryLayerHumidity(FT;
            dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
            vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
            thermal_exchange_depth = 0.05, porosity = 0.4),
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = LAI,
                                moisture_stress = CriticalSaturation(0.5), absorbed_par = InteractiveAbsorbedPAR(FT)),
        leaf_boundary_conductance = gᵇ,
        interception = CanopyInterception())

    Ψₐ  = (z = FT(10), u = FT(3), v = FT(0), T = FT(305), p = FT(101325), q = FT(0.006), h_bℓ = FT(600))  # dry, warm → demand
    Ψᵢ  = (u = FT(0), v = FT(0), T = FT(298))
    Ψᵣ  = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
    flx = InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3)); vel = InterfaceVelocities(FT(0), FT(0))
    Wᶜᵐᵃˣ = c * LAI

    Ψwet = AirLandInterfaceState(flx, vel, FT(300), FT(0.012),
            (saturation = FT(0.3), canopy_water_storage = FT(Wᶜᵐᵃˣ), canopy_water_capacity = FT(Wᶜᵐᵃˣ)), (temperature = FT(298),), (leaf_area_index = FT(LAI),))
    Ψdry = AirLandInterfaceState(flx, vel, FT(300), FT(0.012),
            (saturation = FT(0.3), canopy_water_storage = FT(0), canopy_water_capacity = FT(Wᶜᵐᵃˣ)), (temperature = FT(298),), (leaf_area_index = FT(LAI),))
    wet = canopy_air_space_solve(cas, Ψwet, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    dry = canopy_air_space_solve(cas, Ψdry, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

    @test wet.LEᵛ > dry.LEᵛ        # a wet leaf evaporates faster than the dry (stomatal) leaf
    @test wet.E_wet > 0
    @test dry.E_wet == 0

    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    qᵛ  = saturation_specific_humidity(ℂ, wet.Tᵛ, Ψₐ.p, cas.phase)
    E_ρ = (ρᵃᵗ * LAI * gᵇ) * (qᵛ - wet.qᵃᶜ)                          # correct (air density)
    E_M = (default_dry_air_molar_mass * LAI * gᵇ) * (qᵛ - wet.qᵃᶜ)   # erroneous (molar mass)
    @test wet.E_wet ≈ E_ρ rtol = 1e-6
    @test wet.E_wet / E_M ≈ ρᵃᵗ / default_dry_air_molar_mass rtol = 1e-3   # ≈ 40, not 1
end

# The CanopyAirSpace soil branch blends the dry-layer series conductance with the saturated-skin
# wet branch (weight `f_dry` from the soil model). With a Millington–Quirk (power-law) tortuosity
# the raw Gᵉ collapses to ≈ 0 at saturation; the blend must keep the soil evaporating.
@testset "Saturated soil keeps evaporating (dry-layer wet blend in CanopyAirSpace)" begin
    FT = Float64
    ℂ  = AtmosphereThermodynamicsParameters(FT)
    ℙₐ = (thermodynamics_parameters = ℂ, gravitational_acceleration = FT(9.81))
    soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                            dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
        vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                            molecular_diffusivity = 2.4e-5, tortuosity = PowerLawTortuosity()),
        thermal_exchange_depth = 0.05, porosity = 0.4)
    # Bare soil (LAI = 0) isolates the soil branch from the canopy.
    bare(gᵘᶜ) = CanopyAirSpace(FT; soil,
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = 0.0,
                            moisture_stress = CriticalSaturation(0.5), absorbed_par = InteractiveAbsorbedPAR(FT)),
        soil_skin_flux = SoilConductiveFlux(1.5, 0.05), undercanopy_conductance = gᵘᶜ)

    Ψₐ  = (z = FT(10), u = FT(3), v = FT(0), T = FT(305), p = FT(101325), q = FT(0.006), h_bℓ = FT(600))
    Ψᵢ  = (u = FT(0), v = FT(0), T = FT(298))
    Ψᵣ  = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
    flx = InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3)); vel = InterfaceVelocities(FT(0), FT(0))
    Ψ(𝒮) = AirLandInterfaceState(flx, vel, FT(300), FT(0.012), (saturation = FT(𝒮),),
            (temperature = FT(300),), (leaf_area_index = FT(0),))
    LEᵍ(gᵘᶜ, 𝒮) = canopy_air_space_solve(bare(gᵘᶜ), Ψ(𝒮), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ).LEᵍ

    # A saturated soil evaporates a substantial positive latent flux (the pre-fix stall gives ≈ 0),
    # rising monotonically as the soil↔canopy-air path opens up.
    E = [LEᵍ(g, 0.99) for g in (0.05, 0.5, 5.0)]
    @test all(E .> 50)
    @test issorted(E)

    # Dry limit: at low saturation the dry-branch weight f_dry ≈ 1, so the blended soil
    # conductance reduces to the raw dry-layer Gᵉ (the blend is inactive where the soil is dry).
    Gᵉ, qᵉ, f_dry, qⁱⁿ⁺ = dry_layer_terms(soil, FT(300), Ψ(0.1), Ψₐ, ℙₐ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    @test f_dry > 0.99
    @test f_dry * Gᵉ + (1 - f_dry) * (ρᵃᵗ * 0.5) ≈ Gᵉ rtol = 0.02
end

# A CanopyAirSpace in both interface slots is a combined formulation: one shared solve returns
# both Tᵃᶜ and qᵃᶜ. This must be bit-identical to running the two separate solves.
@testset "Combined CanopyAirSpace solve equals separate temperature/humidity solves" begin
    for FT in (Float32, Float64)
        ℂ  = AtmosphereThermodynamicsParameters(FT)
        ℙₐ = (thermodynamics_parameters = ℂ, gravitational_acceleration = FT(9.81))
        cas = CanopyAirSpace(FT;
            soil = DryLayerHumidity(FT;
                dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                    dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
                vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                    molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
                thermal_exchange_depth = 0.05, porosity = 0.4),
            canopy = CanopyConductanceHumidity(FT; leaf_area_index = 3.0,
                                    moisture_stress = CriticalSaturation(0.5), absorbed_par = InteractiveAbsorbedPAR(FT)))
        Ψₛ = AirLandInterfaceState(InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3)),
                InterfaceVelocities(FT(0), FT(0)), FT(300), FT(0.012),
                (saturation = FT(0.3),), (temperature = FT(298),), (leaf_area_index = FT(3),))
        Ψₐ = (z = FT(10), u = FT(3), v = FT(0), T = FT(300), p = FT(101325), q = FT(0.008), h_bℓ = FT(600))
        Ψᵢ = (u = FT(0), v = FT(0), T = FT(298))
        Ψᵣ = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))

        Tₛ = compute_interface_temperature(cas, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ, ℙₐ, ℙₐ)
        qₛ = compute_interface_humidity(cas, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
        Tc, qc = interface_temperature_and_humidity(cas, cas, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ, ℙₐ, ℙₐ)
        @test Tc === Tₛ
        @test qc === qₛ
    end
end
