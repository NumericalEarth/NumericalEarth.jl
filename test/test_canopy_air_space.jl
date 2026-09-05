include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, CanopyInterception,
    DryLayerHumidity, StorageBasedDryLayerDepth, DryLayerVaporPistonVelocity, PowerLawTortuosity,
    CriticalSaturation, InteractiveAbsorbedPAR, absorbed_par_value,
    SoilConductiveFlux, SoilSkinTemperature, canopy_air_space_solve, dry_layer_terms,
    saturation_specific_humidity, default_dry_air_molar_mass, AtmosphericThermodynamics,
    AirLandInterfaceState, InterfaceFluxScales, InterfaceVelocities, AirLandRadiationState,
    AreaIndexUndercanopyConductance, FrictionVelocityUndercanopyConductance, undercanopy_conductance,
    SellersSoilResistance, LitterResistance, soil_surface_resistance, litter_resistance,
    CanopyAirSpaceDiagnostics, DiagnosticSkin,
    default_atmosphere_land_fluxes
using NumericalEarth.Atmospheres: PrescribedAtmosphere, AtmosphereThermodynamicsParameters
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties

build_canopy_air_space(FT; optics...) = CanopyAirSpace(FT;
    soil = bare_soil_humidity(FT),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05), optics...)

@testset "CanopyAirSpace" begin
    for arch in test_architectures
        cas = build_canopy_air_space(Float64)
        model = coupled_land_model(arch; atmosphere_land_interface_temperature = cas,
                                   atmosphere_land_interface_specific_humidity = cas)
        ali = model.interfaces.atmosphere_land_interface
        Ts = ali.temperature

        @test Ts isa CanopyAirSpaceDiagnostics
        Tᵃᶜ = Array(interior(Ts.interface))[1, 1, 1]
        Tˡᵉᵃᶠ  = Array(interior(Ts.canopy))[1, 1, 1]
        Tᵍ = Array(interior(Ts.soil_skin))[1, 1, 1]
        Tₑ  = Array(interior(Ts.effective))[1, 1, 1]
        𝒬ᵍ  = Array(interior(Ts.ground_heat_flux))[1, 1, 1]
        𝒬ᵀ  = Array(interior(ali.fluxes.sensible_heat))[1, 1, 1]
        𝒬ᵛ  = Array(interior(ali.fluxes.latent_heat))[1, 1, 1]

        # Finite and physical.
        @test all(isfinite, (Tᵃᶜ, Tˡᵉᵃᶠ, Tᵍ, Tₑ, 𝒬ᵍ, 𝒬ᵀ, 𝒬ᵛ))
        @test 285 < Tᵃᶜ < 320

        # Sunlit: the leaf is warmer than the shaded soil skin, and the node lies between
        # its coolest and warmest sources.
        @test Tᵍ < Tˡᵉᵃᶠ
        θᵃᵗ = 300.0
        @test min(Tᵍ, Tˡᵉᵃᶠ, θᵃᵗ) - 1 ≤ Tᵃᶜ ≤ max(Tᵍ, Tˡᵉᵃᶠ, θᵃᵗ) + 1

        # Conservation: the slab is driven by the skin→bulk conduction, Es = −𝒬ᵍ.
        Es = Array(interior(model.land.fluxes.surface_energy_flux))[1, 1, 1]
        @test Es ≈ -𝒬ᵍ atol = 1e-6

        # Two-source flux shares sum to the atmosphere-facing totals (node continuity).
        Hˡᵉᵃᶠ  = Array(interior(Ts.canopy_sensible_heat))[1, 1, 1]
        Hᵍ  = Array(interior(Ts.soil_sensible_heat))[1, 1, 1]
        LEˡᵉᵃᶠ = Array(interior(Ts.canopy_latent_heat))[1, 1, 1]
        LEᵍ = Array(interior(Ts.soil_latent_heat))[1, 1, 1]
        @test all(isfinite, (Hˡᵉᵃᶠ, Hᵍ, LEˡᵉᵃᶠ, LEᵍ))
        @test Hˡᵉᵃᶠ + Hᵍ ≈ 𝒬ᵀ rtol = 1e-6
        @test LEˡᵉᵃᶠ + LEᵍ ≈ 𝒬ᵛ rtol = 1e-6
        # Sunlit dense canopy: transpiration is the larger latent source.
        @test LEˡᵉᵃᶠ > LEᵍ

        # A brighter sun warms the leaf.
        model_dark = coupled_land_model(arch; shortwave = 0.0, atmosphere_land_interface_temperature = cas,
                                        atmosphere_land_interface_specific_humidity = cas)
        Tˡᵉᵃᶠ_dark = Array(interior(model_dark.interfaces.atmosphere_land_interface.temperature.canopy))[1, 1, 1]
        @test Tˡᵉᵃᶠ > Tˡᵉᵃᶠ_dark
    end

    # An ordinary temperature closure keeps a plain-Field interface temperature.
    for arch in test_architectures
        model = coupled_land_model(arch; Tair = 290.0, qair = 0.006, wind = 5.0, Tland = 300.0, water = 90.0,
                                   radiation = nothing,
                                   atmosphere_land_interface_temperature = SoilSkinTemperature(1.5, 0.05; storage = DiagnosticSkin()))
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
                                   (saturation = FT(0.3),), (temperature = FT(298), time_step = FT(0)), (leaf_area_index = FT(3),))
        Ψₐ = (z = FT(10), u = FT(3), v = FT(0), T = FT(300), p = FT(101325), q = FT(0.008), h_bℓ = FT(600))
        Ψᵢ = (u = FT(0), v = FT(0), T = FT(298))
        Ψᵣ = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
        @inferred canopy_air_space_solve(cas, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    end
end

@testset "Absorbed PAR absorbs on the canopy's own structure" begin
    FT = Float64
    par = InteractiveAbsorbedPAR(FT; leaf_albedo_par = 0.08, extinction = 0.5, clumping = 1)
    Ψᵣ = AirLandRadiationState(FT(5.67e-8), FT(0.2), FT(0.95), FT(800), FT(350))
    LAI = FT(3)
    incident = par.par_fraction * Ψᵣ.ℐꜜˢʷ * par.photon_per_joule

    # With no canopy to supply one, the closure builds the transmittance from its own geometry.
    own = exp(-par.extinction * LAI * par.clumping)
    @test absorbed_par_value(par, Ψᵣ, LAI, nothing) ≈
          (1 - par.leaf_albedo_par) * (1 - own) * incident / LAI

    # A sparse canopy intercepts little, so the vanishing absorbed fraction cancels the per-leaf
    # division and a leaf never receives more than the flux falling on it.
    @test absorbed_par_value(par, Ψᵣ, FT(1e-3), nothing) < incident
    @test absorbed_par_value(par, Ψᵣ, FT(0), nothing) == 0

    # A supplied transmittance wins, and the closure's own geometry is not consulted.
    transmittance = FT(0.4)
    supplied = absorbed_par_value(par, Ψᵣ, LAI, transmittance)
    @test supplied ≈ (1 - par.leaf_albedo_par) * (1 - transmittance) * incident / LAI
    @test absorbed_par_value(InteractiveAbsorbedPAR(FT; leaf_albedo_par = 0.08, extinction = 0.9),
                             Ψᵣ, LAI, transmittance) ≈ supplied

    # A canopy hands down the transmittance its own shortwave split uses.
    cas = CanopyAirSpace(FT; soil = bare_soil_humidity(FT),
                         canopy = CanopyConductanceHumidity(FT; leaf_area_index = LAI, absorbed_par = par),
                         extinction = 0.55, clumping = 0.75)
    @test exp(-cas.extinction * LAI * cas.clumping) != own

    @test absorbed_par_value(FT(5e-4), Ψᵣ, LAI, FT(0.3)) == FT(5e-4)
end

# Per-cell optics reach the coupled solve: two cells sharing a canopy closure but differing
# in leaf albedo end up with different canopy temperatures, and a cell whose field matches
# the scalar the closure would otherwise carry is unchanged.
@testset "Coupled per-cell canopy optics" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT; size = (2, 1, 1), latitude = (10, 11),
                                     longitude = (10, 12), z = (-1, 0),
                                     topology = (Bounded, Bounded, Bounded))
        atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
        fill!(parent(atmosphere.temperature), 300.0)
        fill!(parent(atmosphere.specific_humidity), 0.008)
        fill!(parent(atmosphere.velocities.u), 3.0)
        fill!(parent(atmosphere.pressure), 101325.0)
        land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
        set!(land; T = 298.0)
        fill!(parent(land.water_storage), 45.0)
        radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                        land_surface = SurfaceRadiationProperties(0.2, 0.95))
        fill!(parent(radiation.downwelling_shortwave), 600.0)
        fill!(parent(radiation.downwelling_longwave), 350.0)
        update_state!(radiation)

        scalar_cas = build_canopy_air_space(FT)

        # Cell 1 keeps the closure's own leaf albedo, cell 2 is a bright leaf.
        leaf_albedo = Field{Center, Center, Nothing}(grid)
        set!(leaf_albedo, (λ, φ) -> ifelse(λ < 11, scalar_cas.leaf_albedo, 0.6))

        function canopy_model(cas)
            interface = atmosphere_land_interface(grid, atmosphere, land;
                                                  fluxes = default_atmosphere_land_fluxes(land, FT),
                                                  temperature = cas, specific_humidity = cas)
            model = AtmosphereLandModel(atmosphere, land; radiation,
                                        atmosphere_land_interface = interface)
            update_state!(model.land)
            update_state!(model)
            return model
        end

        model = canopy_model(build_canopy_air_space(FT; leaf_albedo))
        Tˡᵉᵃᶠ = Array(interior(model.interfaces.atmosphere_land_interface.temperature.canopy))

        # The bright leaf absorbs less shortwave and runs cooler.
        @test Tˡᵉᵃᶠ[2, 1, 1] < Tˡᵉᵃᶠ[1, 1, 1]

        # The cell carrying the closure's own albedo is bit-identical to the scalar run,
        # so a configuration with no per-cell optics is unchanged.
        Tˡᵉᵃᶠ_scalar = Array(interior(canopy_model(scalar_cas).interfaces.atmosphere_land_interface.temperature.canopy))
        @test Tˡᵉᵃᶠ[1, 1, 1] == Tˡᵉᵃᶠ_scalar[1, 1, 1]
    end
end

# The wet-canopy vapor mass conductance is gˡᵇ = ρᵃᵗ·LAI·gᵇ. A molar-mass factor (Mᵈ ≈ 0.029)
# in place of the air density (ρᵃᵗ ≈ 1.2) would make it ~40× too small — smaller than the dry
# stomatal conductance, so a wet leaf would evaporate *slower* than a dry one.
@testset "Wet-canopy vapor conductance scales with air density" begin
    FT = Float64
    ℂ  = AtmosphereThermodynamicsParameters(FT)
    ℙₐ = (thermodynamics_parameters = ℂ, gravitational_acceleration = FT(9.81))
    LAI = 3.0; gᵇ = 0.02; c = 0.1
    cas = CanopyAirSpace(FT;
        soil = bare_soil_humidity(FT),
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = LAI,
                                moisture_stress = CriticalSaturation(0.5), absorbed_par = InteractiveAbsorbedPAR(FT)),
        leaf_boundary_conductance = gᵇ,
        interception = CanopyInterception())

    Ψₐ  = (z = FT(10), u = FT(3), v = FT(0), T = FT(305), p = FT(101325), q = FT(0.006), h_bℓ = FT(600))  # dry, warm → demand
    Ψᵢ  = (u = FT(0), v = FT(0), T = FT(298))
    Ψᵣ  = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
    # χθ = χq = 0.1 gives the node a physical aerodynamic branch (gᵃ = ρ u★ χ).
    flx = InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3), FT(0.1), FT(0.1)); vel = InterfaceVelocities(FT(0), FT(0))
    Wᶜᵐᵃˣ = c * LAI

    Ψwet = AirLandInterfaceState(flx, vel, FT(300), FT(0.012),
            (saturation = FT(0.3), canopy_water_storage = FT(Wᶜᵐᵃˣ), canopy_water_capacity = FT(Wᶜᵐᵃˣ)), (temperature = FT(298), time_step = FT(0)), (leaf_area_index = FT(LAI),))
    Ψdry = AirLandInterfaceState(flx, vel, FT(300), FT(0.012),
            (saturation = FT(0.3), canopy_water_storage = FT(0), canopy_water_capacity = FT(Wᶜᵐᵃˣ)), (temperature = FT(298), time_step = FT(0)), (leaf_area_index = FT(LAI),))
    wet = canopy_air_space_solve(cas, Ψwet, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    dry = canopy_air_space_solve(cas, Ψdry, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

    @test wet.LEˡᵉᵃᶠ > dry.LEˡᵉᵃᶠ        # a wet leaf evaporates faster than the dry (stomatal) leaf
    @test wet.Eʷᵉᵗ > 0
    @test dry.Eʷᵉᵗ == 0

    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    qˡᵉᵃᶠ  = saturation_specific_humidity(ℂ, wet.Tˡᵉᵃᶠ, Ψₐ.p, cas.phase)
    E_ρ = (ρᵃᵗ * LAI * gᵇ) * (qˡᵉᵃᶠ - wet.qᵃᶜ)                          # correct (air density)
    E_M = (default_dry_air_molar_mass * LAI * gᵇ) * (qˡᵉᵃᶠ - wet.qᵃᶜ)   # erroneous (molar mass)
    @test wet.Eʷᵉᵗ ≈ E_ρ rtol = 1e-6
    @test wet.Eʷᵉᵗ / E_M ≈ ρᵃᵗ / default_dry_air_molar_mass rtol = 1e-3   # ≈ 40, not 1
end

# The CanopyAirSpace soil branch blends the dry-layer series conductance with the saturated-skin
# wet branch (weight `fᵈ` from the soil model). With a Millington–Quirk (power-law) tortuosity
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
    # Bare soil (LAI = 0, no litter) isolates the soil branch from the canopy.
    bare(gᵘᶜ) = CanopyAirSpace(FT; soil,
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = 0.0,
                            moisture_stress = CriticalSaturation(0.5), absorbed_par = InteractiveAbsorbedPAR(FT)),
        soil_skin_flux = SoilConductiveFlux(1.5, 0.05), undercanopy_conductance = gᵘᶜ,
        litter_resistance = nothing)

    Ψₐ  = (z = FT(10), u = FT(3), v = FT(0), T = FT(305), p = FT(101325), q = FT(0.006), h_bℓ = FT(600))
    Ψᵢ  = (u = FT(0), v = FT(0), T = FT(298))
    Ψᵣ  = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
    # Vapor-coupled (χq) but thermally near-decoupled (χθ = 0) node, so the sweep
    # isolates the soil↔canopy-air vapor path from the skin-temperature response.
    flx = InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3), FT(0), FT(1/6)); vel = InterfaceVelocities(FT(0), FT(0))
    Ψ(𝒮) = AirLandInterfaceState(flx, vel, FT(300), FT(0.012), (saturation = FT(𝒮),),
            (temperature = FT(300), time_step = FT(0)), (leaf_area_index = FT(0),))
    LEᵍ(gᵘᶜ, 𝒮) = canopy_air_space_solve(bare(gᵘᶜ), Ψ(𝒮), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ).LEᵍ

    # A saturated soil evaporates a substantial positive latent flux (the pre-fix stall gives ≈ 0),
    # rising monotonically as the soil↔canopy-air path opens up.
    E = [LEᵍ(g, 0.99) for g in (0.05, 0.5, 5.0)]
    @test all(E .> 50)
    @test issorted(E)

    # Dry limit: at low saturation the dry-branch weight fᵈ ≈ 1, so the blended soil
    # conductance reduces to the raw dry-layer Gᵉ (the blend is inactive where the soil is dry).
    Gᵉ, qᵉ, fᵈ, qᵍ⁺ = dry_layer_terms(soil, FT(300), Ψ(0.1), Ψₐ, ℙₐ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    @test fᵈ > 0.99
    @test fᵈ * Gᵉ + (1 - fᵈ) * (ρᵃᵗ * 0.5) ≈ Gᵉ rtol = 0.02
end

# Ground-surface resistances: the Sakaguchi & Zeng (2009) litter layer sits in series on
# both ground vapor branches (the default), and the Sellers et al. (1992) FIFE fit is the
# bundled soil-plus-litter alternative on the moist-soil branch.
@testset "Ground-surface resistances (litter layer, Sellers fit)" begin
    for FT in (Float32, Float64)
        # Sellers et al. (1992), Eq. (19): rˢ = exp(8.206 − 4.255 𝒮), saturation clamped.
        r = SellersSoilResistance(FT)
        @test soil_surface_resistance(r, FT(1)) ≈ exp(FT(8.206) - FT(4.255))   # ≈ 52 s m⁻¹
        @test soil_surface_resistance(r, FT(2)) == soil_surface_resistance(r, FT(1))
        @test soil_surface_resistance(r, FT(0.4)) > soil_surface_resistance(r, FT(0.7))
        @test soil_surface_resistance(nothing, FT(0.5)) == 0

        # Sakaguchi & Zeng (2009), Eq. (13): rˡ = (1 − e^{−Lˡ}) / (C u★).
        l = LitterResistance(FT)
        @test litter_resistance(l, FT(0.3)) ≈ (1 - exp(-FT(1))) / (FT(0.004) * FT(0.3))  # ≈ 527 s m⁻¹
        @test litter_resistance(LitterResistance(FT; litter_area_index = 0), FT(0.3)) == 0
        @test litter_resistance(l, FT(0.1)) > litter_resistance(l, FT(0.3))   # calm air blocks more
        @test isfinite(litter_resistance(l, FT(0)))
        @test litter_resistance(nothing, FT(0.3)) == 0
    end

    FT = Float64
    ℂ  = AtmosphereThermodynamicsParameters(FT)
    ℙₐ = (thermodynamics_parameters = ℂ, gravitational_acceleration = FT(9.81))
    soil = bare_soil_humidity(FT)
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 0.0,   # LAI = 0 isolates the ground branch
                            moisture_stress = CriticalSaturation(0.5), absorbed_par = InteractiveAbsorbedPAR(FT))
    cas(; kw...) = CanopyAirSpace(FT; soil, canopy, kw...)
    Ψₐ  = (z = FT(10), u = FT(3), v = FT(0), T = FT(305), p = FT(101325), q = FT(0.006), h_bℓ = FT(600))
    Ψᵢ  = (u = FT(0), v = FT(0), T = FT(298))
    Ψᵣ  = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
    # Vapor-coupled (χq) but thermally near-decoupled (χθ = 0) node, so the
    # comparison isolates the ground vapor path from the skin-temperature response.
    flx = InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3), FT(0), FT(1/6)); vel = InterfaceVelocities(FT(0), FT(0))
    Ψ(𝒮) = AirLandInterfaceState(flx, vel, FT(300), FT(0.012), (saturation = FT(𝒮),),
            (temperature = FT(300), time_step = FT(0)), (leaf_area_index = FT(0),))
    LEᵍ(c, 𝒮) = canopy_air_space_solve(c, Ψ(𝒮), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ).LEᵍ

    unresisted = cas(litter_resistance = nothing)
    sellers    = cas(litter_resistance = nothing, wet_soil_resistance = SellersSoilResistance(FT))
    litter     = cas()   # the default configuration

    # Each resistance suppresses moist-soil evaporation; at u★ = 0.26 the litter layer
    # (rˡ ≈ 608 s m⁻¹) blocks more than the Sellers fit (rˢ(0.9) ≈ 113 s m⁻¹).
    @test LEᵍ(litter, 0.9) < LEᵍ(sellers, 0.9) < LEᵍ(unresisted, 0.9)

    # With the litter + undercanopy path in series on both branches, soil evaporation no
    # longer rebounds as the soil dries through the wet → dry-layer handoff (the Sellers-only
    # configuration rebounds because the young dry layer is far more conductive than the fit).
    E = [LEᵍ(litter, 𝒮) for 𝒮 in 0.50:-0.03:0.14]
    @test issorted(E, rev = true)

    # Every resistance configuration keeps the coupled solve inferred.
    for c in (unresisted, sellers, litter)
        @inferred canopy_air_space_solve(c, Ψ(0.5), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    end
end

# The ground↔canopy-air conductance is a closure: a bare number keeps the constant
# behavior bit-for-bit, and the area-index closure responds to canopy density and wind.
@testset "Undercanopy conductance closures" begin
    for FT in (Float32, Float64)
        # A bare number is a constant conductance.
        cas = build_canopy_air_space(FT)
        @test undercanopy_conductance(cas.undercanopy_conductance, FT(3), FT(5), FT(0.3)) === FT(0.013)

        u = AreaIndexUndercanopyConductance(FT)
        g(LAI, Vₐ, u★) = undercanopy_conductance(u, FT(LAI), FT(Vₐ), FT(u★))

        # Denser canopy shields the ground more strongly; stronger wind ventilates it.
        @test g(0.5, 3, 0.26) > g(5, 3, 0.26)
        @test g(3, 5, 0.26) > g(3, 1, 0.26)

        # The sparse-canopy limit binds at the shielding floor and the aerodynamic cap,
        # never producing an infinity.
        @test isfinite(g(0, 3, 0.26))
        @test g(0, 3, 0.26) <= FT(0.26)^2 / 3

        # Stems shield like leaves.
        u_stems = AreaIndexUndercanopyConductance(FT; stem_area_index = 1)
        @test undercanopy_conductance(u_stems, FT(1), FT(3), FT(0.26)) < g(1, 3, 0.26)

        # Calm air shuts the exchange down.
        @test g(3, 0, 0.26) == 0

        # CLM5 closure (Zeng et al. 2005): u★-driven, blending a bare roughness-Reynolds
        # law with the dense-canopy constant Cₛᵈ·u★.
        z = FrictionVelocityUndercanopyConductance(FT)
        gz(LAI, u★) = undercanopy_conductance(z, FT(LAI), FT(3), FT(u★))

        # Dense limit: Cₛᵈ·u★; bare ground ventilates faster than a dense canopy.
        @test gz(20, 0.3) ≈ FT(0.004) * FT(0.3) rtol = 1e-3
        @test gz(0, 0.3) > gz(5, 0.3)

        # Bare endpoint follows the roughness-Reynolds law (k/0.13)(z₀ᵍu★/ν)^(−0.45)·u★.
        Cₛᵇ = (FT(0.4) / FT(0.13)) * (FT(0.01) * FT(0.3) / FT(1.5e-5))^(-FT(0.45))
        @test gz(0, 0.3) ≈ Cₛᵇ * FT(0.3) rtol = 1e-5

        # Shear drives the exchange: calm air (u★ → 0) decouples the ground, finitely.
        @test gz(3, 0) == 0
        @test isfinite(gz(0, 0))

        # A rougher ground slows the bare exchange (thicker interfacial sublayer).
        z_rough = FrictionVelocityUndercanopyConductance(FT; ground_roughness_length = 0.05)
        @test undercanopy_conductance(z_rough, FT(0), FT(3), FT(0.3)) < gz(0, 0.3)

        # Stems shield like leaves here too.
        z_stems = FrictionVelocityUndercanopyConductance(FT; stem_area_index = 1)
        @test undercanopy_conductance(z_stems, FT(1), FT(3), FT(0.3)) < gz(1, 0.3)
    end

    # Every closure keeps the coupled solve inferred.
    for FT in (Float32, Float64)
        ℂ  = AtmosphereThermodynamicsParameters(FT)
        ℙₐ = (thermodynamics_parameters = ℂ, gravitational_acceleration = FT(9.81))
        Ψₐ = (z = FT(10), u = FT(3), v = FT(0), T = FT(300), p = FT(101325), q = FT(0.008), h_bℓ = FT(600))
        Ψᵢ = (u = FT(0), v = FT(0), T = FT(298))
        Ψᵣ = AirLandRadiationState(FT(5.670374e-8), FT(0), FT(0), FT(600), FT(350))
        Ψ(LAI) = AirLandInterfaceState(InterfaceFluxScales(FT(0.26), FT(1e-3), FT(-1e-3), FT(0.1), FT(0.1)),
                                       InterfaceVelocities(FT(0), FT(0)), FT(300), FT(0.012),
                                       (saturation = FT(0.3),), (temperature = FT(298), time_step = FT(0)),
                                       (leaf_area_index = FT(LAI),))

        soil = bare_soil_humidity(FT)
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = 3.0,
                                moisture_stress = CriticalSaturation(0.5), absorbed_par = InteractiveAbsorbedPAR(FT))
        with_undercanopy(gᵘᶜ) = CanopyAirSpace(FT; soil, canopy, undercanopy_conductance = gᵘᶜ)

        @inferred canopy_air_space_solve(with_undercanopy(0.013), Ψ(3), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

        area_index_cas = with_undercanopy(AreaIndexUndercanopyConductance(FT))
        @inferred canopy_air_space_solve(area_index_cas, Ψ(3), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

        friction_cas = with_undercanopy(FrictionVelocityUndercanopyConductance(FT))
        @inferred canopy_air_space_solve(friction_cas, Ψ(3), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

        # Two-source partition responds to canopy density: under identical forcing the
        # sparse canopy routes a larger share of the total latent flux through the soil.
        sparse = canopy_air_space_solve(area_index_cas, Ψ(FT(0.5)), Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
        closed = canopy_air_space_solve(area_index_cas, Ψ(FT(5)),   Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
        soil_share(sol) = sol.LEᵍ / (sol.LEᵍ + sol.LEˡᵉᵃᶠ)
        @test soil_share(sparse) > soil_share(closed)
    end
end
