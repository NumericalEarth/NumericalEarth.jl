include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!, time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    SoilSkinTemperature, EnergyBalanceTemperature, SkinTemperature, SoilConductiveFlux,
    DiagnosticSkin, PrognosticSkin,
    DryLayerHumidity, StorageBasedDryLayerDepth, DryLayerVaporPistonVelocity, ConstantTortuosity
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties,
                                 default_stefan_boltzmann_constant

bare_soil_humidity(FT) = DryLayerHumidity(FT;
    dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
    vapor_exchange  = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                  molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
    thermal_exchange_depth = 0.05, porosity = 0.4)

function bare_soil_model(arch, temperature; shortwave = 600.0, longwave = 350.0,
                         wind = 5.0, Tair = 300.0, qair = 0.008,
                         Tland = 298.0, water = 90.0, α = 0.2, ϵ = 0.95)
    FT = Float64
    grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                 z = (-1, 0), topology = (Flat, Flat, Bounded))
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
    fill!(parent(atmosphere.temperature), Tair)
    fill!(parent(atmosphere.specific_humidity), qair)
    fill!(parent(atmosphere.velocities.u), wind)
    fill!(parent(atmosphere.pressure), 101325.0)
    land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
    set!(land; T = Tland)
    fill!(parent(land.water_storage), water)
    radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                    land_surface = SurfaceRadiationProperties(α, ϵ))
    fill!(parent(radiation.downwelling_shortwave), shortwave)
    fill!(parent(radiation.downwelling_longwave), longwave)
    update_state!(radiation)
    model = AtmosphereLandModel(atmosphere, land; radiation,
                atmosphere_land_interface_temperature = temperature,
                atmosphere_land_interface_specific_humidity = bare_soil_humidity(FT))
    update_state!(model.land)
    update_state!(model)
    return model
end

@inline value1(f) = Array(interior(f))[1, 1, 1]

# Surface energy-balance terms from the model outputs: Rₙ + G − H − LE at the
# stored skin, with Λ = 30 and the test's radiation properties.
function surface_imbalance(model; SW = 600.0, LW = 350.0, α = 0.2, ϵ = 0.95, Λ = 30.0)
    ai = model.interfaces.atmosphere_land_interface
    σ  = default_stefan_boltzmann_constant
    Tₛ = value1(ai.temperature)
    Tˡ = value1(model.land.temperature)
    Rn = (1 - α) * SW + ϵ * (LW - σ * Tₛ^4)
    G  = Λ * (Tˡ - Tₛ)
    H  = value1(ai.fluxes.sensible_heat)
    LE = value1(ai.fluxes.latent_heat)
    return Rn + G - H - LE, Tₛ
end

@testset "Prognostic energy-balance skin" begin
    for arch in test_architectures
        # --- diagnostic default: DiagnosticSkin() keeps the massless root,
        # behaviorally identical to SkinTemperature(SoilConductiveFlux).
        m_ebt = bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05; max_ΔT = 50))
        m_st  = bare_soil_model(arch, SkinTemperature(SoilConductiveFlux(1.5, 0.05); max_ΔT = 50))
        @test value1(m_ebt.interfaces.atmosphere_land_interface.temperature) ≈
              value1(m_st.interfaces.atmosphere_land_interface.temperature) atol = 1e-8

        # --- windy daytime: the prognostic skin relaxes onto the diagnostic answer.
        Tform = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = 1e5))
        mp = bare_soil_model(arch, Tform)
        md = bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05; max_ΔT = 50))
        for _ in 1:36
            time_step!(mp, 300.0)
            time_step!(md, 300.0)
        end
        Tp = value1(mp.interfaces.atmosphere_land_interface.temperature)
        Td = value1(md.interfaces.atmosphere_land_interface.temperature)
        @test Tp ≈ Td atol = 0.5
        @test value1(mp.interfaces.atmosphere_land_interface.fluxes.latent_heat) ≈
              value1(md.interfaces.atmosphere_land_interface.fluxes.latent_heat) atol = 10

        # --- calm moist transition (the issue-549 bare-soil exemplar): the prognostic
        # skin closes the surface energy balance through its storage tendency instead
        # of silently violating it, and every exit stays bounded.
        C, Δt = 1e5, 300.0
        mp = bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = C));
                             shortwave = 50.0, wind = 0.2, Tair = 304.0,
                             Tland = 310.0, water = 135.0)
        ai = mp.interfaces.atmosphere_land_interface
        worst = 0.0
        Tₛ⁻ = value1(ai.temperature)
        for _ in 1:48
            time_step!(mp, Δt)
            F, Tₛ = surface_imbalance(mp; SW = 50.0)
            residual = F - C * (Tₛ - Tₛ⁻) / Δt   # imbalance beyond storage: linearization error only
            worst = max(worst, abs(residual))
            Tₛ⁻ = Tₛ
            @test isfinite(value1(ai.fluxes.latent_heat))
            @test abs(value1(ai.fluxes.latent_heat)) < 2000
        end
        @test worst < 15
    end
end
