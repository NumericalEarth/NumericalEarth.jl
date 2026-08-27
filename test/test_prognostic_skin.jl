include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: update_state!, time_step!
using Oceananigans.Architectures: Adapt
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    SoilSkinTemperature, EnergyBalanceTemperature, SoilConductiveFlux,
    InterfaceProperties,
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

bare_soil_column(arch, FT = Float64) =
    LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                          z = (-1, 0), topology = (Flat, Flat, Bounded))

# Two independent columns side by side, for per-cell (Field-valued) properties.
bare_soil_pair(arch, FT = Float64) =
    LatitudeLongitudeGrid(arch, FT; size = (2, 1, 1), longitude = (10, 12), latitude = (10, 11),
                          z = (-1, 0), topology = (Bounded, Bounded, Bounded))

function bare_soil_model(arch, temperature; grid = bare_soil_column(arch),
                         shortwave = 600.0, longwave = 350.0,
                         wind = 5.0, Tair = 300.0, qair = 0.008,
                         Tland = 298.0, water = 90.0, α = 0.2, ϵ = 0.95)
    FT = Float64
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
        # --- windy daytime: the prognostic skin relaxes onto the diagnostic answer.
        Tform = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = 1e5))
        mp = bare_soil_model(arch, Tform)
        md = bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05; storage = DiagnosticSkin()))
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

@testset "PrognosticSkin is the default storage" begin
    @test SoilSkinTemperature(1.5, 0.05).storage isa PrognosticSkin
    @test EnergyBalanceTemperature(SoilConductiveFlux(1.5, 0.05)).storage isa PrognosticSkin
    @test SoilSkinTemperature(1.5, 0.05; storage = DiagnosticSkin()).storage isa DiagnosticSkin

    for FT in (Float32, Float64)
        @test eltype(PrognosticSkin(FT).heat_capacity) == FT
    end

    # The first `update_state!` (Δt = 0) lands on the energy-balance root at the
    # converged similarity scales, so the default prognostic skin starts near the
    # massless answer rather than at the bulk temperature it is seeded from. It is
    # not identical: the massless root iterates Tₛ jointly with u★, while the
    # prognostic skin is frozen through the fixed point by construction.
    for arch in test_architectures
        Tbulk = 298.0
        Tp = value1(bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05)).interfaces.atmosphere_land_interface.temperature)
        Td = value1(bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05; storage = DiagnosticSkin())).interfaces.atmosphere_land_interface.temperature)
        @test Tp ≈ Td atol = 1
        @test abs(Tp - Td) < abs(Tbulk - Td)
    end
end

@testset "Field-valued skin heat capacity" begin
    for arch in test_architectures
        # A `Field` of capacities must reproduce, cell by cell, the scalar runs it
        # interpolates between. Same grid and same forcing throughout, so the only
        # difference between the three models is C.
        skin(C) = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = C))
        calm = (; shortwave = 50.0, wind = 0.2, Tair = 304.0, Tland = 310.0, water = 135.0)

        grid = bare_soil_pair(arch)
        Cfield = Field{Center, Center, Nothing}(grid)
        set!(Cfield, reshape([1e4, 1e6], 2, 1, 1))

        mixed = bare_soil_model(arch, skin(Cfield); grid, calm...)
        light = bare_soil_model(arch, skin(1e4);    grid, calm...)
        heavy = bare_soil_model(arch, skin(1e6);    grid, calm...)
        for _ in 1:12
            time_step!(mixed, 300.0)
            time_step!(light, 300.0)
            time_step!(heavy, 300.0)
        end

        surface(m) = Array(interior(m.interfaces.atmosphere_land_interface.temperature))[:, 1, 1]
        Tm, Tl, Th = surface(mixed), surface(light), surface(heavy)

        @test all(isfinite, Tm)
        @test Tm[1] ≈ Tl[1] atol = 1e-10   # light cell tracks the C = 1e4 run
        @test Tm[2] ≈ Th[2] atol = 1e-10   # heavy cell tracks the C = 1e6 run
        @test !isapprox(Tl[1], Th[1]; atol = 1e-3)   # ...and the capacities are distinguishable
    end
end

# Λᵍ = 0 fully decouples the skin, and `radiation = nothing` zeroes ϵ, so with calm air
# the conductance sum Σλ has no term left. There is then no root to solve for, and both
# storage paths must hold the skin rather than divide by zero.
@testset "Fully decoupled skin stays finite" begin
    for arch in test_architectures
        for storage in (DiagnosticSkin(), PrognosticSkin(heat_capacity = 1e5))
            model = bare_soil_model(arch, SoilSkinTemperature(0, 0.05; storage);
                                    shortwave = 0.0, longwave = 0.0, wind = 0.0, ϵ = 0)
            @test isfinite(value1(model.interfaces.atmosphere_land_interface.temperature))
            for _ in 1:3
                time_step!(model, 300.0)
            end
            ai = model.interfaces.atmosphere_land_interface
            @test isfinite(value1(ai.temperature))
            @test isfinite(value1(ai.fluxes.latent_heat))
            @test isfinite(value1(ai.fluxes.sensible_heat))
        end
    end
end

# A Field-valued heat capacity must reach the device through every wrapper on the path.
@testset "Field-valued capacity survives adapt" begin
    grid = bare_soil_pair(CPU())
    C = Field{Center, Center, Nothing}(grid)
    leaf = typeof(Adapt.adapt(Array, PrognosticSkin(heat_capacity = C)).heat_capacity)

    t = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = C))
    @test typeof(Adapt.adapt(Array, t).storage.heat_capacity) == leaf

    properties = InterfaceProperties(nothing, t, nothing)
    adapted = Adapt.adapt(Array, properties).temperature_formulation
    @test typeof(adapted.storage.heat_capacity) == leaf
end
