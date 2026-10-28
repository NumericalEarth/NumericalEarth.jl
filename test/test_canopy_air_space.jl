include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.OutputReaders: OnDisk, InMemory
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyInterface, CanopyConductanceHumidity, EnergyBalanceTemperature,
    DryLayerHumidity, StorageBasedDryLayerDepth, DryLayerVaporPistonVelocity, PowerLawTortuosity,
    CriticalSaturation, AbsorbedPhotosyntheticallyActiveRadiation, absorbed_par_value,
    SoilConductiveFlux, SoilSkinTemperature, canopy_air_space_solve, dry_layer_terms,
    saturation_specific_humidity, AtmosphericThermodynamics,
    AirLandInterfaceState, InterfaceFluxScales, InterfaceVelocities, AirLandRadiationState,
    AreaIndexUndercanopyVelocity, FrictionVelocityUndercanopyVelocity, undercanopy_transfer_velocity,
    ExponentialSoilResistance, GroundSurfaceResistance, soil_surface_resistance, ground_surface_resistance,
    kernel_surface_field, canopy_transmittance
using NumericalEarth: stateindex
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters

canopy_ground_surface(FT; leaf_area_index = 4.0, soil_humidity = bare_soil_humidity(FT), kw...) =
    CanopyInterface(FT; soil_humidity, kw...,
        canopy_humidity = CanopyConductanceHumidity(FT; leaf_area_index, moisture_stress = CriticalSaturation(0.5),
                                                    absorbed_par = AbsorbedPhotosyntheticallyActiveRadiation(FT)))

bare(; kw...) = canopy_ground_surface(Float64; leaf_area_index = 0.0, kw...)

power_law_soil_humidity(FT) = DryLayerHumidity(FT;
    dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
    vapor_exchange  = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                  molecular_diffusivity = 2.4e-5, tortuosity = PowerLawTortuosity()))

function canopy_arguments(FT; air_temperature = 300, air_specific_humidity = 0.008, wind = 3, albedo = 0, emissivity = 0,
                          flux_scales = (0.26, 1e-3, -1e-3, 0, 0), temperature = 300, saturation = 0.3,
                          vapor_diffusivity = 2.4e-5, canopy_water_storage = 0, canopy_water_capacity = 0,
                          land_temperature = 298, leaf_area_index = 3)
    Ψⁱⁿ = AirLandInterfaceState(InterfaceFluxScales(FT.(flux_scales)...), InterfaceVelocities(FT(0), FT(0)),
                               FT(temperature), FT(0.012))
    Ψₐ = (z = FT(10), u = FT(wind), v = FT(0), T = FT(air_temperature), p = FT(101325), q = FT(air_specific_humidity), h_bℓ = FT(600))
    Ψˡᵃ = (T = FT(land_temperature), 𝒮 = FT(saturation), effective_vapor_diffusivity = FT(vapor_diffusivity),
           canopy_water_storage = FT(canopy_water_storage), canopy_water_capacity = FT(canopy_water_capacity),
           surface_storage = nothing, thermal_exchange_depth = FT(0.05), leaf_area_index = FT(leaf_area_index))
    Ψᵣ = AirLandRadiationState(FT(5.670374e-8), FT(albedo), FT(emissivity), FT(-600), FT(-350))
    ℙₐ = (thermodynamics_parameters = AtmosphereThermodynamicsParameters(FT), gravitational_acceleration = FT(9.81))
    return Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ
end

canopy_solve(surface, FT = Float64; kw...) = canopy_air_space_solve(surface, canopy_arguments(FT; kw...)..., FT(0))

# Dry warm air over a bare (λˡᵉᵃᶠ = 0) ground; Ξθ = 0 decouples the node thermally to isolate the ground vapor path.
ground_vapor_state(𝒮) = (air_temperature = 305, air_specific_humidity = 0.006, flux_scales = (0.26, 1e-3, -1e-3, 0, 1/6),
                         saturation = 𝒮, land_temperature = 300, leaf_area_index = 0)

@testset "CanopyInterface" begin
    canopy_surface = canopy_ground_surface(Float64)
    canopy_model(arch; kw...) = coupled_land_model(arch; atmosphere_land_interface_state = canopy_surface, kw...)

    for arch in test_architectures
        model = canopy_model(arch)
        Ts = model.interfaces.atmosphere_land_interface.temperature
        fluxes = model.interfaces.atmosphere_land_interface.fluxes
        Tᵃᶜᵃⁿ, Tˡᵉᵃᶠ, Tᵍʳⁿᵈ, Tʳᵃᵈ, 𝒬ᵍʳⁿᵈ = scalar.((Ts.interface, Ts.canopy, Ts.ground, Ts.effective, Ts.ground_heat_flux))
        𝒬ᵀ, 𝒬ᵛ = scalar(fluxes.sensible_heat), scalar(fluxes.latent_heat)
        𝒬ᵛˡᵉᵃᶠ, 𝒬ᵛᵍʳⁿᵈ = scalar(Ts.canopy_latent_heat), scalar(Ts.ground_latent_heat)

        @test all(isfinite, (Tᵃᶜᵃⁿ, Tˡᵉᵃᶠ, Tᵍʳⁿᵈ, Tʳᵃᵈ, 𝒬ᵍʳⁿᵈ, 𝒬ᵀ, 𝒬ᵛ))
        @test 285 < Tᵃᶜᵃⁿ < 320
        @test Tᵍʳⁿᵈ < Tˡᵉᵃᶠ
        @test min(Tᵍʳⁿᵈ, Tˡᵉᵃᶠ, 300) - 1 ≤ Tᵃᶜᵃⁿ ≤ max(Tᵍʳⁿᵈ, Tˡᵉᵃᶠ, 300) + 1   # 300 K is the air temperature
        @test scalar(model.land.fluxes.surface_energy_flux) ≈ -𝒬ᵍʳⁿᵈ atol = 1e-6
        @test scalar(Ts.canopy_sensible_heat) + scalar(Ts.ground_sensible_heat) ≈ 𝒬ᵀ rtol = 1e-10
        @test 𝒬ᵛˡᵉᵃᶠ + 𝒬ᵛᵍʳⁿᵈ ≈ 𝒬ᵛ rtol = 1e-10
        @test 𝒬ᵛˡᵉᵃᶠ > 𝒬ᵛᵍʳⁿᵈ
        @test Tˡᵉᵃᶠ > scalar(canopy_model(arch; shortwave = 0.0).interfaces.atmosphere_land_interface.temperature.canopy)

        skin = SoilSkinTemperature(1.5, 0.05)
        model = coupled_land_model(arch; air_temperature = 290.0, air_specific_humidity = 0.006, wind = 5.0,
                                   land_temperature = 300.0, water = 90.0, radiation = nothing,
                                   atmosphere_land_interface_state = (temperature = skin, specific_humidity = BulkHumidity()))
        @test scalar(model.interfaces.atmosphere_land_interface.temperature) < 300.0
    end

    for FT in (Float32, Float64)
        args = canopy_arguments(FT)
        for surface in (canopy_ground_surface(FT),
                        canopy_ground_surface(FT; undercanopy_transfer_velocity = AreaIndexUndercanopyVelocity(FT)),
                        canopy_ground_surface(FT; undercanopy_transfer_velocity = FrictionVelocityUndercanopyVelocity(FT)),
                        canopy_ground_surface(FT; ground_surface_resistance = nothing),
                        canopy_ground_surface(FT; ground_surface_resistance = nothing, wet_soil_resistance = ExponentialSoilResistance(FT)))
            @inferred canopy_air_space_solve(surface, args..., FT(0))
        end
    end
end

# Leaf and ground energy residuals at a canopy solution, rebuilt from the surface optics.
function canopy_residuals(c, solution, Ψˡᵃ, Ψᵣ)
    L, σ = Ψˡᵃ.leaf_area_index, Ψᵣ.σ
    shortwave, longwave = -Ψᵣ.ℐꜜˢʷ, -Ψᵣ.ℐꜜˡʷ
    τ = canopy_transmittance(c.extinction, c.clumping, L)
    ϵˡᵉᵃᶠ, ϵᵍʳⁿᵈ = c.maximum_canopy_emissivity * (1 - exp(-L)), c.ground_emissivity
    Tˡᵉᵃᶠ, Tᵍʳⁿᵈ = solution.Tˡᵉᵃᶠ, solution.Tᵍʳⁿᵈ
    ground_downwelling = (1 - ϵˡᵉᵃᶠ) * longwave + ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
    ground_upwelling = ϵᵍʳⁿᵈ * σ * Tᵍʳⁿᵈ^4 + (1 - ϵᵍʳⁿᵈ) * ground_downwelling
    Rˡᵉᵃᶠ = (1 - c.leaf_albedo) * (1 - τ) * shortwave + ϵˡᵉᵃᶠ * (longwave + ground_upwelling) -
            2ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4 - solution.𝒬ᵀˡᵉᵃᶠ - solution.𝒬ᵛˡᵉᵃᶠ
    Rᵍʳⁿᵈ = τ * (1 - c.ground_albedo) * shortwave + ϵᵍʳⁿᵈ * (ground_downwelling - σ * Tᵍʳⁿᵈ^4) -
         solution.𝒬ᵀᵍʳⁿᵈ - solution.𝒬ᵛᵍʳⁿᵈ - solution.𝒬ᵍʳⁿᵈ
    return Rˡᵉᵃᶠ, Rᵍʳⁿᵈ
end

@testset "Quasi-Newton skin solve closes the leaf and ground energy balances" begin
    for (FT, tolerance) in ((Float32, 1e-2), (Float64, 1e-4)),
        (leaf_area_index, canopy_water_storage) in ((3, 0), (3, 0.2), (0.5, 0), (6, 0))
        c = canopy_ground_surface(FT)
        args = canopy_arguments(FT; leaf_area_index, canopy_water_storage, canopy_water_capacity = 0.3)
        Rˡᵉᵃᶠ, Rᵍʳⁿᵈ = canopy_residuals(c, canopy_air_space_solve(c, args..., FT(0)), args[3], args[4])
        @test abs(Rˡᵉᵃᶠ) < tolerance
        @test abs(Rᵍʳⁿᵈ) < tolerance
    end
end

@testset "Absorbed PAR absorbs on the canopy's own structure" begin
    FT = Float64
    par = AbsorbedPhotosyntheticallyActiveRadiation(FT; leaf_par_albedo = 0.08, extinction = 0.5, clumping = 1)
    Ψᵣ = AirLandRadiationState(FT(5.67e-8), FT(0.2), FT(0.95), FT(-800), FT(-350))
    λˡᵉᵃᶠ = FT(3)
    incident = - par.par_fraction * Ψᵣ.ℐꜜˢʷ * par.photon_per_joule

    own = exp(-par.extinction * λˡᵉᵃᶠ * par.clumping)
    @test absorbed_par_value(par, Ψᵣ, λˡᵉᵃᶠ, nothing) ≈ (1 - par.leaf_par_albedo) * (1 - own) * incident / λˡᵉᵃᶠ
    @test absorbed_par_value(par, Ψᵣ, FT(1e-3), nothing) < incident
    @test absorbed_par_value(par, Ψᵣ, FT(0), nothing) == 0

    transmittance = FT(0.4)
    supplied = absorbed_par_value(par, Ψᵣ, λˡᵉᵃᶠ, transmittance)
    @test supplied ≈ (1 - par.leaf_par_albedo) * (1 - transmittance) * incident / λˡᵉᵃᶠ
    @test absorbed_par_value(AbsorbedPhotosyntheticallyActiveRadiation(FT; leaf_par_albedo = 0.08, extinction = 0.9), Ψᵣ, λˡᵉᵃᶠ, transmittance) ≈ supplied

    canopy_surface = canopy_ground_surface(FT; leaf_area_index = λˡᵉᵃᶠ, extinction = 0.55, clumping = 0.75)
    @test exp(-canopy_surface.extinction * λˡᵉᵃᶠ * canopy_surface.clumping) != own
    @test absorbed_par_value(FT(5e-4), Ψᵣ, λˡᵉᵃᶠ, FT(0.3)) == FT(5e-4)
end

@testset "Coupled per-cell canopy optics" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (2, 1, 1), latitude = (10, 11), longitude = (10, 12), z = (-1, 0),
                                     topology = (Bounded, Bounded, Bounded))
        scalar_canopy_surface = canopy_ground_surface(Float64)
        leaf_albedo = Field{Center, Center, Nothing}(grid)
        set!(leaf_albedo, (λ, φ) -> ifelse(λ < 11, scalar_canopy_surface.leaf_albedo, 0.6))

        function leaf_temperature(surface)
            model = coupled_land_model(arch; grid, atmosphere_land_interface_state = surface)
            return Array(interior(model.interfaces.atmosphere_land_interface.temperature.canopy))
        end
        Tˡᵉᵃᶠ = leaf_temperature(canopy_ground_surface(Float64; leaf_albedo))
        @test Tˡᵉᵃᶠ[2, 1, 1] < Tˡᵉᵃᶠ[1, 1, 1]
        @test Tˡᵉᵃᶠ[1, 1, 1] == leaf_temperature(scalar_canopy_surface)[1, 1, 1]
    end
end

@testset "Coupled per-cell ground conduction and ground-surface resistance" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (2, 1, 1), latitude = (10, 11), longitude = (10, 12), z = (-1, 0),
                                     topology = (Bounded, Bounded, Bounded))

        function per_cell(soil, water)
            field = Field{Center, Center, Nothing}(grid)
            set!(field, (λ, φ) -> ifelse(λ < 11, soil, water))
            return field
        end

        # Cell 1 keeps the scalar defaults; cell 2 is open water, with a skin held near the bulk
        # temperature and no resistance on the ground vapor path.
        scalar_surface = bare()
        conducting_surface = bare(ground_heat_flux = SoilConductiveFlux(per_cell(1.5, 50.0), 0.05))
        evaporating_surface = bare(ground_surface_resistance = GroundSurfaceResistance(Float64; transfer_coefficient = per_cell(0.00633, 1e6)))

        function ground_state(surface)
            model = coupled_land_model(arch; grid, atmosphere_land_interface_state = surface, land_temperature = 298.0)
            Ts = model.interfaces.atmosphere_land_interface.temperature
            return (Tᵍʳⁿᵈ = Array(interior(Ts.ground))[:, 1, 1], 𝒬ᵛᵍʳⁿᵈ = Array(interior(Ts.ground_latent_heat))[:, 1, 1])
        end

        reference = ground_state(scalar_surface)
        conducting = ground_state(conducting_surface)
        evaporating = ground_state(evaporating_surface)
        @test conducting.Tᵍʳⁿᵈ[1] == reference.Tᵍʳⁿᵈ[1]
        @test abs(conducting.Tᵍʳⁿᵈ[2] - 298) < abs(conducting.Tᵍʳⁿᵈ[1] - 298)
        @test evaporating.𝒬ᵛᵍʳⁿᵈ[1] == reference.𝒬ᵛᵍʳⁿᵈ[1]
        @test evaporating.𝒬ᵛᵍʳⁿᵈ[2] > evaporating.𝒬ᵛᵍʳⁿᵈ[1]
    end
end

@testset "Wet-canopy vapor conductance scales with air density" begin
    λˡᵉᵃᶠ, wˡᵇˡ = 3.0, 0.02
    canopy_surface = canopy_ground_surface(Float64; leaf_area_index = λˡᵉᵃᶠ, leaf_boundary_layer_velocity = wˡᵇˡ)
    state = (air_temperature = 305, air_specific_humidity = 0.006, flux_scales = (0.26, 1e-3, -1e-3, 0.1, 0.1),
             canopy_water_capacity = 0.1λˡᵉᵃᶠ)
    wet = canopy_solve(canopy_surface; state..., canopy_water_storage = 0.1λˡᵉᵃᶠ)
    dry = canopy_solve(canopy_surface; state...)

    @test wet.𝒬ᵛˡᵉᵃᶠ > dry.𝒬ᵛˡᵉᵃᶠ
    @test wet.Jʷᵉᵗ > 0
    @test dry.Jʷᵉᵗ == 0

    ℂ = AtmosphereThermodynamicsParameters(Float64)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂ, 305.0, 101325.0, 0.006)
    qˡᵉᵃᶠ = saturation_specific_humidity(ℂ, wet.Tˡᵉᵃᶠ, 101325.0, canopy_surface.canopy_humidity.phase)
    @test wet.Jʷᵉᵗ ≈ ρᵃᵗ * λˡᵉᵃᶠ * wˡᵇˡ * (qˡᵉᵃᶠ - wet.qᵃᶜᵃⁿ) rtol = 1e-6
end

@testset "Saturated soil keeps evaporating (dry-layer wet blend in CanopyInterface)" begin
    soil = power_law_soil_humidity(Float64)
    bare(wᵘᶜᵃⁿ) = canopy_ground_surface(Float64; leaf_area_index = 0.0, soil_humidity = soil,
                                      undercanopy_transfer_velocity = wᵘᶜᵃⁿ, ground_surface_resistance = nothing)
    state(𝒮) = (; ground_vapor_state(𝒮)..., vapor_diffusivity = 2.4e-5 * (0.4 * (1 - 𝒮))^(10/3) / 0.4^2)

    E = [canopy_solve(bare(wᵘᶜᵃⁿ); state(0.99)...).𝒬ᵛᵍʳⁿᵈ for wᵘᶜᵃⁿ in (0.05, 0.5, 5.0)]
    @test all(E .> 50)
    @test issorted(E)

    Ψⁱⁿ, Ψₐ, Ψˡᵃ, _, ℙₐ = canopy_arguments(Float64; state(0.1)...)
    Gᵉ, qᵉ, fᵈᵇʳ = dry_layer_terms(soil, 300.0, Ψⁱⁿ, Ψₐ, Ψˡᵃ, ℙₐ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℙₐ.thermodynamics_parameters, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    @test fᵈᵇʳ > 0.99
    @test fᵈᵇʳ * Gᵉ + (1 - fᵈᵇʳ) * (ρᵃᵗ * 0.5) ≈ Gᵉ rtol = 0.02
end

@testset "Ground-surface resistances (bulk cover, Sellers fit)" begin
    for FT in (Float32, Float64)
        r = ExponentialSoilResistance(FT)
        @test soil_surface_resistance(r, FT(1)) ≈ exp(FT(8.206) - FT(4.255))   # Sellers et al. (1992), Eq. (19)
        @test soil_surface_resistance(r, FT(2)) == soil_surface_resistance(r, FT(1))
        @test soil_surface_resistance(r, FT(0.4)) > soil_surface_resistance(r, FT(0.7))
        @test soil_surface_resistance(nothing, FT(0.5)) == 0

        r = GroundSurfaceResistance(FT)
        # Sakaguchi & Zeng (2009), Eq. (13), at unit litter area index
        @test ground_surface_resistance(r, FT(0.3)) ≈ (1 - exp(-FT(1))) / (FT(0.004) * FT(0.3)) rtol = 1e-3
        @test ground_surface_resistance(r, FT(0.1)) > ground_surface_resistance(r, FT(0.3))
        @test isfinite(ground_surface_resistance(r, FT(0)))
        @test ground_surface_resistance(FT(50), FT(0.3)) == 50
        @test ground_surface_resistance(nothing, FT(0.3)) == 0
    end

    unresisted = bare(ground_surface_resistance = nothing)
    exponential    = bare(ground_surface_resistance = nothing, wet_soil_resistance = ExponentialSoilResistance(Float64))
    covered    = bare()
    𝒬ᵛᵍʳⁿᵈ(surface, 𝒮) = canopy_solve(surface; ground_vapor_state(𝒮)...).𝒬ᵛᵍʳⁿᵈ

    @test 𝒬ᵛᵍʳⁿᵈ(covered, 0.9) < 𝒬ᵛᵍʳⁿᵈ(exponential, 0.9) < 𝒬ᵛᵍʳⁿᵈ(unresisted, 0.9)
    @test issorted([𝒬ᵛᵍʳⁿᵈ(covered, 𝒮) for 𝒮 in 0.50:-0.03:0.14], rev = true)
end

@testset "Undercanopy transfer velocity closures" begin
    for FT in (Float32, Float64)
        @test undercanopy_transfer_velocity(canopy_ground_surface(FT).undercanopy_transfer_velocity, FT(3), FT(5), FT(0.3)) === FT(0.013)

        w(λˡᵉᵃᶠ, ΔU, u★; kw...) = undercanopy_transfer_velocity(AreaIndexUndercanopyVelocity(FT; kw...), FT(λˡᵉᵃᶠ), FT(ΔU), FT(u★))
        @test w(0.5, 3, 0.26) > w(5, 3, 0.26)
        @test w(3, 5, 0.26) > w(3, 1, 0.26)
        @test isfinite(w(0, 3, 0.26))
        @test w(0, 3, 0.26) <= FT(0.26)^2 / 3
        @test w(1, 3, 0.26; stem_area_index = 1) < w(1, 3, 0.26)
        @test w(3, 0, 0.26) == 0

        wz(λˡᵉᵃᶠ, u★; kw...) = undercanopy_transfer_velocity(FrictionVelocityUndercanopyVelocity(FT; kw...), FT(λˡᵉᵃᶠ), FT(3), FT(u★))
        @test wz(20, 0.3) ≈ FT(0.004) * FT(0.3) rtol = 1e-3
        @test wz(0, 0.3) > wz(5, 0.3)
        # Bare ground follows the roughness-Reynolds law (ϰ/0.13)(ℓᵍʳⁿᵈu★/ν)^(−0.45)·u★
        Cₛᵇ = (FT(0.4) / FT(0.13)) * (FT(0.01) * FT(0.3) / FT(1.5e-5))^(-FT(0.45))
        @test wz(0, 0.3) ≈ Cₛᵇ * FT(0.3) rtol = 1e-5
        @test wz(3, 0) == 0
        @test isfinite(wz(0, 0))
        @test wz(0, 0.3; ground_roughness_length = 0.05) < wz(0, 0.3)
        @test wz(1, 0.3; stem_area_index = 1) < wz(1, 0.3)

        canopy_surface = canopy_ground_surface(FT; leaf_area_index = 3.0, undercanopy_transfer_velocity = AreaIndexUndercanopyVelocity(FT))
        function soil_share(λˡᵉᵃᶠ)
            solution = canopy_solve(canopy_surface, FT; flux_scales = (0.26, 1e-3, -1e-3, 0.1, 0.1), leaf_area_index = λˡᵉᵃᶠ)
            return solution.𝒬ᵛᵍʳⁿᵈ / (solution.𝒬ᵛᵍʳⁿᵈ + solution.𝒬ᵛˡᵉᵃᶠ)
        end
        @test soil_share(0.5) > soil_share(5)
    end
end

@testset "Saturated soil under calm air keeps the canopy solve finite" begin
    canopy_surface = canopy_ground_surface(Float64; leaf_area_index = 3.0, soil_humidity = power_law_soil_humidity(Float64),
                                           undercanopy_transfer_velocity = FrictionVelocityUndercanopyVelocity(Float64))
    solution = canopy_solve(canopy_surface; wind = 0, albedo = 0.2, emissivity = 0.95, flux_scales = (0, 0, 0, 0.1, 0.1),
                            temperature = 298, saturation = 1, vapor_diffusivity = 0)
    @test all(isfinite, (solution.Tˡᵉᵃᶠ, solution.Tᵍʳⁿᵈ, solution.Tᵃᶜᵃⁿ, solution.𝒬ᵛᵍʳⁿᵈ, solution.𝒬ᵀᵍʳⁿᵈ))
    @test solution.𝒬ᵛᵍʳⁿᵈ == 0
end

@testset "Land radiation diagnostics follow the radiating surface" begin
    for arch in test_architectures
        α, ϵ, ℐꜜˢʷ = 0.2, 0.95, 600.0
        skin  = EnergyBalanceTemperature(SoilConductiveFlux(1.5, 0.05))
        canopy_surface = canopy_ground_surface(Float64)
        skin_state = (temperature=skin, specific_humidity=BulkHumidity())
        for state in (skin_state, canopy_surface)
            model = coupled_land_model(arch; atmosphere_land_interface_state = state)
            time_step!(model, 60.0)
            σ  = model.radiation.stefan_boltzmann_constant
            radiation = model.radiation.interface_fluxes.land
            Ts = model.interfaces.atmosphere_land_interface.temperature
            ϵσT⁴, αₛ = state === skin_state ? (ϵ * σ * scalar(Ts)^4, α) : (σ * scalar(Ts.effective)^4, scalar(Ts.effective_albedo))
            @test scalar(radiation.upwelling_longwave) ≈ ϵσT⁴ rtol = 1e-6
            @test scalar(radiation.downwelling_shortwave) ≈ -(1 - αₛ) * ℐꜜˢʷ rtol = 1e-6
        end
    end
end

@testset "Buffered vegetation series are brought into memory before sampling" begin
    grid  = LatitudeLongitudeGrid(CPU(); size = 1, latitude = 10, longitude = 10, z = (-1, 0), topology = (Flat, Flat, Bounded))
    times = 0.0:1.0:4.0
    path  = joinpath(mktempdir(), "lai.jld2")
    fts   = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend = OnDisk(), path, name = "lai")
    field = Field{Center, Center, Nothing}(grid)
    for n in eachindex(times)
        set!(field, 1 + times[n])
        set!(fts, field, n)
    end

    # Two snapshots resident; sampling between the third and fourth has to load them first.
    lai = FieldTimeSeries(path, "lai"; backend = InMemory(2), architecture = CPU())
    vegetation, interpolator = kernel_surface_field(lai, CPU(), 2.5)
    @test stateindex(vegetation, 1, 1, 1, interpolator) ≈ 3.5
end
