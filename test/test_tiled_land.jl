include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior, CenterField
using Oceananigans.TimeSteppers: update_state!, time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, DryLayerHumidity, StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity, ConstantTortuosity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, TiledLandInterface, bare_canopy_air_space, leaf_area_index_cover_fraction,
    SimilarityTheoryFluxes, atmosphere_land_stability_functions
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties

build_tiled_canopy_air_space(FT) = CanopyAirSpace(FT;
    soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                    dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
        vapor_exchange  = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                      molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.05, porosity = 0.4),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 3.0, moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05))

land_roughness(FT, z₀m, z₀s) = SimilarityTheoryFluxes(FT;
    stability_functions          = atmosphere_land_stability_functions(FT),
    momentum_roughness_length    = z₀m,
    temperature_roughness_length = z₀s,
    water_vapor_roughness_length = z₀s)

# Single-column coupled model with a `TiledLandInterface` (veg + bare over a shared slab).
function tiled_land_model(arch, cas; fraction, shortwave = 600.0,
                          vegetated_fluxes = nothing, bare_fluxes = nothing, wind = 3.0)
    FT = Float64
    grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                 z = (-1, 0), topology = (Flat, Flat, Bounded))
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
    fill!(parent(atmosphere.temperature), 300.0)
    fill!(parent(atmosphere.specific_humidity), 0.008)
    fill!(parent(atmosphere.velocities.u), wind)
    fill!(parent(atmosphere.pressure), 101325.0)
    land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
    set!(land; T = 298.0)
    fill!(parent(land.water_storage), 45.0)   # 𝒮 = 0.3
    radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                    land_surface = SurfaceRadiationProperties(0.2, 0.95))
    fill!(parent(radiation.downwelling_shortwave), shortwave)
    fill!(parent(radiation.downwelling_longwave), 350.0)
    update_state!(radiation)

    kw = (; vegetated = cas, fraction)
    isnothing(vegetated_fluxes) || (kw = (; kw..., vegetated_fluxes))
    isnothing(bare_fluxes)      || (kw = (; kw..., bare_fluxes))
    tiled = TiledLandInterface(grid, atmosphere, land; kw...)

    model = AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = tiled)
    update_state!(model.land)
    update_state!(model)
    return model
end

scalar(field) = Array(interior(field))[1, 1, 1]

@testset "TiledLandInterface" begin
    for arch in test_architectures
        cas = build_tiled_canopy_air_space(Float64)

        # --- Blended output is finite, physical, and drives the slab by conduction. ---
        model = tiled_land_model(arch, cas; fraction = 0.6)
        ti = model.interfaces.atmosphere_land_interface
        @test ti.temperature isa NamedTuple

        𝒬ᵀ = scalar(ti.fluxes.sensible_heat)
        𝒬ᵛ = scalar(ti.fluxes.latent_heat)
        Gᶜ = scalar(ti.temperature.ground_heat_flux)
        Tᵉ = scalar(ti.temperature.effective)
        @test all(isfinite, (𝒬ᵀ, 𝒬ᵛ, Gᶜ, Tᵉ))
        @test 285 < scalar(ti.temperature.interface) < 320

        # Slab driven by the blended skin→bulk conduction, Es = −Gcond (radiation
        # internalized per tile ⇒ apply_air_land_radiative_fluxes! adds nothing).
        Es = scalar(model.land.fluxes.surface_energy_flux)
        @test Es ≈ -Gᶜ atol = 1e-10

        # --- Linearity: blended == f·veg + (1−f)·bare from the sub-tile buffers. ---
        f = 0.6
        veg = ti.vegetated
        bare = ti.bare
        for name in (:sensible_heat, :latent_heat, :water_vapor, :x_momentum, :y_momentum)
            b  = scalar(getproperty(ti.fluxes, name))
            fv = scalar(getproperty(veg.fluxes, name))
            bv = scalar(getproperty(bare.fluxes, name))
            @test b ≈ f * fv + (1 - f) * bv rtol = 1e-12
        end
        for name in (:ground_heat_flux, :canopy_latent_heat, :soil_latent_heat,
                     :canopy_sensible_heat, :soil_sensible_heat, :canopy_evaporation, :interface)
            b  = scalar(getproperty(ti.temperature, name))
            fv = scalar(getproperty(veg.temperature, name))
            bv = scalar(getproperty(bare.temperature, name))
            @test b ≈ f * fv + (1 - f) * bv rtol = 1e-12
        end
        # Effective (LST) temperature blends in radiance (T⁴) space.
        Tv = scalar(veg.temperature.effective)
        Tb = scalar(bare.temperature.effective)
        @test scalar(ti.temperature.effective) ≈ (f * Tv^4 + (1 - f) * Tb^4)^(1/4) rtol = 1e-12

        # The bare tile is canopy-free: no transpiration, no leaf sensible.
        @test scalar(bare.temperature.canopy_latent_heat) == 0
        @test scalar(bare.temperature.canopy_sensible_heat) == 0

        # Sunlit veg tile: shaded soil skin cooler than the leaf; more latent than the bare tile.
        @test scalar(veg.temperature.soil_skin) < scalar(veg.temperature.canopy)
        @test scalar(veg.fluxes.latent_heat) > scalar(bare.fluxes.latent_heat)

        # --- Endpoint reduction: f=1 → veg tile, f=0 → bare tile (bit-for-bit). ---
        m1 = tiled_land_model(arch, cas; fraction = 1.0)
        t1 = m1.interfaces.atmosphere_land_interface
        m0 = tiled_land_model(arch, cas; fraction = 0.0)
        t0 = m0.interfaces.atmosphere_land_interface
        for name in (:sensible_heat, :latent_heat, :water_vapor, :x_momentum, :y_momentum)
            @test scalar(getproperty(t1.fluxes, name)) == scalar(getproperty(t1.vegetated.fluxes, name))
            @test scalar(getproperty(t0.fluxes, name)) == scalar(getproperty(t0.bare.fluxes, name))
        end
        @test scalar(t1.temperature.ground_heat_flux) == scalar(t1.vegetated.temperature.ground_heat_flux)
        @test scalar(t0.temperature.ground_heat_flux) == scalar(t0.bare.temperature.ground_heat_flux)

        # --- f=1 tiled ≡ a standalone CanopyAirSpace model (bit-for-bit). ---
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        atmos = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
        fill!(parent(atmos.temperature), 300.0); fill!(parent(atmos.specific_humidity), 0.008)
        fill!(parent(atmos.velocities.u), 3.0); fill!(parent(atmos.pressure), 101325.0)
        lands = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
        set!(lands; T = 298.0); fill!(parent(lands.water_storage), 45.0)
        rads = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                   land_surface = SurfaceRadiationProperties(0.2, 0.95))
        fill!(parent(rads.downwelling_shortwave), 600.0); fill!(parent(rads.downwelling_longwave), 350.0)
        update_state!(rads)
        ms = AtmosphereLandModel(atmos, lands; radiation = rads,
                atmosphere_land_interface_temperature = cas,
                atmosphere_land_interface_specific_humidity = cas)
        update_state!(ms.land); update_state!(ms)
        alis = ms.interfaces.atmosphere_land_interface
        @test scalar(t1.fluxes.sensible_heat) == scalar(alis.fluxes.sensible_heat)
        @test scalar(t1.fluxes.latent_heat) == scalar(alis.fluxes.latent_heat)
        @test scalar(t1.temperature.ground_heat_flux) == scalar(alis.temperature.ground_heat_flux)

        # --- Roughness contrast: a rough veg tile has a larger friction velocity than a smooth bare tile. ---
        mr = tiled_land_model(arch, cas; fraction = 0.5, wind = 6.0,
                              vegetated_fluxes = land_roughness(FT, 1.0, 0.1),
                              bare_fluxes      = land_roughness(FT, 1e-3, 1e-4))
        tr = mr.interfaces.atmosphere_land_interface
        @test scalar(tr.vegetated.fluxes.friction_velocity) > scalar(tr.bare.fluxes.friction_velocity)

        # --- Scalar and Field fractions agree for the same value. ---
        ff = CenterField(grid); set!(ff, 0.6)
        mff = tiled_land_model(arch, cas; fraction = ff)
        tff = mff.interfaces.atmosphere_land_interface
        @test scalar(tff.fluxes.sensible_heat) ≈ scalar(ti.fluxes.sensible_heat) rtol = 1e-12
        @test scalar(tff.fluxes.latent_heat)   ≈ scalar(ti.fluxes.latent_heat)   rtol = 1e-12

        # --- Multi-step: the shared soil column depletes under evapotranspiration, no NaNs. ---
        mstep = tiled_land_model(arch, cas; fraction = 0.6)
        M0 = scalar(mstep.land.water_storage)
        for _ in 1:20
            time_step!(mstep, 600.0)
        end
        M1 = scalar(mstep.land.water_storage)
        @test isfinite(M1) && M1 < M0
        tstep = mstep.interfaces.atmosphere_land_interface
        @test all(isfinite, (scalar(tstep.fluxes.sensible_heat),
                             scalar(tstep.fluxes.latent_heat),
                             scalar(mstep.land.temperature)))
    end

    # --- Float32 / Float64 end-to-end (blended buffers keep the grid float type). ---
    for FT in (Float32, Float64)
        grid = LatitudeLongitudeGrid(CPU(), FT; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
        fill!(parent(atmosphere.temperature), 300); fill!(parent(atmosphere.specific_humidity), 0.008)
        fill!(parent(atmosphere.velocities.u), 3); fill!(parent(atmosphere.pressure), 101325)
        land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150), energy = SlabEnergy(FT))
        set!(land; T = 298); fill!(parent(land.water_storage), 45)
        radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                        land_surface = SurfaceRadiationProperties(0.2, 0.95))
        fill!(parent(radiation.downwelling_shortwave), 600); fill!(parent(radiation.downwelling_longwave), 350)
        update_state!(radiation)
        tiled = TiledLandInterface(grid, atmosphere, land;
                                   vegetated = build_tiled_canopy_air_space(FT), fraction = FT(0.5))
        model = AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = tiled)
        update_state!(model.land); update_state!(model)
        𝒬ᵀ = scalar(model.interfaces.atmosphere_land_interface.fluxes.sensible_heat)
        @test 𝒬ᵀ isa FT
        @test isfinite(𝒬ᵀ)
    end

    # The Beer–Lambert cover helper is monotone in LAI and bounded in [0, 1).
    @test leaf_area_index_cover_fraction(0.0) == 0
    @test 0 < leaf_area_index_cover_fraction(2.0) < leaf_area_index_cover_fraction(6.0) < 1
end
