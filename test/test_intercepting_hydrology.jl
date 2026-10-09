include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, CanopyInterception,
    CriticalSaturation, InteractiveAbsorbedPAR, SoilConductiveFlux
using NumericalEarth.Lands: SlabLand, SlabEnergy, VariablySaturatedHydrology, InterceptingHydrology,
    VanGenuchtenRetention, VanGenuchtenConductivity
soil_hydrology(FT) = VariablySaturatedHydrology(FT;
    slab_depth = 1.0, porosity = 0.4, storage_height = 0.1,
    retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 2.0, pore_size_uniformity = 1.5),
    hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5))

# Coupled single-column model. `interception = true` wraps the soil hydrology in an
# `InterceptingHydrology` and hands the CAS a matching `CanopyInterception`.
function interception_model(arch, FT; leaf_area_index = 3.0, capacity = 0.1, rain = 1e-3,
                            water_storage = 300.0, interception = true)
    hydrology = interception ?
        InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index,
                              capacity_per_leaf_area = capacity) :
        soil_hydrology(FT)
    cas = CanopyAirSpace(FT;
        soil = bare_soil_humidity(FT),
        canopy = CanopyConductanceHumidity(FT; leaf_area_index,
                                           moisture_stress = CriticalSaturation(0.5),
                                           absorbed_par = InteractiveAbsorbedPAR(FT)),
        soil_skin_flux = SoilConductiveFlux(1.5, 0.05),
        interception = interception ? CanopyInterception() : nothing)
    return coupled_land_model(arch, FT; Tair = 298, qair = 0.010, wind = 3, rain, hydrology,
                              Tland = 298, water = water_storage, shortwave = 400,
                              atmosphere_land_interface_temperature = cas)
end

@testset "InterceptingHydrology declarations & container" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        h = InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index = 3.0)
        land = SlabLand(grid; energy = SlabEnergy(FT), hydrology = h)

        @test :canopy_water_storage in keys(land.prognostic)
        @test :canopy_evaporation in keys(land.fluxes)
        @test :liquid_precipitation_flux in keys(land.fluxes)     # from the wrapped soil
        @test :vapor_flux in keys(land.fluxes)
        @test :throughfall in keys(land.diagnostics)
        @test :canopy_water_storage_tendency in keys(land.diagnostics)
        @test :water_storage_tendency in keys(land.diagnostics)   # from the wrapped soil
    end
end

@testset "Dry canopy (Wᶜ = 0) reduces to the bare CanopyAirSpace bit-for-bit" begin
    for arch in test_architectures
        FT = Float64
        with    = interception_model(arch, FT; interception = true)   # store starts at 0
        without = interception_model(arch, FT; interception = false)

        Twith    = with.interfaces.atmosphere_land_interface.temperature
        Twithout = without.interfaces.atmosphere_land_interface.temperature
        Fwith    = with.interfaces.atmosphere_land_interface.fluxes
        Fwithout = without.interfaces.atmosphere_land_interface.fluxes

        @test scalar(with.land.prognostic.canopy_water_storage) == 0
        for name in (:interface, :canopy, :soil_skin, :effective)
            @test scalar(getproperty(Twith, name)) == scalar(getproperty(Twithout, name))
        end
        for name in (:sensible_heat, :latent_heat, :water_vapor)
            @test scalar(getproperty(Fwith, name)) == scalar(getproperty(Fwithout, name))
        end
        # No wet canopy ⇒ no wet-canopy evaporation, and the land loses the full Jᵛ (the
        # source partition closes to the fixed-point tolerance).
        @test scalar(Twith.canopy_evaporation) == 0
        @test scalar(with.land.fluxes.vapor_flux) ≈ scalar(Fwith.water_vapor) rtol = 1e-6
    end
end

@testset "Wet canopy: fill, bounds, and evaporation" begin
    for arch in test_architectures
        FT = Float64
        LAI = 3.0; c = 0.1
        Wᶜᵐᵃˣ = c * LAI
        model = interception_model(arch, FT; leaf_area_index = LAI, capacity = c, rain = 1e-3)
        land = model.land
        Δt = 600.0

        for _ in 1:40
            time_step!(model, Δt)
        end

        Wᶜ = scalar(land.prognostic.canopy_water_storage)
        @test 0 ≤ Wᶜ ≤ Wᶜᵐᵃˣ + eps(Wᶜᵐᵃˣ)
        @test Wᶜ ≈ Wᶜᵐᵃˣ rtol=1e-6                       # heavy rain saturates the canopy
        # Wet canopy evaporates (potential rate), consistent interface ↔ land.
        @test scalar(model.interfaces.atmosphere_land_interface.temperature.canopy_evaporation) > 0
        @test scalar(land.fluxes.canopy_evaporation) ==
              scalar(model.interfaces.atmosphere_land_interface.temperature.canopy_evaporation)
        # The land loses the atmospheric vapor flux less the wet-canopy share (to the
        # fixed-point tolerance of the source partition).
        Jᵛ = scalar(model.interfaces.atmosphere_land_interface.fluxes.water_vapor)
        Ew = scalar(land.fluxes.canopy_evaporation)
        @test scalar(land.fluxes.vapor_flux) ≈ Jᵛ - Ew rtol = 1e-5
    end
end

@testset "Canopy water mass conservation" begin
    for arch in test_architectures
        FT = Float64
        LAI = 3.0; c = 0.1
        model = interception_model(arch, FT; leaf_area_index = LAI, capacity = c, rain = 1e-3)
        land = model.land
        Δt = 600.0

        # Off the clamp boundaries: a partially-full store draining/filling.
        set!(land; canopy_water_storage = 0.5 * c * LAI)
        update_state!(model)   # refresh Eʷᵉᵗ with the reset store

        rain          = scalar(land.fluxes.liquid_precipitation_flux)   # raw rain the step will read
        Ewet_demanded = scalar(land.fluxes.canopy_evaporation)          # interface Eʷᵉᵗ the step will consume

        time_step!(model, Δt)

        dWcdt    = scalar(land.diagnostics.canopy_water_storage_tendency)
        throughf = scalar(land.diagnostics.throughfall)
        # Exact canopy water budget: rain = dWᶜ/dt + Eʷᵉᵗ + throughfall.
        @test rain ≈ dWcdt + Ewet_demanded + throughf atol=1e-14
        # The soil reads the throughfall; the rain accumulator is left as the coupler wrote it.
        @test scalar(land.fluxes.liquid_precipitation_flux) == rain
        # Interception caught a positive fraction ⇒ less than the raw rain reaches the soil.
        @test 0 ≤ throughf < rain
    end
end

@testset "Store stays non-negative under aggressive drydown" begin
    for arch in test_architectures
        FT = Float64
        LAI = 3.0; c = 0.1
        # No rain, large step: the interface caps Eʷᵉᵗ at what the store holds, Wᶜ/Δt (to the
        # tolerance of the canopy solve), so the atmosphere never receives water the canopy
        # did not lose.
        model = interception_model(arch, FT; leaf_area_index = LAI, capacity = c, rain = 0.0)
        land = model.land
        set!(land; canopy_water_storage = 0.8 * c * LAI)
        Δt = 3600.0
        for _ in 1:200
            Wᶜ⁻ = scalar(land.prognostic.canopy_water_storage)
            time_step!(model, Δt)
            @test scalar(land.fluxes.canopy_evaporation) ≤ Wᶜ⁻ / Δt * (1 + 1e-5)
            @test scalar(land.prognostic.canopy_water_storage) ≥ 0
        end
        @test scalar(land.prognostic.canopy_water_storage) < 1e-3   # drains toward empty
    end
end

@testset "Field- and FieldTimeSeries-valued LAI" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))

        # Static Field LAI (GPU-safe): capacity tracks the field value.
        laiF = CenterField(grid); set!(laiF, 2.0)
        hF = InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index = laiF, capacity_per_leaf_area = 0.1)
        landF = SlabLand(grid; energy = SlabEnergy(FT), hydrology = hF)
        set!(landF; T = 295, M = 300, canopy_water_storage = 1.0)   # above capacity 0.1*2 = 0.2
        fill!(parent(landF.fluxes.liquid_precipitation_flux), 0)
        fill!(parent(landF.fluxes.canopy_evaporation), 0)
        time_step!(landF, 600.0)
        @test scalar(landF.prognostic.canopy_water_storage) ≈ 0.2 rtol=1e-6   # capped at c·LAI

        # FieldTimeSeries LAI (CPU only): capacity follows the interpolated value in time.
        if arch isa CPU
            times = [0.0, 100.0]
            lai_fts = FieldTimeSeries{Center, Center, Nothing}(grid, times)
            set!(lai_fts[1], 1.0); set!(lai_fts[2], 3.0)
            hT = InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index = lai_fts, capacity_per_leaf_area = 0.1)
            landT = SlabLand(grid; energy = SlabEnergy(FT), hydrology = hT)   # clock starts at t = 0
            set!(landT; T = 295, M = 300, canopy_water_storage = 1.0)
            fill!(parent(landT.fluxes.liquid_precipitation_flux), 0)
            fill!(parent(landT.fluxes.canopy_evaporation), 0)
            time_step!(landT, 50.0)   # clock → t = 50 ⇒ LAI = 2.0 ⇒ capacity 0.2
            @test scalar(landT.prognostic.canopy_water_storage) ≈ 0.2 rtol=1e-6
        end
    end
end

@testset "Checkpoint round-trip of the canopy water store" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        h = InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index = 3.0)
        land = SlabLand(grid; energy = SlabEnergy(FT), hydrology = h)
        set!(land; T = 290, M = 120, canopy_water_storage = 0.17)

        state = Oceananigans.prognostic_state(land)
        @test :canopy_water_storage in keys(state.prognostic)

        land2 = SlabLand(grid; energy = SlabEnergy(FT),
                         hydrology = InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index = 3.0))
        Oceananigans.restore_prognostic_state!(land2, state)
        @test scalar(land2.prognostic.canopy_water_storage) == 0.17
        @test scalar(land2.temperature) == 290
    end
end

@testset "Checkpoint restore is keyed by name, not position" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        mkland(store) = (l = SlabLand(grid; energy = SlabEnergy(FT),
                             hydrology = InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index = 3.0));
                         set!(l; T = 290, M = 120, canopy_water_storage = store); l)

        # A checkpoint whose prognostic tuple carries an extra *leading* key (schema drift):
        # a by-position restore would load the wrong (0.99) value; by-name loads 0.17.
        sa = Oceananigans.prognostic_state(mkland(0.17))
        sb = Oceananigans.prognostic_state(mkland(0.99))
        drifted = merge(sa, (; prognostic = merge((bogus = sb.prognostic.canopy_water_storage,), sa.prognostic)))
        land = mkland(0.0)
        Oceananigans.restore_prognostic_state!(land, drifted)
        @test scalar(land.prognostic.canopy_water_storage) == 0.17

        # A legacy checkpoint with no `:prognostic` field restores gracefully, leaving the store
        # at its current value (no crash, no length-mismatch mispairing).
        legacy = (; clock = sa.clock, temperature = sa.temperature, water_storage = sa.water_storage)
        land2 = mkland(0.42)
        Oceananigans.restore_prognostic_state!(land2, legacy)
        @test scalar(land2.prognostic.canopy_water_storage) == 0.42
    end
end

@testset "Float32 type stability" begin
    for arch in test_architectures
        FT = Float32
        model = interception_model(arch, FT; leaf_area_index = 3.0f0, capacity = 0.1f0)
        time_step!(model, 600)
        Wᶜ = scalar(model.land.prognostic.canopy_water_storage)
        Ew = scalar(model.land.fluxes.canopy_evaporation)
        @test Wᶜ isa Float32
        @test isfinite(Wᶜ) && Wᶜ ≥ 0
        @test isfinite(Ew)
    end
end
