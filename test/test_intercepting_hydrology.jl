include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!
using Oceananigans.TimeSteppers: update_state!, time_step!
using Oceananigans.OutputReaders: OnDisk, InMemory
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyGroundSurface, CanopyConductanceHumidity,
    CriticalSaturation, InteractiveAbsorbedPAR, SoilConductiveFlux, ComponentExchanger
using NumericalEarth.Lands: SlabLand, SlabEnergy, VariablySaturatedHydrology, InterceptingHydrology,
    VanGenuchtenRetention, VanGenuchtenConductivity, canopy_store_update
soil_hydrology(FT) = VariablySaturatedHydrology(FT;
    slab_depth = 1.0, porosity = 0.4, storage_height = 0.1,
    retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 2.0, pore_size_uniformity = 1.5),
    hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5))

# Coupled single-column model. `interception = true` wraps the soil hydrology in an
# `InterceptingHydrology`.
function interception_model(arch, FT; leaf_area_index = 3.0, capacity = 0.1, rain = 1e-3,
                            water_storage = 300.0, interception = true, cover_fraction = nothing)
    hydrology = interception ?
        InterceptingHydrology(FT; soil = soil_hydrology(FT),
                              capacity_per_leaf_area = capacity) :
        soil_hydrology(FT)
    canopy_surface = CanopyGroundSurface(FT;
        soil_humidity = bare_soil_humidity(FT),
        canopy_humidity = CanopyConductanceHumidity(FT; leaf_area_index,
                                           moisture_stress = CriticalSaturation(0.5),
                                           absorbed_par = InteractiveAbsorbedPAR(FT)),
        ground_heat_flux = SoilConductiveFlux(1.5, 0.05))
    surface = LandSurface(canopy_surface)
    isnothing(cover_fraction) || (surface = TiledLandSurface(surface; cover_fraction))
    return coupled_land_model(arch, FT; air_temperature = 298, air_specific_humidity = 0.010, wind = 3, rain,
                              hydrology, surface, land_temperature = 298, water = water_storage, shortwave = 400)
end

@testset "Tiled interception shares vegetation and conserves water" begin
    for arch in test_architectures
        cover_fraction, rain, Δt = 0.4, 1e-3, 60.0
        model = interception_model(arch, Float64; cover_fraction, rain)
        land = model.land
        before = scalar(land.water_storage) + cover_fraction * scalar(land.prognostic.canopy_water_storage)
        time_step!(model, Δt)
        after = scalar(land.water_storage) + cover_fraction * scalar(land.prognostic.canopy_water_storage)
        vapor = scalar(model.interfaces.atmosphere_land_interface.fluxes.water_vapor)
        @test after - before ≈ Δt * (rain - vapor) atol = 1e-10
        @test land.hydrology.leaf_area_index == 3.0
        @test land.hydrology.cover_fraction == cover_fraction
    end
end

@testset "Changing vegetation cover conserves canopy and soil water" begin
    for rain in (0.0, 1e-3)
        grid = LatitudeLongitudeGrid(CPU(); size = 1, latitude = 10, longitude = 10, z = (-1, 0), topology = (Flat, Flat, Bounded))
        Δt = 60.0
        covers = (0.2, 0.8, 0.2, 0.0, 0.8)
        cover_fraction = FieldTimeSeries{Center, Center, Nothing}(grid, 0.0:Δt:4Δt)
        for n in eachindex(covers)
            set!(cover_fraction[n], covers[n])
        end
        model = interception_model(CPU(), Float64; cover_fraction, rain)
        land = model.land
        set!(land; canopy_water_storage = 0.15)
        update_state!(model)

        for n in 1:4
            before = scalar(land.water_storage) + covers[n] * scalar(land.prognostic.canopy_water_storage)
            time_step!(model, Δt)
            storage = scalar(land.prognostic.canopy_water_storage)
            after = scalar(land.water_storage) + covers[n+1] * storage
            vapor = scalar(model.interfaces.atmosphere_land_interface.fluxes.water_vapor)
            @test after - before ≈ Δt * (rain - vapor) atol = 1e-10
            @test 0 ≤ storage ≤ 0.1 * 3
        end
    end
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

@testset "Buffered canopy properties conserve water across time windows" begin
    arch  = CPU()   # a FieldTimeSeries LAI runs on CPU only
    FT    = Float64
    grid  = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10, z = (-1, 0), topology = (Flat, Flat, Bounded))
    times = 0.0:1.0:4.0
    path  = joinpath(mktempdir(), "lai.jld2")
    fts   = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend = OnDisk(), path, name = "lai")
    fraction = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend = OnDisk(), path, name = "fraction")
    covers = (0.2, 0.8, 0.2, 0.0, 0.8)
    field = Field{Center, Center, Nothing}(grid)
    for n in eachindex(times)
        set!(field, 1 + times[n])
        set!(fts, field, n)
        set!(field, covers[n])
        set!(fraction, field, n)
    end

    lai  = FieldTimeSeries(path, "lai"; backend = InMemory(2), architecture = arch)
    cover_fraction = FieldTimeSeries(path, "fraction"; backend = InMemory(2), architecture = arch)
    h    = InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index = lai, cover_fraction,
                                capacity_per_leaf_area = 0.1)
    land = SlabLand(grid; energy = SlabEnergy(FT), hydrology = h)
    set!(land; M = 300, canopy_water_storage = 0.05)
    before = scalar(land.water_storage) + covers[1] * scalar(land.prognostic.canopy_water_storage)
    for n in 1:3
        time_step!(land, 1.0)
        after = scalar(land.water_storage) + covers[n+1] * scalar(land.prognostic.canopy_water_storage)
        @test after ≈ before atol = 1e-10
    end

    # LAI(t = 3) = 4 sits past the two snapshots resident when the series was opened.
    capacity = ComponentExchanger(land, grid).state.canopy_water_capacity
    @test scalar(compute!(Field(capacity))) ≈ 0.1 * 4
end

@testset "Canopy store budget closes when the store empties mid-step" begin
    Mᶜ, rain, Mᶜ⁺, fⁱⁿᵗ, f, Δt = 0.05, 1e-4, 0.3, 0.6, 0.5, 600.0
    for Jʷᵉᵗ in (0.0, 1e-4, 5e-4)   # the last asks for more than the store and the rain hold
        Mᶜⁿ⁺¹, drip, throughfall = canopy_store_update(Mᶜ, rain, Jʷᵉᵗ, Mᶜ⁺, fⁱⁿᵗ, f, f, Δt)
        @test 0 ≤ Mᶜⁿ⁺¹ ≤ Mᶜ⁺
        @test drip ≥ 0
        @test rain ≈ f * ((Mᶜⁿ⁺¹ - Mᶜ) / Δt + Jʷᵉᵗ) + throughfall atol = 1e-15
    end
end

@testset "Dry canopy (Mᶜ = 0) reduces to the bare CanopyGroundSurface bit-for-bit" begin
    for arch in test_architectures
        FT = Float64
        with    = interception_model(arch, FT; interception = true)   # store starts at 0
        without = interception_model(arch, FT; interception = false)

        Twith    = with.interfaces.atmosphere_land_interface.temperature
        Twithout = without.interfaces.atmosphere_land_interface.temperature
        Fwith    = with.interfaces.atmosphere_land_interface.fluxes
        Fwithout = without.interfaces.atmosphere_land_interface.fluxes

        @test scalar(with.land.prognostic.canopy_water_storage) == 0
        for name in (:interface, :canopy, :ground, :effective)
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
        Mᶜ⁺ = c * LAI
        model = interception_model(arch, FT; leaf_area_index = LAI, capacity = c, rain = 1e-3)
        land = model.land
        Δt = 600.0

        for _ in 1:40
            time_step!(model, Δt)
        end

        Mᶜ = scalar(land.prognostic.canopy_water_storage)
        @test 0 ≤ Mᶜ ≤ Mᶜ⁺ + eps(Mᶜ⁺)
        @test Mᶜ ≈ Mᶜ⁺ rtol=1e-6                       # heavy rain saturates the canopy
        # Wet canopy evaporates (potential rate), consistent interface ↔ land.
        @test scalar(model.interfaces.atmosphere_land_interface.temperature.canopy_evaporation) > 0
        @test scalar(land.fluxes.canopy_evaporation) ==
              scalar(model.interfaces.atmosphere_land_interface.temperature.canopy_evaporation)
        # The land loses the atmospheric vapor flux less the wet-canopy share (to the
        # fixed-point tolerance of the source partition).
        Jᵛ = scalar(model.interfaces.atmosphere_land_interface.fluxes.water_vapor)
        Jʷᵉᵗ = scalar(land.fluxes.canopy_evaporation)
        @test scalar(land.fluxes.vapor_flux) ≈ Jᵛ - Jʷᵉᵗ rtol = 1e-5
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
        update_state!(model)   # refresh Jʷᵉᵗ with the reset store

        rain = scalar(land.fluxes.liquid_precipitation_flux)   # raw rain the step will read
        Jʷᵉᵗ = scalar(land.fluxes.canopy_evaporation)          # interface Jʷᵉᵗ the step will consume

        time_step!(model, Δt)

        dMᶜdt    = scalar(land.diagnostics.canopy_water_storage_tendency)
        throughf = scalar(land.diagnostics.throughfall)
        # Exact canopy water budget: rain = dMᶜ/dt + Jʷᵉᵗ + throughfall.
        @test rain ≈ dMᶜdt + Jʷᵉᵗ + throughf atol=1e-14
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
        # No rain, large step: the interface caps Jʷᵉᵗ at what the store holds, Mᶜ/Δt (to the
        # tolerance of the canopy solve), so the atmosphere never receives water the canopy
        # did not lose.
        model = interception_model(arch, FT; leaf_area_index = LAI, capacity = c, rain = 0.0)
        land = model.land
        set!(land; canopy_water_storage = 0.8 * c * LAI)
        Δt = 3600.0
        for _ in 1:200
            Mᶜ⁻ = scalar(land.prognostic.canopy_water_storage)
            time_step!(model, Δt)
            @test scalar(land.fluxes.canopy_evaporation) ≤ Mᶜ⁻ / Δt * (1 + 1e-5)
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

@testset "Float32 type stability" begin
    for arch in test_architectures
        FT = Float32
        model = interception_model(arch, FT; leaf_area_index = 3.0f0, capacity = 0.1f0)
        time_step!(model, 600)
        Mᶜ = scalar(model.land.prognostic.canopy_water_storage)
        Jʷᵉᵗ = scalar(model.land.fluxes.canopy_evaporation)
        @test Mᶜ isa Float32
        @test isfinite(Mᶜ) && Mᶜ ≥ 0
        @test isfinite(Jʷᵉᵗ)
    end
end
