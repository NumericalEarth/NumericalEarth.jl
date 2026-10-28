include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!
using Oceananigans.TimeSteppers: update_state!, time_step!
using Oceananigans.OutputReaders: OnDisk, InMemory
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyInterface, CanopyConductanceHumidity,
    CriticalSaturation, AbsorbedPhotosyntheticallyActiveRadiation, SoilConductiveFlux, land_canopy_water_capacity
using NumericalEarth.Lands: SlabLand, SlabEnergy, VariablySaturatedHydrology, InterceptingHydrology,
    VanGenuchtenRetention, VanGenuchtenConductivity, canopy_store_update

soil_hydrology(FT) = VariablySaturatedHydrology(FT;
    slab_depth = 1.0, porosity = 0.4, storage_height = 0.1,
    retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 2.0, pore_size_uniformity = 1.5),
    hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5))

column_grid(arch = CPU()) = LatitudeLongitudeGrid(arch; size = 1, latitude = 10, longitude = 10, z = (-1, 0), topology = (Flat, Flat, Bounded))
intercepting_land(grid; kw...) = SlabLand(grid; energy = SlabEnergy(Float64), hydrology = InterceptingHydrology(Float64; soil = soil_hydrology(Float64), kw...))
canopy_storage(land) = scalar(land.prognostic.canopy_water_storage)
vapor_flux(model) = scalar(model.interfaces.atmosphere_land_interface.fluxes.water_vapor)

function interception_model(arch, FT; leaf_area_index = 3.0, capacity = 0.1, rain = 1e-3, interception = true)
    hydrology = interception ? InterceptingHydrology(FT; soil = soil_hydrology(FT), leaf_area_index, capacity_per_leaf_area = capacity) : soil_hydrology(FT)
    canopy_humidity = CanopyConductanceHumidity(FT; leaf_area_index, moisture_stress = CriticalSaturation(0.5), absorbed_par = AbsorbedPhotosyntheticallyActiveRadiation(FT))
    surface = CanopyInterface(FT; soil_humidity = bare_soil_humidity(FT), canopy_humidity, ground_heat_flux = SoilConductiveFlux(1.5, 0.05))
    return coupled_land_model(arch, FT; air_temperature = 298, air_specific_humidity = 0.010, wind = 3, rain,
                              hydrology, atmosphere_land_interface_state=surface, land_temperature = 298, water = 300.0, shortwave = 400)
end

@testset "Interception conserves canopy and soil water" begin
    for arch in test_architectures
        model = interception_model(arch, Float64)
        land = model.land
        before = scalar(land.water_storage) + canopy_storage(land)
        time_step!(model, 60.0)
        @test scalar(land.water_storage) + canopy_storage(land) - before ≈ 60 * (1e-3 - vapor_flux(model)) atol = 1e-10
        @test land.hydrology.leaf_area_index == 3.0
    end
end

@testset "InterceptingHydrology declarations" begin
    for arch in test_architectures
        land = intercepting_land(column_grid(arch); leaf_area_index = 3.0)
        @test :canopy_water_storage in keys(land.prognostic)
        @test (:canopy_evaporation, :liquid_water_precipitation_flux, :vapor_flux) ⊆ keys(land.fluxes)
        @test (:throughfall, :water_storage_tendency) ⊆ keys(land.diagnostics)
    end
end

@testset "Buffered canopy properties conserve water across time windows" begin
    grid, times = column_grid(), 0.0:1.0:4.0
    path = joinpath(mktempdir(), "lai.jld2")
    on_disk(name) = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend = OnDisk(), path, name)
    lai = on_disk("lai")
    field = Field{Center, Center, Nothing}(grid)
    for n in eachindex(times)
        set!(lai, set!(field, 1 + times[n]), n)
    end

    in_memory(name) = FieldTimeSeries(path, name; backend = InMemory(2), architecture = CPU())
    land = intercepting_land(grid; leaf_area_index = in_memory("lai"), capacity_per_leaf_area = 0.1)
    set!(land; M = 300, canopy_water_storage = 0.05)
    before = scalar(land.water_storage) + canopy_storage(land)
    for n in 1:3
        time_step!(land, 1.0)
        @test scalar(land.water_storage) + canopy_storage(land) ≈ before atol = 1e-10
    end

    # λˡᵉᵃᶠ(t = 3) = 4 lies beyond the two snapshots loaded when the series was opened
    capacity = land_canopy_water_capacity(land)
    @test scalar(compute!(Field(capacity))) ≈ 0.1 * 4
end

@testset "Canopy store budget closes when the store empties mid-step" begin
    Mᶜᵃⁿ, rain, Mᶜᵃⁿ⁺, fⁱⁿᵗ, Δt = 0.05, 1e-4, 0.3, 0.6, 600.0
    for Jʷᵉᵗ in (0.0, 1e-4, 5e-4)
        Mᶜᵃⁿⁿ⁺¹, drip, throughfall = canopy_store_update(Mᶜᵃⁿ, rain, Jʷᵉᵗ, Mᶜᵃⁿ⁺, fⁱⁿᵗ, Δt)
        @test 0 ≤ Mᶜᵃⁿⁿ⁺¹ ≤ Mᶜᵃⁿ⁺
        @test drip ≥ 0
        @test rain ≈ (Mᶜᵃⁿⁿ⁺¹ - Mᶜᵃⁿ) / Δt + Jʷᵉᵗ + throughfall atol = 1e-15
    end
end

@testset "Dry canopy reduces to the bare CanopyInterface bit-for-bit" begin
    for arch in test_architectures
        with, without = interception_model(arch, Float64), interception_model(arch, Float64; interception = false)
        interface, bare = with.interfaces.atmosphere_land_interface, without.interfaces.atmosphere_land_interface
        @test canopy_storage(with.land) == 0
        for name in (:interface, :canopy, :ground, :effective)
            @test scalar(getproperty(interface.temperature, name)) == scalar(getproperty(bare.temperature, name))
        end
        for name in (:sensible_heat, :latent_heat, :water_vapor)
            @test scalar(getproperty(interface.fluxes, name)) == scalar(getproperty(bare.fluxes, name))
        end
        @test scalar(interface.temperature.canopy_evaporation) == 0
        @test scalar(with.land.fluxes.vapor_flux) ≈ vapor_flux(with) rtol = 1e-6
    end
end

@testset "Wet canopy fills to capacity and evaporates" begin
    for arch in test_architectures
        model = interception_model(arch, Float64)
        land = model.land
        foreach(_ -> time_step!(model, 600.0), 1:40)
        @test 0 ≤ canopy_storage(land) ≤ 0.1 * 3 + eps(0.1 * 3)
        @test canopy_storage(land) ≈ 0.1 * 3 rtol=1e-6
        Jʷᵉᵗ = scalar(land.fluxes.canopy_evaporation)
        @test Jʷᵉᵗ > 0
        @test Jʷᵉᵗ == scalar(model.interfaces.atmosphere_land_interface.temperature.canopy_evaporation)
        @test scalar(land.fluxes.vapor_flux) ≈ vapor_flux(model) - Jʷᵉᵗ rtol = 1e-5
    end
end

@testset "Canopy water mass conservation" begin
    for arch in test_architectures
        model = interception_model(arch, Float64)
        land = model.land
        set!(land; canopy_water_storage = 0.5 * 0.1 * 3)
        update_state!(model)
        rain = scalar(land.fluxes.liquid_water_precipitation_flux)
        Jʷᵉᵗ = scalar(land.fluxes.canopy_evaporation)
        Mᶜᵃⁿⁿ = canopy_storage(land)
        time_step!(model, 600.0)
        throughfall = scalar(land.diagnostics.throughfall)
        @test rain ≈ (canopy_storage(land) - Mᶜᵃⁿⁿ) / 600 + Jʷᵉᵗ + throughfall atol=1e-14
        @test scalar(land.fluxes.liquid_water_precipitation_flux) == rain
        @test 0 ≤ throughfall < rain
    end
end

@testset "Store stays non-negative under aggressive drydown" begin
    for arch in test_architectures
        model = interception_model(arch, Float64; rain = 0.0)
        land = model.land
        set!(land; canopy_water_storage = 0.8 * 0.1 * 3)
        for _ in 1:200
            Mᶜᵃⁿ⁻ = canopy_storage(land)
            time_step!(model, 3600.0)
            @test scalar(land.fluxes.canopy_evaporation) ≤ Mᶜᵃⁿ⁻ / 3600 * (1 + 1e-5)
            @test canopy_storage(land) ≥ 0
        end
        @test canopy_storage(land) < 1e-3
    end
end

@testset "Field- and FieldTimeSeries-valued λˡᵉᵃᶠ cap the store at c λˡᵉᵃᶠ" begin
    for arch in test_architectures
        grid = column_grid(arch)
        lai_field = set!(CenterField(grid), 2.0)
        lai_series = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 100.0])
        set!(lai_series[1], 1.0)
        set!(lai_series[2], 3.0)
        # The FieldTimeSeries λˡᵉᵃᶠ is interpolated to λˡᵉᵃᶠ(t = 50) = 2 and runs on CPU only
        for (lai, Δt) in ((lai_field, 600.0), (lai_series, 50.0))
            lai isa FieldTimeSeries && !(arch isa CPU) && continue
            land = intercepting_land(grid; leaf_area_index = lai, capacity_per_leaf_area = 0.1)
            set!(land; T = 295, M = 300, canopy_water_storage = 1.0)
            fill!(parent(land.fluxes.liquid_water_precipitation_flux), 0)
            fill!(parent(land.fluxes.canopy_evaporation), 0)
            time_step!(land, Δt)
            @test canopy_storage(land) ≈ 0.2 rtol=1e-6
        end
    end
end

@testset "Checkpoint round-trip of the canopy water store" begin
    for arch in test_architectures
        grid = column_grid(arch)
        land = intercepting_land(grid; leaf_area_index = 3.0)
        set!(land; T = 290, M = 120, canopy_water_storage = 0.17)
        state = Oceananigans.prognostic_state(land)
        @test :canopy_water_storage in keys(state.prognostic)
        restored = intercepting_land(grid; leaf_area_index = 3.0)
        Oceananigans.restore_prognostic_state!(restored, state)
        @test canopy_storage(restored) == 0.17
        @test scalar(restored.temperature) == 290
    end
end

@testset "Float32 type stability" begin
    for arch in test_architectures
        model = interception_model(arch, Float32; leaf_area_index = 3.0f0, capacity = 0.1f0)
        time_step!(model, 600)
        Mᶜᵃⁿ = canopy_storage(model.land)
        @test Mᶜᵃⁿ isa Float32 && isfinite(Mᶜᵃⁿ) && Mᶜᵃⁿ ≥ 0
        @test isfinite(scalar(model.land.fluxes.canopy_evaporation))
    end
end
