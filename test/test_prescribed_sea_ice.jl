include("runtests_setup.jl")

using Breeze
using NumericalEarth.EarthSystemModels: sea_ice_concentration, sea_ice_thickness, interpolate_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger
using Oceananigans.Grids: halo_size
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Units

freezing_temperature = 271.38
melting_temperature = 273.15

@testset "PrescribedSeaIce concentration is clipped after interpolating in time" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size=(8, 4), longitude=(0, 360), latitude=(-90, 90),
                                     topology=(Periodic, Bounded, Flat))

        Nx, Ny, _ = size(grid)
        Hx, Hy, _ = halo_size(grid)
        south = 1:Ny÷2
        north = Ny÷2+1:Ny

        # Overshooting nodes, as in the PCMDI-AMIP boundary conditions
        times = [0.0, 100.0]
        concentration = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        interior(concentration[1], :, south, 1) .= -0.5
        interior(concentration[1], :, north, 1) .= 0.5
        interior(concentration[2], :, south, 1) .= 1
        interior(concentration[2], :, north, 1) .= 2

        sea_ice = PrescribedSeaIce(grid, times; concentration)
        exchanger = ComponentExchanger(sea_ice, grid)
        ice_concentration = exchanger.state.ℵ
        @test ice_concentration === sea_ice_concentration(sea_ice)

        @test all(Array(interior(ice_concentration, :, south, 1)) .== 0)
        @test all(Array(interior(ice_concentration, :, north, 1)) .== 0.5)

        # Halfway, the series is 0.25 and 1.25. Clipping the nodes first would give 0.5 and 0.75.
        time_step!(sea_ice, 50)
        interpolate_state!(exchanger, grid, sea_ice, nothing)
        @test all(Array(interior(ice_concentration, :, south, 1)) .≈ 0.25)
        @test all(Array(interior(ice_concentration, :, north, 1)) .== 1)

        # The periodic halo columns next to the interior, which the interface kernels read, are interpolated too
        data = Array(parent(ice_concentration))
        rows = Hy+1:Hy+Ny
        @test data[Hx, rows, 1] == data[Hx+Nx, rows, 1]
        @test data[Hx+Nx+1, rows, 1] == data[Hx+1, rows, 1]

        time_step!(sea_ice, 50)
        interpolate_state!(exchanger, grid, sea_ice, nothing)
        @test all(Array(interior(ice_concentration)) .== 1)

        # Default thickness: 2 m in the Northern Hemisphere, 1 m in the Southern
        thickness = sea_ice_thickness(sea_ice)
        @test all(Array(interior(thickness, :, south, 1)) .== 1)
        @test all(Array(interior(thickness, :, north, 1)) .== 2)
    end
end

function prescribed_sea_ice_model(arch; thickness, air_temperature, specific_humidity, downwelling_longwave)
    grid = LatitudeLongitudeGrid(arch; size=(8, 4, 1), longitude=(0, 360), latitude=(60, 90), z=(-1, 0))

    atmosphere = PrescribedAtmosphere(grid, [0.0])
    set!(atmosphere; u=5, T=air_temperature, q=specific_humidity, p=101325)

    radiation = PrescribedRadiation(grid)
    set!(radiation; downwelling_longwave)

    # A sea surface temperature below the freezing point, which the interfaces must not see
    ocean = PrescribedOcean(grid)
    set!(ocean, T=freezing_temperature - 2)

    sea_ice = PrescribedSeaIce(grid; thickness)
    set!(sea_ice, ℵ=1)

    return EarthSystemModel(; radiation, atmosphere, sea_ice, ocean)
end

# Residual of the surface energy balance over ice (W m⁻², fluxes positive upward), without the cap
function surface_energy_balance_residual(model)
    interface = model.interfaces.atmosphere_sea_ice_interface
    Tⁱⁿ = Array(interior(interface.temperature))
    𝒬ᵀ = Array(interior(interface.fluxes.sensible_heat))
    𝒬ᵛ = Array(interior(interface.fluxes.latent_heat))
    hˢⁱ = Array(interior(sea_ice_thickness(model.sea_ice)))
    k = model.sea_ice.internal_heat_flux.conductivity
    σ = model.radiation.stefan_boltzmann_constant
    ϵ = model.radiation.surface_properties.sea_ice.emissivity
    ℐꜜˡʷ = - Array(interior(model.radiation.downwelling_longwave[1]))
    return @. 𝒬ᵀ + 𝒬ᵛ + ϵ * σ * Tⁱⁿ^4 + ϵ * ℐꜜˡʷ - k * (freezing_temperature - Tⁱⁿ) / hˢⁱ
end

function equilibrated_surface_temperature(model)
    for _ in 1:40
        time_step!(model, 60)
    end
    return Array(interior(model.interfaces.atmosphere_sea_ice_interface.temperature))
end

@testset "PrescribedSeaIce surface energy balance" begin
    for arch in test_architectures
        winter = (air_temperature=245, specific_humidity=2e-4, downwelling_longwave=180)

        thin = prescribed_sea_ice_model(arch; thickness=1e-6, winter...)
        slab = prescribed_sea_ice_model(arch; thickness=2, winter...)
        thick = prescribed_sea_ice_model(arch; thickness=1e6, winter...)

        @test slab.sea_ice isa PrescribedSeaIce
        @test isnothing(slab.interfaces.sea_ice_ocean_interface)

        thin_ice_temperature  = equilibrated_surface_temperature(thin)
        slab_temperature  = equilibrated_surface_temperature(slab)
        thick_ice_temperature = equilibrated_surface_temperature(thick)

        # Thin ice: the surface is at the temperature of the ocean underneath
        @test all(isapprox.(thin_ice_temperature, freezing_temperature, atol=1e-3))

        # The diagnosed temperature closes the balance, with and without conduction
        @test maximum(abs, surface_energy_balance_residual(slab)) < 1e-2
        @test maximum(abs, surface_energy_balance_residual(thick)) < 1e-2

        # Thick ice: the conduction-free equilibrium, far colder than the ocean
        conduction = slab.sea_ice.internal_heat_flux.conductivity .* (freezing_temperature .- thick_ice_temperature) ./ 1e6
        @test maximum(abs, conduction) < 1e-3
        @test all(thick_ice_temperature .< slab_temperature .< thin_ice_temperature)
        @test all(thick_ice_temperature .< freezing_temperature - 10)

        for temperature in (thin_ice_temperature, slab_temperature, thick_ice_temperature)
            @test all(temperature .≤ melting_temperature)
        end

        # Under a warm atmosphere the surface is capped at the melting point
        summer = (air_temperature=285, specific_humidity=8e-3, downwelling_longwave=380)
        for thickness in (2, 1e6)
            melting = prescribed_sea_ice_model(arch; thickness, summer...)
            @test all(equilibrated_surface_temperature(melting) .== melting_temperature)
        end

        # The sea surface temperature is clipped where the interfaces read it, not in the data
        @test all(Array(interior(slab.ocean.sea_surface_temperature[1])) .== freezing_temperature - 2)
        @test all(Array(interior(slab.interfaces.exchanger.ocean.state.T)) .== freezing_temperature)
        @test all(Array(interior(slab.interfaces.atmosphere_ocean_interface.temperature)) .== freezing_temperature)
    end
end

# A Breeze atmosphere over a PrescribedOcean with prescribed sea ice, and optionally a SlabLand
function breeze_prescribed_sea_ice_model(arch; concentration, with_sea_ice=true, ocean_fraction=nothing)
    grid = RectilinearGrid(arch,
                           size = (16, 16), halo = (5, 5),
                           x = (-10kilometers, 10kilometers),
                           z = (0, 10kilometers),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 265

    atmosphere = atmosphere_simulation(grid; potential_temperature=θ₀)
    set!(atmosphere.model, θ=atmosphere.model.dynamics.reference_state.surface_potential_temperature, u=5)

    surface_grid = RectilinearGrid(arch,
                                   size = grid.Nx,
                                   halo = grid.Hx,
                                   x = (-10kilometers, 10kilometers),
                                   topology = (Periodic, Flat, Flat))

    ocean = PrescribedOcean(surface_grid)
    set!(ocean, T=freezing_temperature - 2)

    sea_ice = with_sea_ice ? PrescribedSeaIce(surface_grid) : nothing
    with_sea_ice && set!(sea_ice, ℵ=concentration)

    if isnothing(ocean_fraction)
        land = nothing
    else
        land = SlabLand(surface_grid)
        set!(land; T=θ₀ + 5)
    end

    return EarthSystemModel(; atmosphere, land, sea_ice, ocean, ocean_fraction)
end

@testset "Fraction-weighted fluxes over prescribed sea ice (Breeze)" begin
    for arch in test_architectures
        net_and_interfaces(model) = (model.interfaces.net_fluxes.atmosphere,
                                     model.interfaces.atmosphere_ocean_interface.fluxes,
                                     model.interfaces.atmosphere_sea_ice_interface.fluxes)

        values(field) = Array(interior(field))

        # Concentration 0: the open-ocean fluxes, identical to a model without sea ice.
        # (The ocean is at the freezing point there too, to compare like with like.)
        ice_free = breeze_prescribed_sea_ice_model(arch; concentration=0)
        net, ao, ai = net_and_interfaces(ice_free)
        @test values(net.ρE) == values(ao.sensible_heat)
        @test values(net.ρqᵛᵉ) == values(ao.water_vapor)
        @test any(values(net.ρE) .!= 0)

        no_sea_ice = breeze_prescribed_sea_ice_model(arch; concentration=0, with_sea_ice=false)
        set!(no_sea_ice.ocean, T=freezing_temperature)
        update_state!(no_sea_ice)
        open_ocean = no_sea_ice.interfaces.net_fluxes.atmosphere
        @test values(net.ρE) ≈ values(open_ocean.ρE)
        @test values(net.ρqᵛᵉ) ≈ values(open_ocean.ρqᵛᵉ)
        @test values(net.ρu) ≈ values(open_ocean.ρu)

        # Concentration 1: the ice fluxes
        ice_covered = breeze_prescribed_sea_ice_model(arch; concentration=1)
        net, ao, ai = net_and_interfaces(ice_covered)
        @test values(net.ρE) == values(ai.sensible_heat)
        @test values(net.ρqᵛᵉ) == values(ai.water_vapor)
        @test any(values(net.ρE) .!= 0)
        @test values(ai.sensible_heat) != values(ao.sensible_heat)

        # In between: each surface weighted by its fraction
        ℵˢⁱ = 0.25
        partial = breeze_prescribed_sea_ice_model(arch; concentration=ℵˢⁱ)
        net, ao, ai = net_and_interfaces(partial)
        @test values(net.ρE) ≈ (1 - ℵˢⁱ) .* values(ao.sensible_heat) .+ ℵˢⁱ .* values(ai.sensible_heat)
        @test values(net.ρqᵛᵉ) ≈ (1 - ℵˢⁱ) .* values(ao.water_vapor) .+ ℵˢⁱ .* values(ai.water_vapor)

        # With land: ocean fraction · (open water + ice) + land fraction · land
        ℵᵒᶜ = 0.6
        mixed = breeze_prescribed_sea_ice_model(arch; concentration=ℵˢⁱ, ocean_fraction=ℵᵒᶜ)
        net, ao, ai = net_and_interfaces(mixed)
        al = mixed.interfaces.atmosphere_land_interface.fluxes
        @test values(net.ρE) ≈ ℵᵒᶜ .* ((1 - ℵˢⁱ) .* values(ao.sensible_heat) .+ ℵˢⁱ .* values(ai.sensible_heat)) .+
                             (1 - ℵᵒᶜ) .* values(al.sensible_heat)


        # A few coupled steps
        for model in (partial, mixed)
            for _ in 1:3
                time_step!(model, 1)
            end
            @test model.clock.iteration == 3
            @test model.sea_ice.clock.time == 3
            @test all(isfinite, values(model.atmosphere.model.temperature))
            @test all(isfinite, values(model.interfaces.net_fluxes.atmosphere.ρE))
            @test all(values(model.interfaces.atmosphere_sea_ice_interface.temperature) .≤ melting_temperature)
        end

        # The land skin is initialized after the partition is first blended, so compare after stepping
        ocean_temperature = values(mixed.interfaces.atmosphere_ocean_interface.temperature)
        ice_temperature = values(mixed.interfaces.atmosphere_sea_ice_interface.temperature)
        land_temperature = values(NumericalEarth.EarthSystemModels.surface_temperature(mixed.interfaces.atmosphere_land_interface))
        surface_temperature = values(mixed.interfaces.surface_partition.surface_temperature)
        @test surface_temperature ≈ ℵᵒᶜ .* ((1 - ℵˢⁱ) .* ocean_temperature .+ ℵˢⁱ .* ice_temperature) .+
                                    (1 - ℵᵒᶜ) .* land_temperature
        @test all(ocean_temperature .== freezing_temperature)
    end
end

@testset "PrescribedSeaIce checkpoints its clock and its skin temperature" begin
    for arch in test_architectures
        model = prescribed_sea_ice_model(arch; thickness=2, air_temperature=250, specific_humidity=1e-4, downwelling_longwave=200)
        for _ in 1:3
            time_step!(model, 60)
        end

        sea_ice = model.sea_ice
        state = deepcopy(Oceananigans.prognostic_state(sea_ice))
        @test keys(state) == (:clock, :top_surface_temperature, :initial_surface_temperature)

        skin = Array(interior(sea_ice.top_surface_temperature))
        first_guess = Array(interior(sea_ice.initial_surface_temperature))
        @test all(skin .< freezing_temperature - 1)
        @test all(first_guess .< freezing_temperature - 1)

        # The skin solve starts from the temperature of the previous step: another `update_state!` from the same
        # state, as after a restore, finds the same skin temperature bit for bit
        update_state!(model)
        @test Array(interior(sea_ice.top_surface_temperature)) == skin
        @test Array(interior(sea_ice.initial_surface_temperature)) == first_guess

        restored = PrescribedSeaIce(sea_ice.grid; thickness=2)
        Oceananigans.restore_prognostic_state!(restored, state)
        @test restored.clock.iteration == 3
        @test restored.clock.time == 180
        @test Array(interior(restored.top_surface_temperature)) == skin
        @test Array(interior(restored.initial_surface_temperature)) == first_guess

        # A checkpoint written before the skin temperature was saved restores the clock only
        older = PrescribedSeaIce(sea_ice.grid; thickness=2)
        Oceananigans.restore_prognostic_state!(older, (; clock = state.clock))
        @test older.clock.iteration == 3
        @test all(Array(interior(older.top_surface_temperature)) .== freezing_temperature)
    end
end
