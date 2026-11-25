include("runtests_setup.jl")

using Breeze
using RRTMGP   # activates Breeze's radiative-transfer extension (RadiativeTransferModel)
using Dates: DateTime
using Oceananigans: initialize!
using Oceananigans.OutputWriters: Checkpointer
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.TimeSteppers: time_step!, reconcile_state!
using Oceananigans.Units

using NumericalEarth.EarthSystemModels: sea_ice_concentration, sea_ice_thickness
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

const σ = default_stefan_boltzmann_constant
const freezing_temperature = 271.38
const melting_temperature = 273.15
const land_emissivity = 0.95
const midsummer_noon = DateTime(2024, 6, 20, 12)
const midwinter_noon = DateTime(2024, 12, 21, 12)

surface_values(field) = Array(interior(field))[:, :, 1]

# Fractions that differ in every cell, with all three surfaces present everywhere
pattern(a, b, Nx, Ny) = [mod(a * i + b * j, 1) for i in 1:Nx, j in 1:Ny]

# A Breeze atmosphere and RadiativeTransferModel over a mosaic of SlabLand, PrescribedOcean and PrescribedSeaIce
# on a polar `LatitudeLongitudeGrid`, whose cell areas shrink by a factor of 3 from south to north.
function mosaic_model(arch; optics = GrayOptics(), albedos = (; surface_albedo = 0.2), epoch = midsummer_noon,
                      with_land = true, with_radiation = true, uniform_sea_ice = false, thickness = 2,
                      potential_temperature = 265, times = [0.0], interface_kw...)
    Nx, Ny = 8, 6
    longitude, latitude = (0, 60), (70, 85)

    grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, 8), halo = (5, 5, 5), longitude, latitude, z = (0, 8kilometers))
    surface_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny), halo = (5, 5), longitude, latitude,
                                         topology = (Bounded, Bounded, Flat))

    constants = ThermodynamicConstants(Float64)
    atmosphere = atmosphere_simulation(grid; thermodynamic_constants = constants, potential_temperature)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.surface_potential_temperature, u = 5)

    # A sea surface temperature below the freezing point, which the interfaces must not see
    ocean = PrescribedOcean(surface_grid, times)
    set!(ocean, T = freezing_temperature - 2)

    sea_ice = PrescribedSeaIce(surface_grid, times; thickness)
    concentration = uniform_sea_ice ? ones(Nx, Ny) : 0.1 .+ 0.8 .* pattern(0.37, 0.61, Nx, Ny)
    copyto!(interior(sea_ice.concentration[1]), on_architecture(arch, reshape(concentration, Nx, Ny, 1)))
    set!(sea_ice)

    if with_land
        land = SlabLand(surface_grid)
        set!(land; T = potential_temperature + 3)
        ocean_fraction = on_architecture(arch, 0.2 .+ 0.6 .* pattern(0.23, 0.71, Nx, Ny))
    else
        land = nothing
        ocean_fraction = nothing
    end

    radiation = with_radiation ? RadiativeTransferModel(grid, optics, constants;
                                                        solar_position = ApparentSolarPosition(; epoch),
                                                        albedos...,
                                                        surface_emissivity = land_emissivity,
                                                        schedule = IterationInterval(1)) : nothing

    return EarthSystemModel(; radiation, atmosphere, land, sea_ice, ocean, ocean_fraction, interface_kw...)
end

# With the atmosphere held at its initial state, alternate the radiation solve and the interface computations,
# so that the interface states respond to the downwelling fluxes that their own surface produces. The last
# radiation solve is from exactly the surface that the interface computations left, so the RTM's fluxes and
# the fluxes of the components describe one state. (During time stepping the two are one coupling step apart.)
function equilibrate_surface!(model; cycles = 8)
    for _ in 1:cycles
        reconcile_state!(model)
    end
    initialize!(model)
    return model
end

function surface_fractions(model)
    ℵᵒᶜ = model.interfaces.surface_partition.ocean_fraction
    ℵˢⁱ = surface_values(sea_ice_concentration(model.sea_ice))
    ℵᵒᶜ = ℵᵒᶜ isa Field ? surface_values(ℵᵒᶜ) : ones(size(ℵˢⁱ))
    return (land = 1 .- ℵᵒᶜ, ocean = ℵᵒᶜ .* (1 .- ℵˢⁱ), sea_ice = ℵᵒᶜ .* ℵˢⁱ)
end

# Fraction-weighted sum over the surfaces of a quantity `f(interface_radiation_flux)` (W m⁻², positive upward)
function mosaic_sum(f, model)
    fluxes = model.interfaces.exchanger.radiation.state.interface_fluxes
    fractions = surface_fractions(model)
    return sum(fractions[name] .* f(fluxes[name]) for name in (:land, :ocean, :sea_ice))
end

net_longwave(fluxes) = surface_values(fluxes.upwelling_longwave) .+ surface_values(fluxes.downwelling_longwave)
net_shortwave(fluxes) = surface_values(fluxes.downwelling_shortwave)

# The RTM's net fluxes at the surface face (W m⁻², positive upward). Gray optics has no upwelling shortwave
# stream: the surface absorbs (1 - α) of the direct beam and the rest leaves the model.
rtm_net_longwave(rtm) = surface_values(rtm.upwelling_longwave_flux) .+ surface_values(rtm.downwelling_longwave_flux)

rtm_net_shortwave(rtm, ::GrayOptics) =
    (1 .- surface_values(rtm.surface_radiation.direct_surface_albedo)) .* surface_values(rtm.downwelling_shortwave_flux)

rtm_net_shortwave(rtm, optics) =
    surface_values(rtm.upwelling_shortwave_flux) .+ surface_values(rtm.downwelling_shortwave_flux)

function cell_areas(model)
    grid = model.interfaces.exchanger.grid
    Nx, Ny, _ = size(grid)
    return @allowscalar [Azᶜᶜᶜ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny]
end

@testset "The Breeze RTM sees the mosaic of land, open ocean and sea ice" begin
    for arch in test_architectures
        model = equilibrate_surface!(mosaic_model(arch))
        rtm = model.radiation
        state = model.interfaces.exchanger.radiation.state
        surface = state.radiative_surface
        fractions = surface_fractions(model)

        # The RTM reads the mosaic held by the exchanger
        @test rtm.surface_radiation.surface_temperature === surface.temperature
        @test rtm.surface_radiation.surface_emissivity === surface.emissivity
        @test rtm.surface_radiation.direct_surface_albedo === surface.direct_albedo
        @test rtm.surface_radiation.diffuse_surface_albedo === surface.diffuse_albedo

        @test all(0.1 .< fractions.land .< 0.9)
        @test all(fractions.ocean .> 0.01)
        @test all(fractions.sea_ice .> 0.01)

        Tᵒᶜ = surface_values(model.interfaces.atmosphere_ocean_interface.temperature)
        Tˢⁱ = surface_values(model.interfaces.atmosphere_sea_ice_interface.temperature)
        Tˡᵃ = surface_values(model.interfaces.atmosphere_land_interface.temperature)
        αᵒᶜ = surface_values(state.surface_properties.ocean.albedo.field)
        αˢⁱ = surface_values(state.surface_properties.sea_ice.albedo.field)
        ϵᵒᶜ = state.surface_properties.ocean.emissivity
        ϵˢⁱ = state.surface_properties.sea_ice.emissivity

        # Default optics. Ocean: Large and Yeager (2009) latitude-dependent albedo, emissivity 0.97.
        # Sea ice: CCSM3 bare-ice albedo of `SeaIceAlbedo`, 0.54 cold and 0.465 at the melting point; emissivity 1.
        φ = Array(φnodes(model.interfaces.exchanger.grid, Center()))
        @test αᵒᶜ ≈ [0.069 - 0.011 * cosd(2φ[j]) for i in 1:size(αᵒᶜ, 1), j in 1:size(αᵒᶜ, 2)]
        @test ϵᵒᶜ == 0.97
        @test ϵˢⁱ == 1
        @test all(0.465 - 1e-12 .≤ αˢⁱ .≤ 0.54 + 1e-12)
        @test all(αˢⁱ .> αᵒᶜ .+ 0.3)

        ϵ = fractions.land .* land_emissivity .+ fractions.ocean .* ϵᵒᶜ .+ fractions.sea_ice .* ϵˢⁱ
        ϵT⁴ = fractions.land .* land_emissivity .* Tˡᵃ .^ 4 .+ fractions.ocean .* ϵᵒᶜ .* Tᵒᶜ .^ 4 .+
              fractions.sea_ice .* ϵˢⁱ .* Tˢⁱ .^ 4
        α = fractions.land .* 0.2 .+ fractions.ocean .* αᵒᶜ .+ fractions.sea_ice .* αˢⁱ

        @test surface_values(surface.emissivity) ≈ ϵ rtol=1e-14
        @test surface_values(surface.temperature) ≈ (ϵT⁴ ./ ϵ) .^ (1/4) rtol=1e-14
        @test surface_values(surface.direct_albedo) ≈ α rtol=1e-14
        @test surface_values(surface.diffuse_albedo) ≈ α rtol=1e-14

        # The mosaic temperature lies between those of its surfaces, which differ
        T = surface_values(surface.temperature)
        @test all(min.(Tˡᵃ, Tᵒᶜ, Tˢⁱ) .≤ T .≤ max.(Tˡᵃ, Tᵒᶜ, Tˢⁱ))
        @test maximum(Tˡᵃ) < minimum(Tᵒᶜ)

        # The RTM radiates from the mosaic: emission plus the reflected part of the downwelling longwave
        ℐꜜˡʷ = surface_values(rtm.downwelling_longwave_flux)
        @test surface_values(rtm.upwelling_longwave_flux) ≈ σ .* ϵT⁴ .- (1 .- ϵ) .* ℐꜜˡʷ rtol=1e-12

        # The surfaces receive the RTM's downwelling fluxes
        @test surface_values(state.ℐꜜˡʷ) == ℐꜜˡʷ
        @test surface_values(state.ℐꜜˢʷ) == surface_values(rtm.downwelling_shortwave_flux)
        @test all(ℐꜜˡʷ .< -100)
        @test all(surface_values(state.ℐꜜˢʷ) .< -100)

        # The properties can be set per surface; a radiation that carries its own rejects them
        properties = (; ocean = SurfaceRadiationProperties(0.1, 0.9))
        model = mosaic_model(arch; surface_radiation_properties = properties)
        state = model.interfaces.exchanger.radiation.state
        @test all(surface_values(state.surface_properties.ocean.albedo.field) .== 0.1)
        @test state.surface_properties.ocean.emissivity == 0.9
        @test state.surface_properties.sea_ice.emissivity == 1

        grid = model.interfaces.exchanger.grid
        @test_throws ArgumentError EarthSystemModel(; radiation = PrescribedRadiation(grid),
                                                    atmosphere = PrescribedAtmosphere(grid, [0.0]),
                                                    ocean = PrescribedOcean(grid),
                                                    sea_ice = nothing,
                                                    surface_radiation_properties = properties)
    end
end

@testset "Net surface radiation of the components equals the RTM's net surface flux" begin
    for arch in test_architectures
        two_stream_albedos = (direct_surface_albedo = 0.2, diffuse_surface_albedo = 0.35)

        # Longwave tolerance: gray optics emits σ T⁴, as the surfaces do. RRTMGP's band-integrated Planck
        # source differs from σ T⁴ in the fifth digit (0.01 to 0.02 W m⁻² near 270 K), which bounds the
        # agreement of full-spectrum optics.
        cases = ((optics = GrayOptics(),     albedos = (; surface_albedo = 0.2), longwave_rtol = 1e-12, longwave_atol = 1e-9),
                 (optics = ClearSkyOptics(), albedos = two_stream_albedos,       longwave_rtol = 5e-4,  longwave_atol = 5e-2))

        @testset "$(nameof(typeof(case.optics))) at $epoch on $(typeof(arch))" for case in cases, epoch in (midsummer_noon, midwinter_noon)
            optics = case.optics
            model = equilibrate_surface!(mosaic_model(arch; optics, epoch, albedos = case.albedos))
            rtm = model.radiation
            A = cell_areas(model)
            @test maximum(A) > 2.5 * minimum(A)

            component_longwave = mosaic_sum(net_longwave, model)
            component_shortwave = mosaic_sum(net_shortwave, model)
            radiation_longwave = rtm_net_longwave(rtm)
            radiation_shortwave = rtm_net_shortwave(rtm, optics)

            # The components absorbed the downwelling fluxes of the RTM's last solve
            state = model.interfaces.exchanger.radiation.state
            @test surface_values(state.ℐꜜˡʷ) == surface_values(rtm.downwelling_longwave_flux)
            @test surface_values(state.ℐꜜˢʷ) == surface_values(rtm.downwelling_shortwave_flux)

            # Area-integrated (W), and in each cell (W m⁻²)
            @test sum(A .* component_longwave) ≈ sum(A .* radiation_longwave) rtol=case.longwave_rtol
            @test sum(A .* component_shortwave) ≈ sum(A .* radiation_shortwave) rtol=1e-12 atol=1e-3
            @test maximum(abs, component_longwave .- radiation_longwave) < case.longwave_atol
            @test maximum(abs, component_shortwave .- radiation_shortwave) < 1e-9

            net_component = sum(A .* (component_longwave .+ component_shortwave))
            net_radiation = sum(A .* (radiation_longwave .+ radiation_shortwave))
            @test net_component ≈ net_radiation rtol=case.longwave_rtol

            # The test is sensitive: the surfaces absorb and emit differently
            fluxes = model.interfaces.exchanger.radiation.state.interface_fluxes
            @test maximum(abs, net_longwave(fluxes.sea_ice) .- net_longwave(fluxes.ocean)) > 1
            @test maximum(abs, net_longwave(fluxes.land) .- net_longwave(fluxes.ocean)) > 1

            if epoch == midsummer_noon
                @test sum(A .* component_shortwave) < -100 * sum(A)
                @test maximum(abs, net_shortwave(fluxes.sea_ice) .- net_shortwave(fluxes.ocean)) > 100

                # The land absorbs with its own blue-sky albedo, between the configured direct and diffuse ones
                land_albedo = 1 .- net_shortwave(fluxes.land) ./ surface_values(rtm.downwelling_shortwave_flux)
                @test all(minimum(case.albedos) - 1e-10 .≤ land_albedo .≤ maximum(case.albedos) + 1e-10)
            else
                @test all(component_shortwave .== 0)
            end

            # A few coupled steps: the radiation lags the surface by one step, so the budgets stay close
            for _ in 1:3
                time_step!(model, 1)
            end
            @test model.clock.iteration == 3
            @test all(isfinite, surface_values(model.atmosphere.model.temperature))
            lag = mosaic_sum(net_longwave, model) .- rtm_net_longwave(rtm)
            @test maximum(abs, lag) < 2
        end
    end
end

# Residual of the surface energy balance over ice (W m⁻², fluxes positive upward), without the cap
function sea_ice_energy_balance_residual(model)
    interface = model.interfaces.atmosphere_sea_ice_interface
    state = model.interfaces.exchanger.radiation.state
    Tⁱⁿ = surface_values(interface.temperature)
    𝒬ᵀ = surface_values(interface.fluxes.sensible_heat)
    𝒬ᵛ = surface_values(interface.fluxes.latent_heat)
    hˢⁱ = surface_values(sea_ice_thickness(model.sea_ice))
    k = model.sea_ice.internal_heat_flux.conductivity
    ϵ = state.surface_properties.sea_ice.emissivity
    α = surface_values(state.surface_properties.sea_ice.albedo.field)
    ℐꜜˡʷ = surface_values(state.ℐꜜˡʷ)
    ℐꜜˢʷ = surface_values(state.ℐꜜˢʷ)
    return @. 𝒬ᵀ + 𝒬ᵛ + ϵ * σ * Tⁱⁿ^4 + ϵ * ℐꜜˡʷ + (1 - α) * ℐꜜˢʷ - k * (freezing_temperature - Tⁱⁿ) / hˢⁱ
end

@testset "Polar night: the ice skin cools to the balance with the downwelling longwave" begin
    for arch in test_architectures
        @testset "$(nameof(typeof(optics))) on $(typeof(arch))" for optics in (GrayOptics(), ClearSkyOptics())
            kw = (; optics, epoch = midwinter_noon, with_land = false, uniform_sea_ice = true)
            slab = equilibrate_surface!(mosaic_model(arch; kw...); cycles = 40)
            thick = equilibrate_surface!(mosaic_model(arch; thickness = 1e6, kw...); cycles = 40)
            no_radiation = equilibrate_surface!(mosaic_model(arch; with_radiation = false, kw...); cycles = 40)

            state = slab.interfaces.exchanger.radiation.state
            ℐꜜˡʷ = surface_values(state.ℐꜜˡʷ)
            @test all(surface_values(state.ℐꜜˢʷ) .== 0)
            @test all(surface_values(slab.radiation.downwelling_shortwave_flux) .== 0)
            @test all(ℐꜜˡʷ .< -50)

            skin_temperature(model) = surface_values(model.interfaces.atmosphere_sea_ice_interface.temperature)
            Tˢˡᵃᵇ = skin_temperature(slab)
            Tᵗʰⁱᶜᵏ = skin_temperature(thick)
            air_temperature = surface_values(slab.interfaces.exchanger.atmosphere.state.T)

            # The diagnosed skin closes the balance that includes the RTM's downwelling longwave
            @test maximum(abs, sea_ice_energy_balance_residual(slab)) < 1e-6
            @test maximum(abs, sea_ice_energy_balance_residual(thick)) < 1e-6

            # Without radiation the skin sits between the air and the ocean. With it, the net longwave loss
            # cools the skin, and without conduction from the ocean it cools below the air.
            Tⁿᵒ = skin_temperature(no_radiation)
            @test all(air_temperature .< Tⁿᵒ .< freezing_temperature)
            @test all(Tˢˡᵃᵇ .< Tⁿᵒ .- 1)
            @test all(Tᵗʰⁱᶜᵏ .< Tˢˡᵃᵇ)
            @test all(Tᵗʰⁱᶜᵏ .< air_temperature)
            @test all(Tˢˡᵃᵇ .< freezing_temperature - 5)

            # Thick ice has no conduction: its skin emits what it absorbs from the sky and from the air.
            # The radiative equilibrium with the downwelling longwave is its lower bound.
            radiative_equilibrium = (.- ℐꜜˡʷ ./ σ) .^ (1/4)
            ℐꜜˡʷ_thick = surface_values(thick.interfaces.exchanger.radiation.state.ℐꜜˡʷ)
            thick_fluxes = thick.interfaces.atmosphere_sea_ice_interface.fluxes
            turbulent = surface_values(thick_fluxes.sensible_heat) .+ surface_values(thick_fluxes.latent_heat)
            @test ℐꜜˡʷ_thick ≈ ℐꜜˡʷ
            @test all(turbulent .< 0)
            @test all(radiative_equilibrium .< Tᵗʰⁱᶜᵏ)
            @test maximum(abs, σ .* Tᵗʰⁱᶜᵏ .^ 4 .+ ℐꜜˡʷ .+ turbulent) < 1e-3

            # The RTM radiates at the ice skin
            @test surface_values(state.radiative_surface.temperature) ≈ Tˢˡᵃᵇ rtol=1e-14
            @test all(surface_values(state.radiative_surface.emissivity) .== 1)

            # Time stepping keeps the night dark and the skin near the balance
            for _ in 1:3
                time_step!(slab, 1)
            end
            @test all(surface_values(state.ℐꜜˢʷ) .== 0)
            @test maximum(abs, sea_ice_energy_balance_residual(slab)) < 1e-2
            @test all(skin_temperature(slab) .< freezing_temperature - 5)
        end
    end
end

@testset "Sea surface temperature clipping and ice skin capping under the Breeze RTM" begin
    for arch in test_architectures
        # Midsummer noon over a warm atmosphere: the ice skin is at the melting point
        model = equilibrate_surface!(mosaic_model(arch; potential_temperature = 285, uniform_sea_ice = true, with_land = false))
        state = model.interfaces.exchanger.radiation.state
        skin = model.interfaces.atmosphere_sea_ice_interface.temperature

        @test all(surface_values(state.ℐꜜˢʷ) .< -100)
        @test all(surface_values(skin) .== melting_temperature)
        @test all(surface_values(state.interface_fluxes.sea_ice.upwelling_longwave) .≈ σ * melting_temperature^4)
        @test all(surface_values(state.radiative_surface.temperature) .≈ melting_temperature)

        # The uncapped balance would be warmer: at the cap the surface gains energy, which is discarded
        @test all(sea_ice_energy_balance_residual(model) .< -10)

        for _ in 1:3
            time_step!(model, 1)
        end
        @test all(surface_values(skin) .≤ melting_temperature)

        # The sea surface temperature is clipped where the interfaces and the radiation read it, not in the data
        model = equilibrate_surface!(mosaic_model(arch))
        state = model.interfaces.exchanger.radiation.state
        ϵᵒᶜ = state.surface_properties.ocean.emissivity
        @test all(Array(interior(model.ocean.sea_surface_temperature[1])) .== freezing_temperature - 2)
        @test all(surface_values(model.interfaces.exchanger.ocean.state.T) .== freezing_temperature)
        @test all(surface_values(model.interfaces.atmosphere_ocean_interface.temperature) .== freezing_temperature)
        @test all(surface_values(state.interface_fluxes.ocean.upwelling_longwave) .≈ ϵᵒᶜ * σ * freezing_temperature^4)
        @test all(surface_values(model.interfaces.atmosphere_sea_ice_interface.temperature) .≤ melting_temperature)

        for _ in 1:3
            time_step!(model, 1)
        end
        @test all(surface_values(model.interfaces.atmosphere_ocean_interface.temperature) .== freezing_temperature)
        @test all(surface_values(model.interfaces.atmosphere_sea_ice_interface.temperature) .≤ melting_temperature)
    end
end

@testset "A slab ocean absorbs the radiation of the Breeze RTM" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (8, 8), halo = (5, 5), x = (0, 8kilometers), z = (0, 8kilometers),
                               topology = (Periodic, Flat, Bounded))
        surface_grid = RectilinearGrid(arch; size = 8, halo = 5, x = (0, 8kilometers), topology = (Periodic, Flat, Flat))

        constants = ThermodynamicConstants(Float64)
        atmosphere = atmosphere_simulation(grid; thermodynamic_constants = constants, potential_temperature = 285)
        set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.surface_potential_temperature, u = 5)

        density, heat_capacity, depth = 1025, 4000, 50
        ocean = SlabOcean(surface_grid; depth, density, heat_capacity)
        set!(ocean, T = 288)

        solar_position = ApparentSolarPosition(coordinate = (0, 30), epoch = DateTime(2024, 3, 20, 12))
        radiation = RadiativeTransferModel(grid, GrayOptics(), constants; solar_position, surface_albedo = 0.3,
                                           surface_emissivity = 0.9, schedule = IterationInterval(1))

        model = AtmosphereOceanModel(atmosphere, ocean; radiation)
        state = model.interfaces.exchanger.radiation.state
        fluxes = state.interface_fluxes.ocean
        turbulent = model.interfaces.atmosphere_ocean_interface.fluxes

        # Without land the RTM's configured optics are unused: the ocean's are the default, 0.06 and 0.97
        @test all(surface_values(state.radiative_surface.direct_albedo) .== 0.06)
        @test all(surface_values(state.radiative_surface.emissivity) .== 0.97)
        @test all(surface_values(state.radiative_surface.temperature) .≈ 288)

        radiative = net_longwave(fluxes) .+ net_shortwave(fluxes)
        ℐꜜˡʷ = surface_values(state.ℐꜜˡʷ)
        ℐꜜˢʷ = surface_values(state.ℐꜜˢʷ)
        @test radiative ≈ 0.97 .* σ .* 288 .^ 4 .+ 0.97 .* ℐꜜˡʷ .+ (1 - 0.06) .* ℐꜜˢʷ
        @test all(radiative .< -100)

        # The slab's heat flux is turbulent plus radiative, and warms the slab accordingly
        heat_flux = density * heat_capacity .* surface_values(model.interfaces.net_fluxes.ocean.T)
        @test heat_flux ≈ surface_values(turbulent.sensible_heat) .+ surface_values(turbulent.latent_heat) .+ radiative

        initial_temperature = surface_values(ocean.temperature)
        Δt = 10
        time_step!(model, Δt)
        warming = density * heat_capacity * depth .* (surface_values(ocean.temperature) .- initial_temperature) ./ Δt
        @test warming ≈ .- heat_flux rtol=1e-6
        @test all(warming .> 0)
    end
end

#####
##### The sea ice skin solve converges
#####

@testset "The sea ice skin solve converges under clear-sky midsummer forcing" begin
    for arch in test_architectures
        # One of the 48 cells of this model did not converge in 100 iterations: its skin wandered between 264 and
        # 272 K, the solve returned 267.88 or 268.51 K depending on where it started, and left 11 W m⁻² of the
        # surface energy balance unclosed. The similarity scales do not converge there at fixed temperature either.
        two_stream = (direct_surface_albedo = 0.2, diffuse_surface_albedo = 0.35)
        model = equilibrate_surface!(mosaic_model(arch; optics = ClearSkyOptics(), epoch = midsummer_noon, albedos = two_stream))
        skin = model.interfaces.atmosphere_sea_ice_interface.temperature

        # Solve again from three first guesses, with the radiation and the published albedos held
        solutions = map((268, freezing_temperature, 262)) do first_guess
            fill!(parent(skin), first_guess)
            update_state!(model)

            residual = sea_ice_energy_balance_residual(model)
            Tˢⁱ = surface_values(skin)
            capped = Tˢⁱ .== melting_temperature
            @info "Sea ice skin solve from $first_guess K: max |balance residual| below the cap = $(maximum(abs, residual[.!capped])) W m⁻², capped cells = $(count(capped)) of $(length(capped))"

            # The balance closes below the melting point, and at the cap the surface gains energy
            @test 0 < count(capped) < length(capped)
            @test maximum(abs, residual[.!capped]) < 1e-3
            @test all(residual[capped] .≤ 0)
            Tˢⁱ
        end

        # The solution does not depend on where the iteration starts
        @test maximum(abs, solutions[2] .- solutions[1]) < 1e-5
        @test maximum(abs, solutions[3] .- solutions[1]) < 1e-5
    end
end

#####
##### Restart
#####

# What a restarted run must reproduce: the prognostic state of the atmosphere and the land, the fluxes of the
# radiative transfer model, the interface states, the fluxes that the atmosphere receives, and what the radiation
# exchanger publishes.
function restart_fingerprint(model)
    atmosphere = model.atmosphere.model
    rtm = model.radiation
    interfaces = model.interfaces
    state = interfaces.exchanger.radiation.state
    array(field) = Array(interior(field))

    fields = (; map(array, Oceananigans.prognostic_fields(atmosphere))...,
                land_temperature = array(model.land.temperature),
                rtm_downwelling_longwave = array(rtm.downwelling_longwave_flux),
                rtm_upwelling_longwave = array(rtm.upwelling_longwave_flux),
                rtm_downwelling_shortwave = array(rtm.downwelling_shortwave_flux),
                rtm_flux_divergence = array(rtm.flux_divergence),
                sea_ice_skin_temperature = array(interfaces.atmosphere_sea_ice_interface.temperature),
                sea_ice_sensible_heat = array(interfaces.atmosphere_sea_ice_interface.fluxes.sensible_heat),
                sea_ice_latent_heat = array(interfaces.atmosphere_sea_ice_interface.fluxes.latent_heat),
                ocean_interface_temperature = array(interfaces.atmosphere_ocean_interface.temperature),
                ocean_sensible_heat = array(interfaces.atmosphere_ocean_interface.fluxes.sensible_heat),
                land_interface_temperature = array(interfaces.atmosphere_land_interface.temperature),
                sea_ice_concentration = array(sea_ice_concentration(model.sea_ice)),
                surface_temperature = array(interfaces.surface_partition.surface_temperature),
                net_energy_flux = array(interfaces.net_fluxes.atmosphere.ρE),
                net_x_momentum_flux = array(interfaces.net_fluxes.atmosphere.ρu),
                ocean_albedo = array(state.surface_properties.ocean.albedo.field),
                sea_ice_albedo = array(state.surface_properties.sea_ice.albedo.field),
                land_albedo = array(state.surface_properties.land.albedo),
                mosaic_temperature = array(state.radiative_surface.temperature),
                mosaic_emissivity = array(state.radiative_surface.emissivity),
                mosaic_direct_albedo = array(state.radiative_surface.direct_albedo),
                mosaic_diffuse_albedo = array(state.radiative_surface.diffuse_albedo),
                sea_ice_absorbed_shortwave = array(state.interface_fluxes.sea_ice.downwelling_shortwave),
                ocean_absorbed_shortwave = array(state.interface_fluxes.ocean.downwelling_shortwave),
                land_absorbed_shortwave = array(state.interface_fluxes.land.downwelling_shortwave))

    return fields
end

# A mosaic whose sea surface temperature warms through the freezing point and whose sea ice retreats, so that
# the prescribed series, the clipping and the published albedos all change from step to step
function restart_model(arch; kw...)
    model = mosaic_model(arch; times = [0.0, 24.0], kw...)
    interior(model.ocean.sea_surface_temperature[2]) .= freezing_temperature + 20
    set!(model.ocean)
    return model
end

@testset "A run restored from a checkpoint continues as the uninterrupted run" begin
    for arch in test_architectures
        two_stream = (direct_surface_albedo = 0.2, diffuse_surface_albedo = 0.35)
        cases = ((optics = GrayOptics(), albedos = (; surface_albedo = 0.2)),
                 (optics = ClearSkyOptics(), albedos = two_stream))

        @testset "$(nameof(typeof(case.optics))) on $(typeof(arch))" for case in cases
            Δt = 1
            build() = restart_model(arch; optics = case.optics, albedos = case.albedos)
            prefix = "rtm_surface_restart_$(nameof(typeof(case.optics)))_$(typeof(arch))"
            rm.(filter(startswith(prefix), readdir()), force = true)

            # Uninterrupted: six steps
            uninterrupted = build()
            run!(Simulation(uninterrupted; Δt, stop_iteration = 6, verbose = false))
            reference = restart_fingerprint(uninterrupted)

            # Three steps, and a checkpoint
            first_leg = build()
            simulation = Simulation(first_leg; Δt, stop_iteration = 3, verbose = false)
            simulation.output_writers[:checkpointer] = Checkpointer(first_leg; schedule = IterationInterval(3), prefix)
            run!(simulation)
            halfway = restart_fingerprint(first_leg)
            in_memory_state = deepcopy(Oceananigans.prognostic_state(first_leg))

            # A new model, restored from the file, runs the last three steps
            restored = build()
            simulation = Simulation(restored; Δt, stop_iteration = 6, verbose = false)
            simulation.output_writers[:checkpointer] = Checkpointer(restored; schedule = IterationInterval(3), prefix)
            set!(simulation; checkpoint = :latest)
            @test restored.clock.iteration == 3
            @test restored.sea_ice.clock.iteration == 3

            # Restoring reproduces the state that was checkpointed
            at_restore = restart_fingerprint(restored)
            for name in keys(halfway)
                at_restore[name] == halfway[name] ||
                    @info "At the restore, $name differs by up to $(maximum(abs, at_restore[name] .- halfway[name]))"
                @test at_restore[name] == halfway[name]
            end

            run!(simulation)
            @test restored.clock.iteration == 6
            continued = restart_fingerprint(restored)
            for name in keys(reference)
                continued[name] == reference[name] ||
                    @info "After the restart, $name differs by up to $(maximum(abs, continued[name] .- reference[name]))"
                @test continued[name] == reference[name]
            end

            # The run evolves: the comparison is not between states that stand still
            @test reference.sea_ice_skin_temperature != halfway.sea_ice_skin_temperature
            @test reference.ocean_interface_temperature != halfway.ocean_interface_temperature
            @test reference.sea_ice_concentration != halfway.sea_ice_concentration
            @test reference.rtm_downwelling_longwave != halfway.rtm_downwelling_longwave

            # Sensitivity: without the skin temperature and the published albedos, which earlier checkpoints
            # did not hold, the continuation differs
            incomplete_state = merge(in_memory_state, (; radiation_exchanger = nothing,
                                                        sea_ice = (; clock = in_memory_state.sea_ice.clock)))
            incomplete = build()
            Oceananigans.restore_prognostic_state!(incomplete, incomplete_state)
            for _ in 1:3
                time_step!(incomplete, Δt)
            end
            @test restart_fingerprint(incomplete).sea_ice_skin_temperature != reference.sea_ice_skin_temperature

            rm.(filter(startswith(prefix), readdir()), force = true)
        end
    end
end

#####
##### Fractions in halo cells
#####

@testset "Fractions outside [0, 1] in halo cells do not reach the mosaic or the fluxes of the interior" begin
    for arch in test_architectures
        # What the radiation mosaic, the surface partition and the net fluxes of the atmosphere compute from the
        # fractions, in the interior. On this grid, bounded in both directions, the first faces are walls.
        function interior_results(model)
            interfaces = model.interfaces
            state = interfaces.exchanger.radiation.state
            net = interfaces.net_fluxes.atmosphere
            Nx, Ny, _ = size(interfaces.exchanger.grid)
            return (map(surface_values, (state.radiative_surface.temperature, state.radiative_surface.emissivity,
                                         state.radiative_surface.direct_albedo, state.radiative_surface.diffuse_albedo,
                                         interfaces.surface_partition.surface_temperature, state.surface_properties.land.albedo,
                                         state.interface_fluxes.ocean.downwelling_shortwave,
                                         state.interface_fluxes.sea_ice.upwelling_longwave,
                                         state.interface_fluxes.land.downwelling_longwave,
                                         interfaces.atmosphere_sea_ice_interface.temperature,
                                         net.ρE, net.ρqᵛᵉ))...,
                    surface_values(net.ρu)[2:Nx, :], surface_values(net.ρv)[:, 2:Ny])
        end

        # Overshoot every halo cell of a field, as a halo fill can
        function overshoot_halos!(field, value)
            kept = copy(interior(field))
            fill!(parent(field), value)
            interior(field) .= kept
            return nothing
        end

        control = equilibrate_surface!(mosaic_model(arch))
        model = equilibrate_surface!(mosaic_model(arch))
        @test interior_results(model) == interior_results(control)

        ℵᵒᶜ = model.interfaces.surface_partition.ocean_fraction
        overshoot_halos!(ℵᵒᶜ, 1.13)
        overshoot_halos!(model.sea_ice.concentration[1], 1.13)
        equilibrate_surface!(model; cycles = 1)
        equilibrate_surface!(control; cycles = 1)

        # The cells where the interface kernels and the net fluxes are computed: 0:N+1 in each direction
        grid = model.interfaces.exchanger.grid
        Nx, Ny, _ = size(grid)
        Hx, Hy, _ = Oceananigans.Grids.halo_size(grid)
        computed(field) = Array(parent(field))[Hx:Hx+Nx+1, Hy:Hy+Ny+1, 1]
        @test maximum(computed(ℵᵒᶜ)) == 1.13
        @test maximum(computed(control.interfaces.surface_partition.ocean_fraction)) ≤ 1

        # The sea ice concentration is clipped where it is read from the series, halo columns included
        @test maximum(surface_values(sea_ice_concentration(model.sea_ice))) ≤ 0.9
        @test maximum(computed(sea_ice_concentration(model.sea_ice))) == 1
        @test maximum(computed(sea_ice_concentration(control.sea_ice))) ≤ 0.9

        # The mosaic, the blended skin temperature, the radiative fluxes and the net fluxes in the interior are
        # those of the model whose halos hold fractions in [0, 1], bit for bit: they read the interior only
        results = interior_results(model)
        @test all(all(isfinite, result) for result in results)
        @test results == interior_results(control)
    end
end
