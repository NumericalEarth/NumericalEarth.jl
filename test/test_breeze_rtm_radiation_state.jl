include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using RRTMGP   # activates Breeze's radiative-transfer extension (RadiativeTransferModel)
using Dates: DateTime
using Test

using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand

const rtm_Nx = 8
const rtm_Hx = 5
const rtm_albedo = 0.2
const rtm_emissivity = 0.95

# `hour` is the solar epoch at longitude 0, so 12 is local noon and 0 local midnight.
function build_rtm_land_model(arch; hour = 12, optics = GrayOptics(),
                              albedos = (; surface_albedo = rtm_albedo), surface_temperature = nothing)
    FT = Float64

    grid = RectilinearGrid(arch, FT; size = (rtm_Nx, rtm_Nx), halo = (rtm_Hx, rtm_Hx),
                           x = (-1kilometer, 1kilometer), z = (0, 1kilometer),
                           topology = (Periodic, Flat, Bounded))

    constants = ThermodynamicConstants(FT)
    atmosphere = atmosphere_simulation(grid; thermodynamic_constants = constants,
                                      potential_temperature = 295)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.surface_potential_temperature, u = 2)

    land_grid = RectilinearGrid(arch, FT; size = grid.Nx, halo = grid.Hx,
                               x = (-1kilometer, 1kilometer),
                               topology = (Periodic, Flat, Flat))

    land = SlabLand(land_grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150),
                    energy = SlabEnergy(FT))
    set!(land; T = 298)
    fill!(parent(land.water_storage), 45)

    radiation = RadiativeTransferModel(grid, optics, constants;
                                       solar_position = ApparentSolarPosition(coordinate = (0, 30),
                                                                             epoch = DateTime(2024, 3, 20, hour)),
                                       albedos...,
                                       surface_emissivity = rtm_emissivity,
                                       surface_temperature,
                                       schedule = IterationInterval(1))

    return AtmosphereLandModel(atmosphere, land; radiation)
end

@testset "Breeze RadiativeTransferModel publishes its downwelling fluxes" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Initial radiation and first land step via $advance on $A" for advance in (:time_step!, :run!)
            # The RTM radiates at the land's 298 K, not the configured 310 K.
            model = build_rtm_land_model(arch; surface_temperature = 310)
            radiation = model.interfaces.exchanger.radiation.state
            land = model.land
            initial_temperature = Array(interior(land.temperature))
            initial_energy_flux = Array(interior(land.fluxes.surface_energy_flux))

            @test all(Array(interior(radiation.ℐꜜˢʷ)) .< 0)
            @test all(Array(interior(radiation.ℐꜜˡʷ)) .< 0)
            @test all(initial_energy_flux .< 0)

            σ = NumericalEarth.Radiations.default_stefan_boltzmann_constant
            expected_longwave = rtm_emissivity .* σ .* initial_temperature .^ 4 .-
                                (1 - rtm_emissivity) .* Array(interior(radiation.ℐꜜˡʷ))
            @test Array(interior(model.radiation.upwelling_longwave_flux))[:, :, 1] ≈ expected_longwave

            Δt = 1.0
            if advance == :time_step!
                time_step!(model, Δt)
            else
                simulation = Simulation(model; Δt, stop_iteration = 1, verbose = false)
                run!(simulation)
            end

            temperature = Array(interior(land.temperature))
            water_storage = Array(interior(land.water_storage))
            heat_capacity = land.energy.dry_heat_capacity .+ land.energy.liquid_heat_capacity .* water_storage

            @test all(temperature .> initial_temperature)
            @test heat_capacity .* (temperature .- initial_temperature) ≈ -Δt .* initial_energy_flux rtol=1e-7
        end

        @testset "Interface radiation state and bulk surface energy budget on $A" begin
            model = build_rtm_land_model(arch)
            rtm = model.radiation
            exchanger = model.interfaces.exchanger.radiation
            interface = model.interfaces.atmosphere_land_interface

            # A step's land fluxes are computed from the radiation published before it.
            ℐꜜˡʷ = Array(interior(rtm.downwelling_longwave_flux))[:, :, 1]
            ℐꜜˢʷ = Array(interior(rtm.downwelling_shortwave_flux))[:, :, 1]
            time_step!(model, 1.0)

            @test Array(interior(exchanger.state.ℐꜜˢʷ)) ≈ Array(interior(rtm.downwelling_shortwave_flux))[:, :, 1]
            @test Array(interior(exchanger.state.ℐꜜˡʷ)) ≈ Array(interior(rtm.downwelling_longwave_flux))[:, :, 1]

            σ = NumericalEarth.Radiations.default_stefan_boltzmann_constant
            Tₛ = Array(interior(interface.temperature))
            ℐꜛˡʷ = rtm_emissivity .* σ .* Tₛ .^ 4 .- (1 - rtm_emissivity) .* ℐꜜˡʷ

            𝒬 = Array(interior(interface.fluxes.sensible_heat)) .+
                Array(interior(interface.fluxes.latent_heat))

            # Positive upward: turbulent plus net upward radiative.
            Jᴱs = Array(interior(model.land.fluxes.surface_energy_flux))
            @test Jᴱs ≈ 𝒬 .+ ℐꜛˡʷ .+ ℐꜜˡʷ .+ (1 - rtm_albedo) .* ℐꜜˢʷ
        end

        @testset "Two-stream surface shortwave with distinct direct and diffuse albedos on $A" begin
            model = build_rtm_land_model(arch; optics = ClearSkyOptics(),
                                         albedos = (direct_surface_albedo = 0.2, diffuse_surface_albedo = 0.6))
            rtm = model.radiation
            interface = model.interfaces.atmosphere_land_interface

            ℐꜜˡʷ = Array(interior(rtm.downwelling_longwave_flux))[:, :, 1]
            ℐꜜˢʷ = Array(interior(rtm.downwelling_shortwave_flux))[:, :, 1]
            ℐꜛˢʷ = Array(interior(rtm.upwelling_shortwave_flux))[:, :, 1]
            time_step!(model, 1.0)

            σ = NumericalEarth.Radiations.default_stefan_boltzmann_constant
            Tₛ = Array(interior(interface.temperature))

            𝒬 = Array(interior(interface.fluxes.sensible_heat)) .+
                Array(interior(interface.fluxes.latent_heat))

            # The land absorbs the RTM's net surface shortwave.
            Jᴱs = Array(interior(model.land.fluxes.surface_energy_flux))
            @test Jᴱs ≈ 𝒬 .+ rtm_emissivity .* σ .* Tₛ .^ 4 .+ rtm_emissivity .* ℐꜜˡʷ .+ ℐꜜˢʷ .+ ℐꜛˢʷ
        end

        @testset "No shortwave at night on $A" begin
            night = build_rtm_land_model(arch; hour = 0)
            time_step!(night, 1.0)

            @test all(Array(interior(night.interfaces.exchanger.radiation.state.ℐꜜˢʷ)) .== 0)
            @test all(Array(interior(night.interfaces.exchanger.radiation.state.ℐꜜˡʷ)) .< 0)
        end
    end
end
