include("runtests_setup.jl")

using Breeze
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using RRTMGP   # activates Breeze's RadiativeTransferModel extension

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyInterface, CanopyConductanceHumidity, CriticalSaturation, AbsorbedPhotosyntheticallyActiveRadiation,
    TiledLandInterface
using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

const rtm_albedo = 0.2
const rtm_emissivity = 0.95
const σ = default_stefan_boltzmann_constant

# Explicit acoustics on 125 m cells keep Δt under Δz / c ≈ 0.35 s.
const Δt = 0.25

# `hour` is the solar epoch at longitude 0, so 12 is local noon and 0 local midnight.
function rtm_canopy_components(arch; hour = 12, surface_storage = nothing)
    grid = RectilinearGrid(arch; size = (8, 8), halo = (5, 5), x = (-1kilometer, 1kilometer), z = (0, 1kilometer),
                           topology = (Periodic, Flat, Bounded))
    constants = ThermodynamicConstants(Float64)
    atmosphere = atmosphere_simulation(grid; thermodynamic_constants = constants, potential_temperature = 295)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.surface_potential_temperature, u = 2)

    solar_position = ApparentSolarPosition(coordinate = (0, 30), epoch = DateTime(2024, 3, 20, hour))
    radiation = RadiativeTransferModel(grid, GrayOptics(), constants; solar_position, surface_albedo = rtm_albedo,
                                       surface_emissivity = rtm_emissivity, schedule = IterationInterval(1))

    canopy = CanopyInterface(Float64; soil_humidity = bare_soil_humidity(Float64),
        canopy_humidity = CanopyConductanceHumidity(Float64; leaf_area_index = 4, moisture_stress = CriticalSaturation(0.5),
                                                    absorbed_par = AbsorbedPhotosyntheticallyActiveRadiation(Float64)))

    land_grid = RectilinearGrid(arch; size = grid.Nx, halo = grid.Hx, x = (-1kilometer, 1kilometer), topology = (Periodic, Flat, Flat))
    land = SlabLand(land_grid; surface_storage, hydrology = BucketHydrology(Float64; maximum_water_storage = 150), energy = SlabEnergy(Float64))
    set!(land; T = 298, M = 45)

    return atmosphere, land, radiation, canopy
end

function rtm_canopy_model(arch; hour = 12, surface_storage = nothing)
    atmosphere, land, radiation, canopy = rtm_canopy_components(arch; hour, surface_storage)
    return AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface_state=canopy)
end

@testset "Breeze RadiativeTransferModel driving a CanopyInterface" begin
    for arch in test_architectures, surface_storage in (nothing, PrognosticCanopyAir())
        A = typeof(arch)

        @testset "Canopy radiation budget under an RTM with $surface_storage on $A" begin
            model = rtm_canopy_model(arch; surface_storage)
            time_step!(model, Δt)
            canopy = model.interfaces.atmosphere_land_interface.temperature

            @test Array(interior(model.land.fluxes.surface_energy_flux)) ≈ -Array(interior(canopy.ground_heat_flux))
            # The effective albedo blends the leaf (0.15) and ground (0.2) albedos
            @test all(0.15 .< Array(interior(model.radiation.surface_radiation.direct_surface_albedo)) .< 0.2)
            @test all(Array(interior(canopy.canopy)) .> Array(interior(canopy.ground)))
            @test all(Array(interior(canopy.effective)) .> Array(interior(canopy.interface)))
            @test Array(interior(model.radiation.upwelling_longwave_flux))[:, :, 1] ≈ σ .* Array(interior(canopy.effective, :, :, 1)) .^ 4 rtol = 1e-3

            night = rtm_canopy_model(arch; hour = 0, surface_storage)
            time_step!(night, Δt)
            @test all(Array(interior(canopy.canopy)) .> Array(interior(night.interfaces.atmosphere_land_interface.temperature.canopy)))
        end

        @testset "Tiled interface keeps the bare tile's configured optics on $A" begin
            f = 0.6
            atmosphere, land, radiation, canopy = rtm_canopy_components(arch)
            tiled = TiledLandInterface(land.grid, atmosphere, land; vegetated = canopy, cover_fraction = f)
            model = AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = tiled)
            update_state!(model)

            mosaic = model.interfaces.atmosphere_land_interface.temperature
            @test all(Array(interior(mosaic.effective_albedo)) .≈ f * 0.15 + (1 - f) * rtm_albedo)

            Tᵛ = Array(interior(tiled.vegetated.temperature.effective))
            Tᵇ = Array(interior(tiled.bare.temperature))
            ℐꜜˡʷ = Array(interior(model.interfaces.exchanger.radiation.state.ℐꜜˡʷ))
            bare_longwave = @. rtm_emissivity * σ * Tᵇ^4 - (1 - rtm_emissivity) * ℐꜜˡʷ
            @test σ .* Array(interior(mosaic.effective)) .^ 4 ≈ @. f * σ * Tᵛ^4 + (1 - f) * bare_longwave
        end
    end
end
