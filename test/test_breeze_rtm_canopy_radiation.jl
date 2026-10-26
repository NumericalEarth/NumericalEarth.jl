include("runtests_setup.jl")

using Breeze
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using RRTMGP   # activates Breeze's RadiativeTransferModel extension

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyGroundSurface, CanopyConductanceHumidity, CriticalSaturation, InteractiveAbsorbedPAR
using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

const rtm_albedo = 0.2
const rtm_emissivity = 0.95
const σ = default_stefan_boltzmann_constant

# Explicit acoustics on 125 m cells keep Δt under Δz / c ≈ 0.35 s.
const Δt = 0.25

# `hour` is the solar epoch at longitude 0, so 12 is local noon and 0 local midnight.
function rtm_canopy_components(arch; hour = 12)
    grid = RectilinearGrid(arch; size = (8, 8), halo = (5, 5), x = (-1kilometer, 1kilometer), z = (0, 1kilometer),
                           topology = (Periodic, Flat, Bounded))
    constants = ThermodynamicConstants(Float64)
    atmosphere = atmosphere_simulation(grid; thermodynamic_constants = constants, potential_temperature = 295)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.surface_potential_temperature, u = 2)

    solar_position = ApparentSolarPosition(coordinate = (0, 30), epoch = DateTime(2024, 3, 20, hour))
    radiation = RadiativeTransferModel(grid, GrayOptics(), constants; solar_position, surface_albedo = rtm_albedo,
                                       surface_emissivity = rtm_emissivity, schedule = IterationInterval(1))

    canopy = CanopyGroundSurface(Float64; soil_humidity = bare_soil_humidity(Float64),
        canopy_humidity = CanopyConductanceHumidity(Float64; leaf_area_index = 4, moisture_stress = CriticalSaturation(0.5),
                                                    absorbed_par = InteractiveAbsorbedPAR(Float64)))
    surface = LandSurface(canopy)

    land_grid = RectilinearGrid(arch; size = grid.Nx, halo = grid.Hx, x = (-1kilometer, 1kilometer), topology = (Periodic, Flat, Flat))
    land = SlabLand(land_grid; hydrology = BucketHydrology(Float64; maximum_water_storage = 150), energy = SlabEnergy(Float64), surface)
    set!(land; T = 298, M = 45)

    return atmosphere, land, radiation
end

function rtm_canopy_model(arch; hour = 12)
    atmosphere, land, radiation = rtm_canopy_components(arch; hour)
    return AtmosphereLandModel(atmosphere, land; radiation)
end

@testset "Breeze RadiativeTransferModel driving a CanopyGroundSurface" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Canopy radiation budget under an RTM on $A" begin
            model = rtm_canopy_model(arch)
            time_step!(model, Δt)
            canopy = model.interfaces.atmosphere_land_interface.temperature

            @test Array(interior(model.land.fluxes.surface_energy_flux)) ≈ -Array(interior(canopy.ground_heat_flux))
            # Leaf and ground albedo both default to 0.15
            @test all(Array(interior(model.radiation.surface_radiation.direct_surface_albedo)) .≈ 0.15)
            @test all(Array(interior(canopy.canopy)) .> Array(interior(canopy.ground)))
            @test all(Array(interior(canopy.effective)) .> Array(interior(canopy.interface)))
            @test Array(interior(model.radiation.upwelling_longwave_flux))[:, :, 1] ≈ σ .* Array(interior(canopy.effective, :, :, 1)) .^ 4 rtol = 1e-3

            night = rtm_canopy_model(arch; hour = 0)
            time_step!(night, Δt)
            @test all(Array(interior(canopy.canopy)) .> Array(interior(night.interfaces.atmosphere_land_interface.temperature.canopy)))
        end
    end
end
