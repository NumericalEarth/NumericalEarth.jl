include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!, update_state!
using RRTMGP   # activates Breeze's radiative-transfer extension (RadiativeTransferModel)
using Dates: DateTime
using Test

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyGroundSurface, CanopyConductanceHumidity, CriticalSaturation, InteractiveAbsorbedPAR, TiledLandInterface
using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

const rtm_albedo = 0.2
const rtm_emissivity = 0.95

# The atmosphere steps its acoustics explicitly on 125 m cells, so Δt stays under Δz / c ≈ 0.35 s.
const Δt = 0.25

# `hour` is the solar epoch at longitude 0, so 12 is local noon and 0 local midnight. Gray
# optics needs no lookup tables.
function rtm_canopy_components(arch; hour = 12, cover_fraction = nothing)
    FT = Float64

    grid = RectilinearGrid(arch, FT; size = (8, 8), halo = (5, 5),
                           x = (-1kilometer, 1kilometer), z = (0, 1kilometer),
                           topology = (Periodic, Flat, Bounded))

    constants = ThermodynamicConstants(FT)
    atmosphere = atmosphere_simulation(grid; thermodynamic_constants = constants,
                                      potential_temperature = 295)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.surface_potential_temperature, u = 2)

    land_grid = RectilinearGrid(arch, FT; size = grid.Nx, halo = grid.Hx,
                               x = (-1kilometer, 1kilometer),
                               topology = (Periodic, Flat, Flat))

    radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                       solar_position = ApparentSolarPosition(coordinate = (0, 30),
                                                                             epoch = DateTime(2024, 3, 20, hour)),
                                       surface_albedo = rtm_albedo,
                                       surface_emissivity = rtm_emissivity,
                                       schedule = IterationInterval(1))

    canopy = CanopyGroundSurface(FT;
        soil_humidity = bare_soil_humidity(FT),
        canopy_humidity = CanopyConductanceHumidity(FT; leaf_area_index = 4,
                                           moisture_stress = CriticalSaturation(0.5),
                                           absorbed_par = InteractiveAbsorbedPAR(FT)))

    surface = LandSurface(canopy)
    isnothing(cover_fraction) || (surface = TiledLandSurface(surface; cover_fraction))
    land = SlabLand(land_grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150),
                    energy = SlabEnergy(FT), surface)
    set!(land; T = 298, M = 45)

    return atmosphere, land, radiation, canopy
end

function build_rtm_canopy_model(arch; hour = 12)
    atmosphere, land, radiation, canopy = rtm_canopy_components(arch; hour)
    return AtmosphereLandModel(atmosphere, land; radiation)
end

@testset "Breeze RadiativeTransferModel driving a CanopyGroundSurface" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Canopy radiation budget under an RTM on $A" begin
            model = build_rtm_canopy_model(arch)
            rtm = model.radiation
            canopy = model.interfaces.atmosphere_land_interface.temperature

            time_step!(model, Δt)

            # The canopy absorbs radiation in its own solve, so the slab receives only the skin conduction.
            @test Array(interior(model.land.fluxes.surface_energy_flux)) ≈ -Array(interior(canopy.ground_heat_flux))

            # Leaf and ground albedo default to 0.15, so the column albedo the RTM reflects is 0.15 at any leaf area.
            @test all(Array(interior(rtm.surface_radiation.direct_surface_albedo)) .≈ 0.15)

            # Sunlit leaves over shaded soil, radiating warmer than the canopy air.
            @test all(Array(interior(canopy.canopy)) .> Array(interior(canopy.ground)))
            @test all(Array(interior(canopy.effective)) .> Array(interior(canopy.interface)))

            σ = default_stefan_boltzmann_constant
            Tᵉᶠᶠ = Array(interior(canopy.effective, :, :, 1))
            @test Array(interior(rtm.upwelling_longwave_flux))[:, :, 1] ≈ σ .* Tᵉᶠᶠ .^ 4 rtol = 1e-3

            # The same column at midnight has cooler leaves.
            night = build_rtm_canopy_model(arch; hour = 0)
            time_step!(night, Δt)
            @test all(Array(interior(canopy.canopy)) .> Array(interior(night.interfaces.atmosphere_land_interface.temperature.canopy)))
        end

        @testset "Tiled interface keeps the bare tile's configured optics on $A" begin
            f = 0.6
            atmosphere, land, radiation, canopy = rtm_canopy_components(arch; cover_fraction = f)
            tiled = TiledLandInterface(land.grid, atmosphere, land)
            model = AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = tiled)
            update_state!(model)

            # The vegetated column reflects 0.15; the bare tile reflects and emits with the RTM's
            # configured albedo and emissivity, so the mosaic is their area weight.
            mosaic = model.interfaces.atmosphere_land_interface.temperature
            @test all(Array(interior(mosaic.effective_albedo)) .≈ f * 0.15 + (1 - f) * rtm_albedo)

            σ = default_stefan_boltzmann_constant
            Tᵛ = Array(interior(tiled.vegetated.temperature.effective))
            Tᵇ = Array(interior(tiled.bare.temperature))
            ℐꜜˡʷ = Array(interior(model.interfaces.exchanger.radiation.state.ℐꜜˡʷ))
            bare_longwave = @. rtm_emissivity * σ * Tᵇ^4 - (1 - rtm_emissivity) * ℐꜜˡʷ
            @test σ .* Array(interior(mosaic.effective)) .^ 4 ≈ @. f * σ * Tᵛ^4 + (1 - f) * bare_longwave
        end
    end
end
