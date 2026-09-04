include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using RRTMGP   # activates Breeze's radiative-transfer extension (RadiativeTransferModel)
using Dates: DateTime
using Test

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, ConstantTortuosity, CriticalSaturation,
    DryLayerHumidity, DryLayerVaporPistonVelocity, InteractiveAbsorbedPAR, StorageBasedDryLayerDepth
using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

const rtm_albedo = 0.2
const rtm_emissivity = 0.95

# `hour` is the solar epoch at longitude 0, so 12 is local noon and 0 local midnight. Gray
# optics needs no lookup tables.
function build_rtm_canopy_model(arch; hour = 12)
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

    land = SlabLand(land_grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150),
                    energy = SlabEnergy(FT))
    set!(land; T = 298)
    fill!(parent(land.water_storage), 45)   # 𝒮 = 0.3

    radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                       solar_position = ApparentSolarPosition(coordinate = (0, 30),
                                                                             epoch = DateTime(2024, 3, 20, hour)),
                                       surface_albedo = rtm_albedo,
                                       surface_emissivity = rtm_emissivity,
                                       schedule = IterationInterval(1))

    canopy = CanopyAirSpace(FT;
        soil = DryLayerHumidity(FT;
            dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                        dry_layer_onset_saturation = 0.5,
                                                        dry_layer_exponent = 2),
            vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                         molecular_diffusivity = 2.4e-5,
                                                         tortuosity = ConstantTortuosity()),
            thermal_exchange_depth = 0.05, porosity = 0.4),
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = 4,
                                           moisture_stress = CriticalSaturation(0.5),
                                           absorbed_par = InteractiveAbsorbedPAR(FT)))

    return AtmosphereLandModel(atmosphere, land; radiation,
                               atmosphere_land_interface_temperature = canopy,
                               atmosphere_land_interface_specific_humidity = canopy)
end

@testset "Breeze RadiativeTransferModel driving a CanopyAirSpace" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Canopy radiation budget under an RTM on $A" begin
            model = build_rtm_canopy_model(arch)
            rtm = model.radiation
            canopy = model.interfaces.atmosphere_land_interface.temperature

            time_step!(model, 1.0)

            # The canopy absorbs radiation in its own solve, so the slab receives only the skin conduction.
            @test Array(interior(model.land.fluxes.surface_energy_flux)) ≈ -Array(interior(canopy.ground_heat_flux))

            # The RTM reflects with the canopy's albedo rather than its configured one: leaf and
            # ground albedo are both 0.15 by default, so the column's is too at any leaf area.
            @test all(Array(interior(rtm.surface_properties.direct_surface_albedo)) .≈ 0.15)

            # Sunlit leaves over shaded soil, radiating warmer than the canopy air.
            Tᵃᶜ = Array(interior(canopy.interface))
            @test all(Array(interior(canopy.canopy)) .> Array(interior(canopy.soil_skin)))
            @test all(Array(interior(canopy.effective)) .> Tᵃᶜ)

            # RRTMGP solves at the start of a step from the temperature bound at the end of the
            # previous one, so its surface emission reproduces the canopy's upwelling longwave.
            σ = default_stefan_boltzmann_constant
            Tᵉᶠᶠ = Array(interior(canopy.effective))
            time_step!(model, 1.0)
            @test Array(interior(rtm.upwelling_longwave_flux))[:, :, 1] ≈ σ .* Tᵉᶠᶠ .^ 4 rtol = 1e-3
        end

        @testset "Canopy cools at night on $A" begin
            night = build_rtm_canopy_model(arch; hour = 0)
            time_step!(night, 1.0)

            day = build_rtm_canopy_model(arch)
            time_step!(day, 1.0)

            Tˡᵉᵃᶠ(model) = Array(interior(model.interfaces.atmosphere_land_interface.temperature.canopy))
            @test all(Tˡᵉᵃᶠ(day) .> Tˡᵉᵃᶠ(night))
        end
    end
end
