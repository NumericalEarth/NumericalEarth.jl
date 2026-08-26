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
    DryLayerHumidity, DryLayerVaporPistonVelocity, InteractiveAbsorbedPAR,
    SoilConductiveFlux, StorageBasedDryLayerDepth
using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

const rtm_albedo = 0.2
const rtm_emissivity = 0.95

build_canopy_air_space(FT) = CanopyAirSpace(FT;
    soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                    dry_layer_onset_saturation = 0.5,
                                                    dry_layer_exponent = 2),
        vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                     molecular_diffusivity = 2.4e-5,
                                                     tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.05, porosity = 0.4),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 4.0,
                                       moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05))

# `hour` is the solar epoch at longitude 0, so 12 is local noon and 0 local midnight. Gray
# optics needs no lookup tables.
function build_rtm_land_model(arch; hour = 12)
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

    cas = build_canopy_air_space(FT)
    return AtmosphereLandModel(atmosphere, land; radiation,
                               atmosphere_land_interface_temperature = cas,
                               atmosphere_land_interface_specific_humidity = cas)
end

@testset "Breeze RadiativeTransferModel driving the land interface" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "CanopyAirSpace under an RTM on $A" begin
            model = build_rtm_land_model(arch)
            rtm = model.radiation
            interface = model.interfaces.atmosphere_land_interface

            time_step!(model, 1.0)

            # Radiation is internalized in the canopy solve, so no flux is added on top.
            Jᴱs = Array(interior(model.land.fluxes.surface_energy_flux))
            Gᶜ = Array(interior(interface.temperature.ground_heat_flux))
            @test Jᴱs ≈ -Gᶜ

            Hᵛ = Array(interior(interface.temperature.canopy_sensible_heat))
            Hᵍ = Array(interior(interface.temperature.soil_sensible_heat))
            LEᵛ = Array(interior(interface.temperature.canopy_latent_heat))
            LEᵍ = Array(interior(interface.temperature.soil_latent_heat))
            @test Hᵛ .+ Hᵍ ≈ Array(interior(interface.fluxes.sensible_heat))
            @test LEᵛ .+ LEᵍ ≈ Array(interior(interface.fluxes.latent_heat))

            # The canopy's two-source albedo reaches the shortwave solver's boundary condition.
            αᶜ = Array(interior(interface.temperature.effective_albedo))
            @test all(0 .< αᶜ .< 1)
            @test vec(Array(rtm.shortwave_solver.bcs.sfc_alb_direct)) ≈ vec(αᶜ)

            # RRTMGP solves at the start of a step from the temperature bound at the end of the
            # previous one, so its emitted surface longwave reproduces the canopy's upwelling.
            σ = default_stefan_boltzmann_constant
            Tᵉᶠᶠ = Array(interior(interface.temperature.effective))
            time_step!(model, 1.0)
            @test Array(interior(rtm.upwelling_longwave_flux))[:, :, 1] ≈ σ .* Tᵉᶠᶠ .^ 4 rtol = 1e-3

            # Sunlit canopy over shaded soil, both above the canopy-air node.
            Tᵃᶜ = Array(interior(interface.temperature.interface))
            @test all(Array(interior(interface.temperature.canopy)) .>
                      Array(interior(interface.temperature.soil_skin)))
            @test all(Array(interior(interface.temperature.effective)) .> Tᵃᶜ)
        end

        @testset "Canopy cools at night on $A" begin
            night = build_rtm_land_model(arch; hour = 0)
            time_step!(night, 1.0)

            day = build_rtm_land_model(arch)
            time_step!(day, 1.0)
            Tᵛ_day = Array(interior(day.interfaces.atmosphere_land_interface.temperature.canopy))
            Tᵛ_night = Array(interior(night.interfaces.atmosphere_land_interface.temperature.canopy))
            @test all(Tᵛ_day .> Tᵛ_night)
        end
    end
end
