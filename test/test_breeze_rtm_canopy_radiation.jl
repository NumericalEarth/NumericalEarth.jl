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
    SoilConductiveFlux, StorageBasedDryLayerDepth, kernel_radiation_properties
using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

NumericalEarthBreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
@test !isnothing(NumericalEarthBreezeExt)

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

# Tiny coupled Breeze + gray-optics RTM + SlabLand for CI. `hour` sets the solar epoch at
# longitude 0, so `hour = 12` is local noon and `hour = 0` local midnight. Gray optics needs
# no lookup tables, and the RTM is built without a `surface_temperature` so the coupler binds
# the interface's radiating temperature.
function build_rtm_land_model(arch; hour = 12, canopy = true)
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

    canopy || return AtmosphereLandModel(atmosphere, land; radiation)

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

            # The RTM radiates from the canopy's two-source blend, not the canopy-air node.
            @test rtm.surface_properties.surface_temperature === interface.temperature.effective

            # The RTM publishes the land's radiative properties and an exchanger state, so the
            # canopy solve reads the same contract it reads under a `PrescribedRadiation`.
            exchanger = model.interfaces.exchanger.radiation
            @test !isnothing(exchanger)
            @test haskey(kernel_radiation_properties(rtm).surface_properties, :land)

            time_step!(model, 1.0)

            # Downwelling fluxes reach the interface as positive-down magnitudes (Breeze stores
            # them negative, positive-up).
            ℐꜜˢʷ = Array(interior(exchanger.state.ℐꜜˢʷ))
            ℐꜜˡʷ = Array(interior(exchanger.state.ℐꜜˡʷ))
            @test all(ℐꜜˢʷ .> 0)
            @test all(ℐꜜˡʷ .> 0)
            @test ℐꜜˢʷ ≈ -Array(interior(rtm.downwelling_shortwave_flux))[:, :, 1]
            @test ℐꜜˡʷ ≈ -Array(interior(rtm.downwelling_longwave_flux))[:, :, 1]

            # Radiation is internalized in the canopy solve, so the slab is driven by the
            # skin→bulk conduction alone — no radiative flux added on top.
            Jᴱs = Array(interior(model.land.fluxes.surface_energy_flux))
            Gᶜ = Array(interior(interface.temperature.ground_heat_flux))
            @test Jᴱs ≈ -Gᶜ

            # Sunlit leaves run warmer than the shaded ground, and the radiating temperature
            # separates from the canopy-air node — with a zero radiation state the two skins and
            # `Teff` all collapse onto the node.
            Tᵃᶜ = Array(interior(interface.temperature.interface))
            Tᵛ = Array(interior(interface.temperature.canopy))
            Tᵍ = Array(interior(interface.temperature.soil_skin))
            Teff = Array(interior(interface.temperature.effective))
            @test all(Tᵛ .> Tᵍ)
            @test all(Teff .> Tᵃᶜ)
        end

        @testset "No shortwave at night on $A" begin
            night = build_rtm_land_model(arch; hour = 0)
            time_step!(night, 1.0)

            @test all(Array(interior(night.interfaces.exchanger.radiation.state.ℐꜜˢʷ)) .== 0)
            @test all(Array(interior(night.interfaces.exchanger.radiation.state.ℐꜜˡʷ)) .> 0)

            # A nocturnal canopy cools below the daytime one — the solve responds to the
            # shortwave it now receives.
            day = build_rtm_land_model(arch)
            time_step!(day, 1.0)
            Tᵛ_day = Array(interior(day.interfaces.atmosphere_land_interface.temperature.canopy))
            Tᵛ_night = Array(interior(night.interfaces.atmosphere_land_interface.temperature.canopy))
            @test all(Tᵛ_day .> Tᵛ_night)
        end

        @testset "Bulk interface keeps the radiative flux add on $A" begin
            model = build_rtm_land_model(arch; canopy = false)
            interface = model.interfaces.atmosphere_land_interface

            # No canopy: the atmosphere-facing temperature is also the radiating one.
            @test model.radiation.surface_properties.surface_temperature === interface.temperature

            time_step!(model, 1.0)

            rtm = model.radiation
            σ = default_stefan_boltzmann_constant
            ε = rtm_emissivity
            α = rtm_albedo

            Tₛ = Array(interior(interface.temperature))
            ℐˡʷꜜ = Array(interior(rtm.downwelling_longwave_flux))[:, :, 1]
            ℐˢʷꜜ = Array(interior(rtm.downwelling_shortwave_flux))[:, :, 1]
            ℐˡʷꜛ = ε .* σ .* Tₛ .^ 4 .- (1 - ε) .* ℐˡʷꜜ

            𝒬 = Array(interior(interface.fluxes.sensible_heat)) .+
                Array(interior(interface.fluxes.latent_heat))

            # `surface_energy_flux` is positive upward: turbulent plus net upward radiative.
            Jᴱs = Array(interior(model.land.fluxes.surface_energy_flux))
            @test Jᴱs ≈ 𝒬 .+ ℐˡʷꜛ .+ ℐˡʷꜜ .+ (1 - α) .* ℐˢʷꜜ
        end

        @testset "TiledLandInterface radiating temperature on $A" begin
            model = build_rtm_land_model(arch)
            cas = build_canopy_air_space(Float64)
            tiled = TiledLandInterface(model.land.grid, model.atmosphere, model.land;
                                      vegetated = cas, fraction = 0.5)

            # The mosaic radiates from its radiance-weighted blend, and the same
            # `CanopyAirSpaceDiagnostics` signal keeps `apply_air_land_radiative_fluxes!` a no-op.
            @test radiating_temperature(tiled) === tiled.temperature.effective
            @test surface_temperature(tiled) === tiled.temperature.interface
        end
    end
end
