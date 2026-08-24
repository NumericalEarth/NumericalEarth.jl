include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using RRTMGP   # activates Breeze's radiative-transfer extension (RadiativeTransferModel)
using Dates: DateTime
using Test

using NumericalEarth.EarthSystemModels.InterfaceComputations: kernel_radiation_properties
using NumericalEarth.Lands: BucketHydrology, SlabEnergy, SlabLand

NumericalEarthBreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
@test !isnothing(NumericalEarthBreezeExt)

const rtm_albedo = 0.2
const rtm_emissivity = 0.95

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
    fill!(parent(land.water_storage), 45)

    radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                       solar_position = ApparentSolarPosition(coordinate = (0, 30),
                                                                             epoch = DateTime(2024, 3, 20, hour)),
                                       surface_albedo = rtm_albedo,
                                       surface_emissivity = rtm_emissivity,
                                       schedule = IterationInterval(1))

    return AtmosphereLandModel(atmosphere, land; radiation)
end

@testset "Breeze RadiativeTransferModel publishes its downwelling fluxes" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Interface radiation state on $A" begin
            model = build_rtm_land_model(arch)
            rtm = model.radiation

            exchanger = model.interfaces.exchanger.radiation
            @test !isnothing(exchanger)
            @test haskey(kernel_radiation_properties(rtm).surface_properties, :land)

            time_step!(model, 1.0)

            # Breeze stores downwelling negative; the interface wants positive-down magnitudes.
            ℐꜜˢʷ = Array(interior(exchanger.state.ℐꜜˢʷ))
            ℐꜜˡʷ = Array(interior(exchanger.state.ℐꜜˡʷ))
            @test all(ℐꜜˢʷ .> 0)
            @test all(ℐꜜˡʷ .> 0)
            @test ℐꜜˢʷ ≈ -Array(interior(rtm.downwelling_shortwave_flux))[:, :, 1]
            @test ℐꜜˡʷ ≈ -Array(interior(rtm.downwelling_longwave_flux))[:, :, 1]
        end

        @testset "No shortwave at night on $A" begin
            night = build_rtm_land_model(arch; hour = 0)
            time_step!(night, 1.0)

            @test all(Array(interior(night.interfaces.exchanger.radiation.state.ℐꜜˢʷ)) .== 0)
            @test all(Array(interior(night.interfaces.exchanger.radiation.state.ℐꜜˡʷ)) .> 0)
        end

        @testset "Bulk land is unchanged on $A" begin
            # Nothing on the land side reads the interface radiation state yet, so publishing it
            # must leave the existing budget alone: turbulent plus net upward radiative.
            model = build_rtm_land_model(arch)
            interface = model.interfaces.atmosphere_land_interface
            time_step!(model, 1.0)

            rtm = model.radiation
            σ = NumericalEarth.Radiations.default_stefan_boltzmann_constant
            Tₛ = Array(interior(interface.temperature))
            ℐˡʷꜜ = Array(interior(rtm.downwelling_longwave_flux))[:, :, 1]
            ℐˢʷꜜ = Array(interior(rtm.downwelling_shortwave_flux))[:, :, 1]
            ℐˡʷꜛ = rtm_emissivity .* σ .* Tₛ .^ 4 .- (1 - rtm_emissivity) .* ℐˡʷꜜ

            𝒬 = Array(interior(interface.fluxes.sensible_heat)) .+
                Array(interior(interface.fluxes.latent_heat))

            Jᴱs = Array(interior(model.land.fluxes.surface_energy_flux))
            @test Jᴱs ≈ 𝒬 .+ ℐˡʷꜛ .+ ℐˡʷꜜ .+ (1 - rtm_albedo) .* ℐˢʷꜜ
        end

    end
end
