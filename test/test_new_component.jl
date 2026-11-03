include("runtests_setup.jl")

using ConservativeRegridding
using ClimaSeaIce.SeaIceThermodynamics: LinearLiquidus
using NumericalEarth
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using Oceananigans
using Oceananigans.Fields: ConstantField, ZeroField, interior
using Oceananigans.TimeSteppers: TimeSteppers, Clock, time_step!
using Oceananigans.Units
using Test

using NumericalEarth.EarthSystemModels: EarthSystemModels
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations, regrid_surface_field!

# One model per role, written from the contract in `docs/src/developers/coupling_a_new_component.md` alone

struct MixedLayerOcean{G, T, F}
    grid :: G
    temperature :: T
    net_fluxes :: F
end

function MixedLayerOcean(grid)
    net_fluxes = (; u = CenterField(grid),
                    v = CenterField(grid),
                    T = CenterField(grid),
                    S = CenterField(grid),
                    η = CenterField(grid),
                    freshwater_heat_content = CenterField(grid),
                    u_coefficient = nothing,
                    v_coefficient = nothing)

    return MixedLayerOcean(grid, CenterField(grid), net_fluxes)
end

function TimeSteppers.time_step!(ocean::MixedLayerOcean, Δt)
    parent(ocean.temperature) .-= parent(ocean.net_fluxes.T) .* Δt ./ ocean.grid.Lz
    return nothing
end

Oceananigans.Grids.grid(ocean::MixedLayerOcean) = ocean.grid
EarthSystemModels.surface_state(ocean::MixedLayerOcean) =(; u = ZeroField(Float64), v = ZeroField(Float64), T = ocean.temperature, S = ConstantField(35.0))
InterfaceComputations.net_fluxes(ocean::MixedLayerOcean) = ocean.net_fluxes
EarthSystemModels.reference_density(::MixedLayerOcean) = 1025.0
EarthSystemModels.heat_capacity(::MixedLayerOcean) = 4000.0

struct StaticSeaIce{G, C, S, F, M}
    grid :: G
    clock :: C
    Δt :: Float64
    state :: S
    net_fluxes :: F
    mass_fluxes :: M
end

function StaticSeaIce(grid; Δt)
    F = Field{Center, Center, Nothing}
    state = (; u = ZeroField(Float64), v = ZeroField(Float64), hi = F(grid), hc = ConstantField(0.1), ℵ = F(grid),
               hs = ZeroField(Float64), S = ConstantField(5.0), T = F(grid))
    net_fluxes = (; top = (; heat = F(grid), snowfall = F(grid), u = F(grid), v = F(grid)), bottom = (; heat = F(grid)))
    mass_fluxes = (; ice = F(grid), snow = F(grid))
    return StaticSeaIce(grid, Clock(grid), Δt, state, net_fluxes, mass_fluxes)
end

TimeSteppers.time_step!(::StaticSeaIce, Δt) = nothing

Oceananigans.Grids.grid(sea_ice::StaticSeaIce) = sea_ice.grid
EarthSystemModels.surface_state(sea_ice::StaticSeaIce) = sea_ice.state
InterfaceComputations.net_fluxes(sea_ice::StaticSeaIce) = sea_ice.net_fluxes
EarthSystemModels.reference_density(::StaticSeaIce) = 900.0
EarthSystemModels.heat_capacity(::StaticSeaIce) = 2100.0
EarthSystemModels.sea_ice_phase_transitions(::StaticSeaIce) = (; liquidus = LinearLiquidus(Float64), reference_latent_heat = 334e3)
EarthSystemModels.sea_ice_mass_fluxes(sea_ice::StaticSeaIce) = sea_ice.mass_fluxes

struct BucketLand{G, T, S, F}
    grid :: G
    temperature :: T
    saturation :: S
    fluxes :: F
end

BucketLand(grid) = BucketLand(grid, CenterField(grid), CenterField(grid), (; surface_energy_flux = CenterField(grid)))

function TimeSteppers.time_step!(land::BucketLand, Δt)
    heat_capacity_per_area = 1e6
    parent(land.temperature) .-= parent(land.fluxes.surface_energy_flux) .* Δt ./ heat_capacity_per_area
    return nothing
end

Oceananigans.Grids.grid(land::BucketLand) = land.grid
EarthSystemModels.surface_state(land::BucketLand; kw...) =(; T = land.temperature, 𝒮 = land.saturation)

function EarthSystemModels.update_net_fluxes!(coupled_model, land::BucketLand)
    fluxes = coupled_model.interfaces.atmosphere_land_interface.fluxes
    interior(land.fluxes.surface_energy_flux) .= interior(fluxes.sensible_heat) .+ interior(fluxes.latent_heat)
    return nothing
end

struct BoxAtmosphere{G, S, Q}
    grid :: G
    state :: S
    heat_flux :: Q
end

function BoxAtmosphere(grid)
    F = Field{Center, Center, Nothing}
    state = (; u = F(grid), v = F(grid), T = F(grid), p = F(grid), q = F(grid), ℐꜜˢʷ = F(grid), ℐꜜˡʷ = F(grid),
               Jʳⁿ = F(grid), Jˢⁿ = F(grid))
    return BoxAtmosphere(grid, state, F(grid))
end

TimeSteppers.time_step!(::BoxAtmosphere, Δt) = nothing

Oceananigans.Grids.grid(atmosphere::BoxAtmosphere) = atmosphere.grid
EarthSystemModels.surface_state(atmosphere::BoxAtmosphere) = atmosphere.state
InterfaceComputations.net_fluxes(atmosphere::BoxAtmosphere) = (; atmosphere.heat_flux)
EarthSystemModels.thermodynamics_parameters(::BoxAtmosphere) = AtmosphereThermodynamicsParameters(Float64)
EarthSystemModels.surface_layer_height(::BoxAtmosphere) = 10.0
EarthSystemModels.boundary_layer_height(::BoxAtmosphere) = 600.0

function EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::BoxAtmosphere)
    regridder = coupled_model.interfaces.exchanger.atmosphere.regridder
    surface_fluxes = coupled_model.interfaces.surface_partition.fluxes
    regrid_surface_field!(atmosphere.heat_flux, regridder.exchange.to_component, surface_fluxes.sensible_heat)
    return nothing
end

surface_atmosphere_grid(size) = LatitudeLongitudeGrid(; size, longitude = (0, 360), latitude = (-40, 40),
                                                        topology = (Periodic, Bounded, Flat), halo = (3, 3))

# An eastward wind in air warmer and moister than the surface below it
function warm_prescribed_atmosphere()
    atmosphere = PrescribedAtmosphere(surface_atmosphere_grid((8, 4)), [0.0])
    set!(atmosphere.velocities.u, 10)
    set!(atmosphere.temperature, 318.15)
    set!(atmosphere.specific_humidity, 0.06)
    set!(atmosphere.pressure, 101325)
    return atmosphere
end

surface_grid(size) = LatitudeLongitudeGrid(; size, longitude = (0, 360), latitude = (-40, 40), z = (-50, 0))

@testset "Models written from the contract couple" begin
    @testset "Ocean" begin
        ocean = MixedLayerOcean(surface_grid((8, 4, 1)))
        set!(ocean.temperature, 20)

        model = EarthSystemModel(; atmosphere = warm_prescribed_atmosphere(), ocean)
        time_step!(model, 10minutes)

        @test all(interior(ocean.temperature) .> 20)
    end

    @testset "Ocean below the freezing limiter" begin
        ocean = MixedLayerOcean(surface_grid((8, 4, 1)))
        set!(ocean.temperature, -10)

        model = OceanOnlyModel(ocean; atmosphere = warm_prescribed_atmosphere())
        time_step!(model, 10minutes)

        @test all(interior(ocean.temperature) .> -3)
    end

    @testset "Sea ice" begin
        grid = surface_grid((8, 4, 1))
        ocean = MixedLayerOcean(grid)
        set!(ocean.temperature, 5)

        sea_ice = StaticSeaIce(grid; Δt = 10minutes)
        set!(sea_ice.state.hi, 1)
        set!(sea_ice.state.ℵ, 1)
        set!(sea_ice.state.T, -2)

        model = EarthSystemModel(; atmosphere = warm_prescribed_atmosphere(), sea_ice, ocean)
        time_step!(model, 10minutes)

        # The warm air heats the ice from above, and the warm ocean loses heat to the ice above it
        @test all(interior(sea_ice.net_fluxes.top.heat) .< 0)
        @test all(interior(ocean.net_fluxes.T) .> 0)
    end

    @testset "Land" begin
        land = BucketLand(surface_grid((8, 4, 1)))
        set!(land.temperature, 290)
        set!(land.saturation, 1)

        model = AtmosphereLandModel(warm_prescribed_atmosphere(), land)
        time_step!(model, 10minutes)

        @test all(interior(land.temperature) .> 290)
    end

    @testset "Atmosphere on a different horizontal grid" begin
        atmosphere = BoxAtmosphere(surface_atmosphere_grid((8, 4)))
        set!(atmosphere.state.u, 10)
        set!(atmosphere.state.T, 318.15)
        set!(atmosphere.state.q, 0.06)
        set!(atmosphere.state.p, 101325)

        ocean = MixedLayerOcean(surface_grid((32, 16, 1)))
        set!(ocean.temperature, 20)

        model = EarthSystemModel(; atmosphere, ocean)
        time_step!(model, 10minutes)

        @test all(interior(ocean.temperature) .> 20)
        @test all(interior(atmosphere.heat_flux) .< 0)
    end
end
