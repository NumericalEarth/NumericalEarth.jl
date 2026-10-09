module SeaIces

export sea_ice_simulation, FreezingLimitedOceanTemperature

using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: inactive_node, Face, Center
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids
using Oceananigans.Simulations: Simulation
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: minutes
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

using ..EarthSystemModels: EarthSystemModels, default_stop_time
using ..EarthSystemModels.InterfaceComputations: InterfaceComputations, ComponentExchanger,
                                                 ThreeEquationHeatFlux

include("freezing_limited_ocean_temperature.jl")
include("sea_ice_simulation.jl")
include("assemble_net_sea_ice_fluxes.jl")

EarthSystemModels.default_sea_ice() = FreezingLimitedOceanTemperature()

# A sea ice simulation is not an ocean — used to catch swapped positional args
# in the convenience constructors (e.g. `OceanSeaIceModel`).
EarthSystemModels.is_ocean_component(::Simulation{<:SeaIceModel}) = false

# When using an ClimaSeaIce simulation, we assume that the exchange grid is the sea-ice grid
EarthSystemModels.interpolate_state!(exchanger, grid, ::Simulation{<:SeaIceModel},       coupled_model) = nothing
EarthSystemModels.interpolate_state!(exchanger, grid, ::FreezingLimitedOceanTemperature, coupled_model) = nothing

# ComponentExchangers
InterfaceComputations.ComponentExchanger(sea_ice::FreezingLimitedOceanTemperature, grid) = nothing

function InterfaceComputations.ComponentExchanger(sea_ice::Simulation{<:SeaIceModel}, grid)
    sea_ice_grid = sea_ice.model.grid

    if sea_ice_grid == grid
        u  = sea_ice.model.velocities.u
        v  = sea_ice.model.velocities.v
        hi = sea_ice.model.ice_thickness
        hc = sea_ice.model.ice_consolidation_thickness
        ℵ  = sea_ice.model.ice_concentration
        hs = sea_ice.model.snow_thickness
    else
        u  = Field{Center, Center, Nothing}(grid)
        v  = Field{Center, Center, Nothing}(grid)
        hi = Field{Center, Center, Nothing}(grid)
        hc = Field{Center, Center, Nothing}(grid)
        ℵ  = Field{Center, Center, Nothing}(grid)
        hs = Field{Center, Center, Nothing}(grid)
    end

    # When there's no snow model, use ZeroField so kernels can read hs[i,j,1] = 0
    if isnothing(hs)
        hs = ZeroField(eltype(grid))
    end

    return ComponentExchanger((; u, v, hi, hc, ℵ, hs), nothing)
end

end # module SeaIces
