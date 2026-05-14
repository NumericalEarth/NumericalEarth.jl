module SeaIces

export sea_ice_simulation, FreezingLimitedOceanTemperature

import Oceananigans
using Oceananigans.Units: Units, minutes
using Oceananigans.Utils: Utils, launch!, with_tracers
using Oceananigans.Grids: architecture, Center, Face
using Oceananigans.Fields: ZeroField, Field
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, inactive_node
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids
using Oceananigans.Operators: Operators, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Simulations: Simulation
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using KernelAbstractions: @kernel, @index
using ClimaSeaIce: ClimaSeaIce, SeaIceModel
using ClimaSeaIce.SeaIceThermodynamics.HeatBoundaryConditions: PrescribedTemperature
using NumericalEarth.EarthSystemModels: EarthSystemModels, EarthSystemModel
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations, SkinTemperature

import NumericalEarth.EarthSystemModels: interpolate_state!,
                                     sea_ice_concentration,
                                     sea_ice_thickness,
                                     reference_density,
                                     heat_capacity,
                                     update_net_fluxes!,
                                     default_sea_ice

import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger,
                                                           compute_atmosphere_sea_ice_fluxes!,
                                                           compute_sea_ice_ocean_fluxes!,
                                                           net_fluxes,
                                                           ThreeEquationHeatFlux,
                                                           default_ai_temperature

import Oceananigans.TimeSteppers: time_step!

include("freezing_limited_ocean_temperature.jl")
include("sea_ice_simulation.jl")
include("assemble_net_sea_ice_fluxes.jl")

default_sea_ice() = FreezingLimitedOceanTemperature()

# When using an ClimaSeaIce simulation, we assume that the exchange grid is the sea-ice grid
interpolate_state!(exchanger, grid, ::Simulation{<:SeaIceModel},       coupled_model) = nothing
interpolate_state!(exchanger, grid, ::FreezingLimitedOceanTemperature, coupled_model) = nothing

# ComponentExchangers
ComponentExchanger(sea_ice::FreezingLimitedOceanTemperature, grid) = nothing

function ComponentExchanger(sea_ice::Simulation{<:SeaIceModel}, grid)
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

end
