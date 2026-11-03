module SeaIces

export sea_ice_simulation, FreezingLimitedOceanTemperature, LandfastBasalStress

using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, ValueBoundaryCondition, Zipper
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: Grids, inactive_node, Face, Center
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids
using Oceananigans.Simulations: Simulation
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: minutes
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

using ..EarthSystemModels: EarthSystemModels, default_stop_time
using ..EarthSystemModels.InterfaceComputations: InterfaceComputations, ThreeEquationHeatFlux

include("freezing_limited_ocean_temperature.jl")
include("sea_ice_simulation.jl")
include("assemble_net_sea_ice_fluxes.jl")

EarthSystemModels.default_sea_ice() = FreezingLimitedOceanTemperature()

# A sea ice simulation is not an ocean — used to catch swapped positional args
# in the convenience constructors (e.g. `OceanSeaIceModel`).
EarthSystemModels.is_ocean_component(::Simulation{<:SeaIceModel}) = false

InterfaceComputations.ComponentExchanger(sea_ice::FreezingLimitedOceanTemperature, grid) = nothing

function EarthSystemModels.surface_state(model::SeaIceModel)
    u  = model.velocities.u
    v  = model.velocities.v
    hi = model.ice_thickness
    hc = model.ice_consolidation_thickness
    ℵ  = model.ice_concentration
    hs = model.snow_thickness

    # When there's no snow model, use ZeroField so kernels can read hs[i,j,1] = 0
    if isnothing(hs)
        hs = ZeroField(eltype(Grids.grid(model)))
    end

    snow_thermodynamics = model.snow_thermodynamics
    top_thermodynamics = isnothing(snow_thermodynamics) ? model.ice_thermodynamics : snow_thermodynamics

    return (; u, v, hi, hc, ℵ, hs, S = model.tracers.S, T = top_thermodynamics.top_surface_temperature)
end

EarthSystemModels.sea_ice_phase_transitions(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.phase_transitions
EarthSystemModels.sea_ice_mass_fluxes(sea_ice::Simulation{<:SeaIceModel}) = sea_ice.model.mass_fluxes.thermodynamics

function EarthSystemModels.sea_ice_ocean_stress(sea_ice::Simulation{<:SeaIceModel})
    dynamics = sea_ice.model.dynamics
    return isnothing(dynamics) ? nothing : dynamics.external_momentum_stresses.bottom
end

end # module SeaIces
