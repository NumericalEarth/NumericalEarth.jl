module SeaIces

export sea_ice_simulation, FreezingLimitedOceanTemperature, LandfastBasalStress

using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, ValueBoundaryCondition, Zipper
using Oceananigans.Fields: Field, ZeroField
using Oceananigans.Grids: inactive_node, Face, Center
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids
using Oceananigans.Simulations: Simulation
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: minutes
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

using ..EarthSystemModels: EarthSystemModels, default_stop_time
using ..EarthSystemModels.InterfaceComputations: InterfaceComputations, ComponentExchanger, ThreeEquationHeatFlux

include("freezing_limited_ocean_temperature.jl")
include("sea_ice_simulation.jl")
include("assemble_net_sea_ice_fluxes.jl")

EarthSystemModels.default_sea_ice() = FreezingLimitedOceanTemperature()

# A sea ice simulation is not an ocean — used to catch swapped positional args
# in the convenience constructors (e.g. `OceanSeaIceModel`).
EarthSystemModels.is_ocean_component(::Simulation{<:SeaIceModel}) = false

# A separate sea-ice vertical grid shares the ocean's horizontal cells.
# Transfer the two-dimensional state used by the interface fluxes.
@kernel function _copy_sea_ice_surface_state!(destination, source, ice_grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        destination.u[i, j, 1]  = ℑxᶠᵃᵃ(i, j, 1, ice_grid, source.u)
        destination.v[i, j, 1]  = ℑyᵃᶠᵃ(i, j, 1, ice_grid, source.v)
        destination.hi[i, j, 1] = source.hi[i, j, 1]
        destination.hc[i, j, 1] = source.hc[i, j, 1]
        destination.ℵ[i, j, 1]  = source.ℵ[i, j, 1]
        destination.hs[i, j, 1] = source.hs[i, j, 1]
    end
end

function EarthSystemModels.interpolate_state!(exchanger, grid, sea_ice::Simulation{<:SeaIceModel}, coupled_model)
    ice_grid = sea_ice.model.grid
    ice_grid == grid && return nothing
    isnothing(sea_ice.model.snow_thickness) &&
        error("The separate-grid sea-ice exchanger currently requires a snow layer")
    size(ice_grid)[1:2] == size(grid)[1:2] ||
        error("Sea-ice and ocean exchange grids must have the same horizontal size")
    model = sea_ice.model
    source = (u = model.velocities.u, v = model.velocities.v,
              hi = model.ice_thickness, hc = model.ice_consolidation_thickness,
              ℵ = model.ice_concentration, hs = model.snow_thickness)
    launch!(architecture(grid), grid, :xy, _copy_sea_ice_surface_state!,
            exchanger.state, source, ice_grid)
    return nothing
end
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
