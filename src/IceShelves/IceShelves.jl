module IceShelves

export IceShelfOceanInterface,
       TEOS10Liquidus,
       VelocityBasedFrictionVelocity,
       compute_ice_shelf_fluxes!,
       ice_shelf_boundary_conditions,
       ice_shelf_tracer_forcing

using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.BuoyancyFormulations: BuoyancyForce, SeawaterBuoyancy
using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: Center, Face, rnode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryCondition, MutableGridOfSomeKind, immersed_cell
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, Δzᶜᶜᶜ
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using ClimaSeaIce.SeaIceThermodynamics: LinearLiquidus

import ClimaSeaIce.SeaIceThermodynamics: melting_temperature

using ..EarthSystemModels.InterfaceComputations: ThreeEquationHeatFlux, compute_interface_heat_flux, ice_ocean_interface_fluxes,
                                                   ice_base_height
import ..EarthSystemModels.InterfaceComputations: ice_ocean_interface_heat_flux
using ..Oceans: bottom_drag_bc, free_surface_grid,
                u_immersed_drag_coefficient, v_immersed_drag_coefficient,
                u_immersed_bottom_drag, v_immersed_bottom_drag,
                surface_layer_average

include("teos10_liquidus.jl")
include("ice_shelf_ocean_interface.jl")
include("ice_shelf_fluxes.jl")
include("ice_shelf_boundary_conditions.jl")
include("ice_shelf_tracer_forcing.jl")

end # module IceShelves
