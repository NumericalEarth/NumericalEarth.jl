module IceShelves

export IceShelfOceanInterface,
       PressureDependentLiquidus,
       TEOS10Liquidus,
       VelocityBasedFrictionVelocity,
       compute_ice_shelf_fluxes!,
       ice_shelf_boundary_conditions,
       ice_shelf_tracer_forcing

using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.BuoyancyFormulations: BuoyancyForce, SeawaterBuoyancy
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryCondition, immersed_cell
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, Δzᶜᶜᶜ
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

using ..EarthSystemModels.InterfaceComputations: ThreeEquationHeatFlux, compute_interface_heat_flux
using ..Oceans: bottom_drag_bc,
                u_immersed_drag_coefficient, v_immersed_drag_coefficient,
                u_immersed_bottom_drag, v_immersed_bottom_drag

include("pressure_dependent_liquidus.jl")
include("teos10_liquidus.jl")
include("ice_shelf_ocean_interface.jl")
include("ice_shelf_fluxes.jl")
include("ice_shelf_boundary_conditions.jl")
include("ice_shelf_tracer_forcing.jl")

end # module IceShelves
