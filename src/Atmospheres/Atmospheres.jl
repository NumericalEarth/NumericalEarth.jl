module Atmospheres

export atmosphere_model, atmosphere_simulation, breeze_prognostic_state, bulk_drag,
       hydrostatic_pressure_from_surface, density_from_pressure, PrescribedAtmosphere, PrescribedPrecipitationFlux

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, prognostic_state, restore_prognostic_state!
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrids
using Oceananigans.Fields: Field, Face, Center
using Oceananigans.Grids: grid_name, topology, Bounded, Flat, LatitudeLongitudeGrid, λnodes, φnodes,
                          minimum_xspacing, minimum_yspacing
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time, meters, second
using Oceananigans.Utils: Utils, prettysummary, launch!
using Thermodynamics.Parameters: AbstractThermodynamicsParameters

using ...NumericalEarth: NumericalEarth
using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent, set_prescribed_field!,
                           default_gas_constant, default_dry_air_molar_mass
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters, ComponentExchanger

# Can be extended by atmosphere models. `atmosphere_model` builds the model; `atmosphere_simulation`
# wraps it in a `Simulation`.
function atmosphere_model end
function atmosphere_simulation end

"""
    bulk_drag(model; roughness_length = 0.1, von_karman_constant = 0.4)

Return a callback `f(simulation)` that fills the model's coupling bottom-stress fields (ρτˣ, ρτʸ)
with a bulk neutral surface stress ρτ = −ρ Cᵈ |U| U, with a per-column log-law drag coefficient
Cᵈ = (κ / ln(z₁/z₀))² evaluated at the first-cell-center height z₁ above the local surface —
a stand-in for land/ocean surface coupling. Methods live in atmosphere-model extensions
(e.g. NumericalEarthBreezeExt).
"""
function bulk_drag end

# Map a moist thermodynamic state (T, qᵛ, qᶜ, qⁱ, p) to an atmosphere model's
# prognostic fields. Extended by atmosphere models (see NumericalEarthBreezeExt).
function breeze_prognostic_state end

# Conservative advective Δt cap (s) for the outer time-step wizard: safety · min(Δx, Δy) / jet_speed,
# assuming a jet_speed ≈ 100 m/s upper bound. On a LatitudeLongitudeGrid the smallest zonal spacing sits
# at the highest |latitude|, so minimum_xspacing is the binding constraint. Atmosphere twin of the ocean's
# estimate_maximum_Δt in Oceans.
estimate_maximum_Δt(grid; jet_speed = 100meters/second, safety = 0.5) =
    safety * min(minimum_xspacing(grid), minimum_yspacing(grid)) / jet_speed

include("hydrostatic_pressure.jl")
include("thermodynamic_parameters.jl")
include("prescribed_atmosphere.jl")
include("prescribed_atmosphere_regridder.jl")
include("interpolate_atmospheric_state.jl")

EarthSystemModels.InterfaceComputations.net_fluxes(::PrescribedAtmosphere) = nothing

end # module Atmospheres
