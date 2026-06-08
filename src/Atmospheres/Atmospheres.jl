module Atmospheres

export atmosphere_simulation, breeze_prognostic_state, PrescribedAtmosphere, PrescribedPrecipitationFlux

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, prognostic_state, restore_prognostic_state!
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field, Face, Center
using Oceananigans.Grids: grid_name, topology, Flat
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: Utils, prettysummary, launch!
using Thermodynamics.Parameters: AbstractThermodynamicsParameters

using ...NumericalEarth: NumericalEarth
using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters, ComponentExchanger

# Can be extended by atmosphere models
function atmosphere_simulation end

# Map a moist thermodynamic state (T, qᵛ, qᶜ, qⁱ, p) to an atmosphere model's
# prognostic fields. Extended by atmosphere models (see NumericalEarthBreezeExt).
function breeze_prognostic_state end

include("thermodynamic_parameters.jl")
include("prescribed_atmosphere.jl")
include("prescribed_atmosphere_regridder.jl")
include("interpolate_atmospheric_state.jl")

EarthSystemModels.InterfaceComputations.net_fluxes(::PrescribedAtmosphere) = nothing

end # module Atmospheres
