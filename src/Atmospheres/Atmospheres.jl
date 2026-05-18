module Atmospheres

export atmosphere_simulation, PrescribedAtmosphere, PrescribedPrecipitationFlux

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field, Face, Center
using Oceananigans.Grids: grid_name, topology, Flat
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: Utils, prettysummary, launch!
using Thermodynamics.Parameters: AbstractThermodynamicsParameters

using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters, ComponentExchanger

# Can be extended by atmosphere models
function atmosphere_simulation end

include("thermodynamic_parameters.jl")
include("prescribed_atmosphere.jl")
include("prescribed_atmosphere_regridder.jl")
include("interpolate_atmospheric_state.jl")

EarthSystemModels.InterfaceComputations.net_fluxes(::PrescribedAtmosphere) = nothing

end # module Atmospheres
