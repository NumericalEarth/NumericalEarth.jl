module Atmospheres

export atmosphere_simulation, PrescribedAtmosphere, PrescribedPrecipitationFlux

import Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Center, Face, Field
using Oceananigans.Grids: grid_name, topology, Flat
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units: Time
using Oceananigans.Utils: Utils, launch!, prettysummary

using Adapt: Adapt, adapt
using Thermodynamics.Parameters: AbstractThermodynamicsParameters
using KernelAbstractions: @kernel, @index
using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters, ComponentExchanger
using Oceananigans.TimeSteppers: update_state!

# Can be extended by atmosphere models
function atmosphere_simulation end

include("thermodynamic_parameters.jl")
include("prescribed_atmosphere.jl")
include("prescribed_atmosphere_regridder.jl")
include("interpolate_atmospheric_state.jl")

NumericalEarth.EarthSystemModels.is_prescribed_atmosphere(::PrescribedAtmosphere) = true
NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(::PrescribedAtmosphere) = nothing

end # module Atmospheres
