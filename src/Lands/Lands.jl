module Lands

export PrescribedLand

import Oceananigans

using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Center, Face, Field
using Oceananigans.Grids: grid_name
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units: Time
using Oceananigans.Utils: prettysummary

using KernelAbstractions: @kernel, @index
using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters

import Oceananigans.TimeSteppers: time_step!, update_state!

import NumericalEarth.EarthSystemModels: interpolate_state!,
                                         update_net_fluxes!

import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, initialize!

include("prescribed_land.jl")
include("prescribed_land_regridder.jl")
include("interpolate_land_state.jl")

end # module Lands
