module Lands

export PrescribedLand

using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field, Center, Face
using Oceananigans.Grids: grid_name
using Oceananigans.OutputReaders: update_field_time_series!, extract_field_time_series
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!, prettysummary

using ..NumericalEarth: NumericalEarth
using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters, ComponentExchanger

include("prescribed_land.jl")
include("prescribed_land_regridder.jl")
include("interpolate_land_state.jl")

end # module Lands
