module Radiations

export PrescribedRadiation,
       SurfaceRadiationProperties,
       InterfaceRadiationFlux,
       LatitudeDependentAlbedo,
       TabulatedAlbedo

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Center, Face, Field, ZeroField, FractionalIndices, instantiate, interpolator
using Oceananigans.Grids: grid_name, prettysummary, ηnode, _node, topology, Flat, on_architecture
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series, cpu_interpolating_time_indices
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!

using Adapt
using KernelAbstractions: @kernel, @index

import NumericalEarth: stateindex
using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters

import Oceananigans.TimeSteppers: time_step!, update_state!
import Oceananigans.Architectures: on_architecture

import NumericalEarth.EarthSystemModels: interpolate_state!,
                                         update_net_fluxes!,
                                         apply_air_sea_radiative_fluxes!,
                                         apply_air_sea_ice_radiative_fluxes!,
                                         allocate_interface_fluxes!

import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger,
                                                                initialize!,
                                                                kernel_radiation_properties,
                                                                air_sea_interface_radiation_state,
                                                                air_sea_ice_interface_radiation_state

include("surface_radiation_properties.jl")
include("interface_radiation_flux.jl")
include("radiation_kernels.jl")
include("latitude_dependent_albedo.jl")
include("tabulated_albedo.jl")
include("prescribed_radiation.jl")
include("prescribed_radiation_regridder.jl")
include("interpolate_radiation_state.jl")
include("air_sea_interface_radiation_state.jl")
include("apply_air_sea_radiative_fluxes.jl")
include("apply_air_sea_ice_radiative_fluxes.jl")

end # module Radiations
