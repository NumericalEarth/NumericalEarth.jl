module Radiations

export PrescribedRadiation,
       SurfaceRadiationProperties,
       InterfaceRadiationFlux,
       LatitudeDependentAlbedo,
       TabulatedAlbedo,
       default_stefan_boltzmann_constant

# CODATA 2018 value of the Stefan–Boltzmann constant, in W m⁻² K⁻⁴.
const default_stefan_boltzmann_constant = 5.670374419e-8

using Adapt: Adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture, CPU
using Oceananigans.Fields: Center, Face, Field, ZeroField, FractionalIndices
using Oceananigans.Grids: grid_name, ηnode, _node, topology, Flat, on_architecture
using Oceananigans.OutputReaders: FieldTimeSeries, update_field_time_series!, extract_field_time_series, cpu_interpolating_time_indices
using Oceananigans.Simulations: Simulation
using Oceananigans.TimeSteppers: Clock, tick!, update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!, prettysummary, interpolator

using ..NumericalEarth: NumericalEarth, stateindex
using ..EarthSystemModels: EarthSystemModels, AbstractPrescribedComponent
using ..EarthSystemModels.InterfaceComputations: interface_kernel_parameters,
                                                 ComponentExchanger,
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
