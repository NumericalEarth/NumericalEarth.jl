# KPP (K-Profile Parameterization) vertical mixing closure.
#
# Implementation of Large, McWilliams, & Doney (1994) following
# MITgcm/pkg/kpp. Vendored as a leaf submodule of OMIPSimulations
# alongside `nori_base_closure.jl`. Submodule scoping isolates KPP's
# helper names (e.g. `shear_squaredᶜᶜᶠ`, `interior_viscosityᶜᶜᶠ`) from
# NORi's same-named helpers at the OMIPSimulations level.

module KPP

export KPPVerticalDiffusivity, KPPParameters

using Adapt
using KernelAbstractions: @index, @kernel

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: getbc, fill_halo_regions!, FieldBoundaryConditions
using Oceananigans.BuoyancyFormulations: ∂z_b, top_buoyancy_flux,
                                          thermal_expansionᶜᶜᶜ, haline_contractionᶜᶜᶜ,
                                          buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face, znode, peripheral_node, inactive_node, static_column_depthᶜᶜᵃ
using Oceananigans.Operators
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑxyᶜᶜᵃ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ, Δzᶜᶜᶜ
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, VerticalFormulation,
                                       VerticallyImplicitTimeDiscretization,
                                       ExplicitTimeDiscretization, VerticallyBoundedGrid,
                                       getclosure

const VITD = VerticallyImplicitTimeDiscretization
using Oceananigans.Utils: launch!, prettysummary

import Oceananigans.TurbulenceClosures: viscosity, diffusivity,
                                        viscosity_location, diffusivity_location,
                                        with_tracers, time_discretization,
                                        compute_closure_fields!, build_closure_fields,
                                        diffusive_flux_z

using NumericalEarth.Oceans: get_radiative_forcing

include("kpp_parameters.jl")
include("kpp_vertical_diffusivity.jl")
include("kpp_velocity_scales.jl")
include("kpp_interior_mixing.jl")
include("kpp_surface_forcing.jl")
include("kpp_boundary_layer_depth.jl")
include("kpp_boundary_layer_mixing.jl")
include("kpp_nonlocal_flux.jl")
include("kpp_compute_closure_fields.jl")

end # module KPP
