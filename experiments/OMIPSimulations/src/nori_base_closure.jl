# NORi Base Vertical Diffusivity closure.
#
# Verbatim copy of the closure defined in
#   https://github.com/xkykai/NORiOceanParameterization.jl
#   /blob/main/NORiImplementation/src/NORiBaseVerticalDiffusivity.jl
# vendored here so OMIPSimulations stays a leaf project (no extra dep
# resolution against the upstream repo). The JLD2 / `parameter_file`
# loading branch has been removed because we don't carry the calibrated
# parameter file — the constructor defaults are the trained values
# reported in the upstream README.
#
# Physics: Richardson-number-based vertical mixing.
#   - Convective (Ri < 0):  tanh transition between shear and convective
#                           viscosities.
#   - Stable    (Ri > 0):  linear interpolation between background and
#                           shear viscosities, clamped to [ν₀, νˢʰ].
#   - Separate Prandtl numbers for convective vs shear regimes.
#   - Horizontal 9-point filter on Ri before evaluation.

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.Grids: inactive_node, total_size
using Oceananigans.Operators
using Oceananigans.Operators: ℑxyᶠᶠᵃ, ℑxyᶜᶜᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures:
    AbstractScalarDiffusivity,
    VerticalFormulation,
    VerticallyImplicitTimeDiscretization,
    getclosure

import Oceananigans.TurbulenceClosures: viscosity, diffusivity, compute_closure_fields!, build_closure_fields
using Oceananigans.Utils: KernelParameters, launch!, prettysummary, time_difference_seconds

using Adapt
using KernelAbstractions: @index, @kernel

#####
##### NORi Base Vertical Diffusivity Closure
#####

"""
    NORiBaseVerticalDiffusivity{TD, FT}

Richardson-number-based vertical diffusivity closure.

Fields
- `ν₀::FT`        background viscosity
- `νˢʰ::FT`       shear-driven viscosity
- `νᶜⁿ::FT`       convective viscosity
- `Pr_convₜ::FT`  Prandtl number, convective regime
- `Pr_shearₜ::FT` Prandtl number, shear regime
- `Riᶜ::FT`       critical Richardson number
- `δRi::FT`       Richardson-number transition scale
"""
struct NORiBaseVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    ν₀        :: FT
    νˢʰ       :: FT
    νᶜⁿ       :: FT
    Pr_convₜ  :: FT
    Pr_shearₜ :: FT
    Riᶜ       :: FT
    δRi       :: FT
end

"""
    NORiBaseVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization(),
                                FT=Float64; kwargs...)

Construct a NORi base vertical diffusivity closure with the calibrated
defaults reported in xkykai/NORiOceanParameterization.jl.
"""
function NORiBaseVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                     FT = Float64;
                                     ν₀        = 1e-5,
                                     νˢʰ       = 0.0615914063656973,
                                     νᶜⁿ       = 0.7612393837759673,
                                     Pr_convₜ  = 0.1749433627329692,
                                     Pr_shearₜ = 1.0842017486284887,
                                     Riᶜ       = 0.4366901962987793,
                                     δRi       = 0.009695724988589002)

    TD = typeof(time_discretization)

    return NORiBaseVerticalDiffusivity{TD, FT}(
        convert(FT, ν₀),
        convert(FT, νˢʰ),
        convert(FT, νᶜⁿ),
        convert(FT, Pr_convₜ),
        convert(FT, Pr_shearₜ),
        convert(FT, Riᶜ),
        convert(FT, δRi)
    )
end

NORiBaseVerticalDiffusivity(FT::DataType; kw...) =
    NORiBaseVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Adapt.adapt_structure(to, clo::NORiBaseVerticalDiffusivity{TD, FT}) where {TD, FT} =
    NORiBaseVerticalDiffusivity{TD, FT}(clo.ν₀, clo.νˢʰ, clo.νᶜⁿ, clo.Pr_convₜ, clo.Pr_shearₜ, clo.Riᶜ, clo.δRi)

#####
##### Diffusivity field utilities
#####

const NBVD = NORiBaseVerticalDiffusivity
const NBVDArray = AbstractArray{<:NBVD}
const FlavorOfNBVD = Union{NBVD, NBVDArray}
const _nori_c = Center()
const _nori_f = Face()

@inline viscosity_location(::FlavorOfNBVD)   = (_nori_c, _nori_c, _nori_f)
@inline diffusivity_location(::FlavorOfNBVD) = (_nori_c, _nori_c, _nori_f)

@inline viscosity(::FlavorOfNBVD, diffusivities) = diffusivities.κᵘ
@inline diffusivity(::FlavorOfNBVD, diffusivities, id) = diffusivities.κᶜ

with_tracers(tracers, closure::FlavorOfNBVD) = closure

function build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfNBVD)
    κᶜ = Field((Center(), Center(), Face()), grid)
    κᵘ = Field((Center(), Center(), Face()), grid)
    Ri = Field((Center(), Center(), Face()), grid)
    previous_compute_time = Ref(clock.time)
    return (; κᶜ, κᵘ, Ri, previous_compute_time)
end

function update_previous_compute_time!(closure_fields, model)
    Δt = time_difference_seconds(model.clock.time, closure_fields.previous_compute_time[])
    closure_fields.previous_compute_time[] = model.clock.time
    return Δt
end

#####
##### Compute diffusivities
#####

function compute_closure_fields!(diffusivities, closure::FlavorOfNBVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    Δt = update_previous_compute_time!(diffusivities, model)
    Δt == 0 && return nothing

    # Step 1: Compute Richardson number on the interior only — halo cells of T/S
    # may not be filled with physical values (e.g. under flux BCs), and
    # ∂z_b → TEOS10 throws DomainError on non-physical salinity. We fill Ri's
    # halos via fill_halo_regions! afterwards.
    launch!(arch, grid, parameters, compute_ri_number!,
            diffusivities, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock)

    # Step 2: Fill halos (use only_local_halos to avoid communication)
    fill_halo_regions!(diffusivities.Ri; only_local_halos=true)

    # Step 3: Compute diffusivities based on Richardson number
    launch!(arch, grid, parameters, compute_NORi_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock)

    return nothing
end

#####
##### Richardson number calculation
#####

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    return ∂z_u² + ∂z_v²
end

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    Ri = N² / (S² + 1e-11)
    return ifelse(N² == 0, zero(grid), Ri)
end

@kernel function compute_ri_number!(diffusivities, grid, closure::FlavorOfNBVD,
                                    velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
end

#####
##### Diffusivity computation kernel
#####

@kernel function compute_NORi_diffusivities!(diffusivities, grid, closure::FlavorOfNBVD,
                                             velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    _compute_NORi_diffusivities!(i, j, k, diffusivities, grid, closure,
                                 velocities, tracers, buoyancy, tracer_bcs, clock)
end

@inline function _compute_NORi_diffusivities!(i, j, k, diffusivities, grid, closure,
                                              velocities, tracers, buoyancy, tracer_bcs, clock)
    closure_ij = getclosure(i, j, closure)

    ν₀        = closure_ij.ν₀
    νˢʰ       = closure_ij.νˢʰ
    νᶜⁿ       = closure_ij.νᶜⁿ
    Pr_convₜ  = closure_ij.Pr_convₜ
    Pr_shearₜ = closure_ij.Pr_shearₜ
    Riᶜ       = closure_ij.Riᶜ
    δRi       = closure_ij.δRi

    κ₀  = ν₀ / Pr_shearₜ
    κˢʰ = νˢʰ / Pr_shearₜ
    κᶜⁿ = νᶜⁿ / Pr_convₜ

    # 9-point horizontal filter on Ri.
    Ri = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyᶠᶠᵃ, diffusivities.Ri)

    convecting = Ri < 0

    if convecting
        ν_local = (νˢʰ - νᶜⁿ) * tanh(Ri / δRi) + νˢʰ
        κ_local = (κˢʰ - κᶜⁿ) * tanh(Ri / δRi) + κˢʰ
    else
        ν_local = clamp((ν₀ - νˢʰ) * Ri / Riᶜ + νˢʰ, ν₀, νˢʰ)
        κ_local = clamp((κ₀ - κˢʰ) * Ri / Riᶜ + κˢʰ, κ₀, κˢʰ)
    end

    is_interior = k > 1 && k < grid.Nz + 1

    @inbounds diffusivities.κᵘ[i, j, k] = ifelse(is_interior, ν_local, 0)
    @inbounds diffusivities.κᶜ[i, j, k] = ifelse(is_interior, κ_local, 0)

    return nothing
end

#####
##### Show
#####

@inline time_discretization(::NORiBaseVerticalDiffusivity{TD}) where TD = TD()

function Base.summary(closure::NBVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("NORiBaseVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, closure::NBVD)
    print(io, summary(closure))
    print(io, '\n')
    print(io, "├── ν₀: ", prettysummary(closure.ν₀), '\n',
              "├── νˢʰ: ", prettysummary(closure.νˢʰ), '\n',
              "├── νᶜⁿ: ", prettysummary(closure.νᶜⁿ), '\n',
              "├── Pr_convₜ: ", prettysummary(closure.Pr_convₜ), '\n',
              "├── Pr_shearₜ: ", prettysummary(closure.Pr_shearₜ), '\n',
              "├── Riᶜ: ", prettysummary(closure.Riᶜ), '\n',
              "└── δRi: ", prettysummary(closure.δRi))
end
