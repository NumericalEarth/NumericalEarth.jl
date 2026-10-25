using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans: UpdateStateCallsite
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Simulations: Callback

#####
##### Self-attraction and loading of the ocean tide
#####

abstract type AbstractSelfAttractionAndLoading end

"""
    ScalarSelfAttractionAndLoading(β = 0.1)

The scalar approximation to self-attraction and loading (Accad and Pekeris, 1978), which replaces
the barotropic pressure gradient `g ∇η` with `g (1 - β) ∇η`. Published values of `β` range from about
0.08 to 0.12.

`ocean_simulation` implements the approximation by reducing the gravitational acceleration of the
free surface, so it is treated as implicitly as the free surface itself.
"""
struct ScalarSelfAttractionAndLoading{FT} <: AbstractSelfAttractionAndLoading
    β :: FT
end

ScalarSelfAttractionAndLoading(; β = 0.1) = ScalarSelfAttractionAndLoading(β)

Base.summary(sal::ScalarSelfAttractionAndLoading) = "ScalarSelfAttractionAndLoading(β = $(sal.β))"
Base.show(io::IO, sal::ScalarSelfAttractionAndLoading) = print(io, summary(sal))

"""
    SphericalHarmonicSelfAttractionAndLoading(grid; truncation = 40, nlat_half = 24, love_numbers = nothing,
                                              reference_density = 1035, earth_density = 5517)

Self-attraction and loading computed from the ocean bottom pressure anomaly with load Love numbers, as in
the `sal` package of MITgcm used for LLC4320v2. Each time the model state is updated, the kinematic bottom
pressure anomaly `pᵇ = g η - ∫ b dz` (relative to its value when the simulation starts, and zero over land)
is averaged conservatively onto a Gaussian grid, transformed to spherical harmonics up to degree
`truncation`, and multiplied by

```math
γₗ = \\frac{3 ρ₀}{ρₑ} \\frac{1 + k′ₗ - h′ₗ}{2l + 1},
```

where `k′ₗ` and `h′ₗ` are load Love numbers. The potential `Ψ = - Σₗ γₗ pᵇₗ` is then evaluated
at each cell center of `grid` and added to the barotropic potential, so that `u` and `v` are forced by `-∇Ψ`.
With `γₗ = β` for all `l` this reduces to [`ScalarSelfAttractionAndLoading`](@ref).

The defaults follow LLC4320v2: degree 40 on a Gaussian grid with 48 latitudes, and (with `love_numbers = nothing`)
the load Love numbers of Gegout (2010) in the center-of-mass frame. `reference_density` should be the density
that converts the model's kinematic pressure to pressure.
Requires `using SpeedyTransforms, ConservativeRegridding`.
"""
struct SphericalHarmonicSelfAttractionAndLoading{Ψ, P, O} <: AbstractSelfAttractionAndLoading
    potential :: Ψ
    bottom_pressure_anomaly :: P
    reference_bottom_pressure :: P
    reference_set :: Base.RefValue{Bool}
    operator :: O
end

function SphericalHarmonicSelfAttractionAndLoading(grid; kwargs...)
    operator = spherical_harmonic_loading_operator(grid; kwargs...)
    potential = Field{Center, Center, Nothing}(grid)
    bottom_pressure_anomaly = Field{Center, Center, Nothing}(grid)
    reference_bottom_pressure = Field{Center, Center, Nothing}(grid)
    return SphericalHarmonicSelfAttractionAndLoading(potential, bottom_pressure_anomaly, reference_bottom_pressure,
                                                     Ref(false), operator)
end

Base.summary(sal::SphericalHarmonicSelfAttractionAndLoading) =
    string("SphericalHarmonicSelfAttractionAndLoading with ", summary(sal.operator))

Base.show(io::IO, sal::SphericalHarmonicSelfAttractionAndLoading) = print(io, summary(sal))

"""
    spherical_harmonic_loading_operator(grid; kwargs...)

Return the operator that maps the kinematic bottom pressure anomaly to the self-attraction and loading
potential. Methods are provided by the NumericalEarth extension for SpeedyTransforms and ConservativeRegridding.
"""
spherical_harmonic_loading_operator(args...; kwargs...) =
    error("SphericalHarmonicSelfAttractionAndLoading requires `using SpeedyTransforms, ConservativeRegridding`.")

"""
    apply_loading_operator!(Ψ, operator, pᵇ)

Compute the self-attraction and loading potential `Ψ` from the kinematic bottom pressure anomaly `pᵇ`.
"""
function apply_loading_operator! end

#####
##### Bottom pressure
#####

# Kinematic bottom pressure, up to a constant: pᵇ = g η - Σₖ b Δz over the active cells of each column.
# Columns that are entirely land get zero.
@kernel function _compute_kinematic_bottom_pressure!(pᵇ, grid, η, buoyancy, tracers, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    p = g * @inbounds η[i, j, Nz+1]
    for k in 1:Nz
        b = buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, buoyancy, tracers)
        inactive = inactive_node(i, j, k, grid, Center(), Center(), Center())
        p -= ifelse(inactive, zero(p), b * Δzᶜᶜᶜ(i, j, k, grid))
    end

    land = inactive_node(i, j, Nz, grid, Center(), Center(), Center())
    @inbounds pᵇ[i, j, 1] = ifelse(land, zero(p), p)
end

function compute_bottom_pressure_anomaly!(sal, model)
    grid = model.grid
    buoyancy = model.buoyancy.formulation
    g = buoyancy.gravitational_acceleration
    η = model.free_surface.displacement
    pᵇ = sal.bottom_pressure_anomaly

    launch!(architecture(grid), grid, :xy, _compute_kinematic_bottom_pressure!, pᵇ, grid, η, buoyancy, model.tracers, g)

    # The first bottom pressure is the reference, as in MITgcm when no reference file is given
    if !sal.reference_set[]
        parent(sal.reference_bottom_pressure) .= parent(pᵇ)
        sal.reference_set[] = true
    end

    parent(pᵇ) .-= parent(sal.reference_bottom_pressure)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the self-attraction and loading potential from the current state of `model`.
"""
function compute_self_attraction_and_loading!(model, sal::SphericalHarmonicSelfAttractionAndLoading)
    compute_bottom_pressure_anomaly!(sal, model)
    apply_loading_operator!(sal.potential, sal.operator, sal.bottom_pressure_anomaly)
    fill_halo_regions!(sal.potential)
    return nothing
end

#####
##### Hooks for ocean_simulation
#####

self_attraction_and_loading_free_surface(free_surface, sal) = free_surface

function self_attraction_and_loading_free_surface(free_surface::SplitExplicitFreeSurface{E}, sal::ScalarSelfAttractionAndLoading) where E
    g = free_surface.gravitational_acceleration * (1 - sal.β)
    return SplitExplicitFreeSurface{E}(free_surface.displacement,
                                       free_surface.barotropic_velocities,
                                       free_surface.filtered_state,
                                       g,
                                       free_surface.kernel_parameters,
                                       free_surface.substepping,
                                       free_surface.timestepper,
                                       free_surface.boundary_transport)
end

self_attraction_and_loading_free_surface(free_surface::ImplicitFreeSurface, sal::ScalarSelfAttractionAndLoading) =
    ImplicitFreeSurface(; gravitational_acceleration = free_surface.gravitational_acceleration * (1 - sal.β),
                        solver_method = free_surface.solver_method,
                        free_surface.solver_settings...)

self_attraction_and_loading_potential(Φ, sal) = Φ
self_attraction_and_loading_potential(::Nothing, sal::SphericalHarmonicSelfAttractionAndLoading) = sal.potential
self_attraction_and_loading_potential(Φ, sal::SphericalHarmonicSelfAttractionAndLoading) = (Φ, sal.potential)
self_attraction_and_loading_potential(Φ::Tuple, sal::SphericalHarmonicSelfAttractionAndLoading) = (Φ..., sal.potential)

add_self_attraction_and_loading_callback!(simulation, sal) = nothing

function add_self_attraction_and_loading_callback!(simulation, sal::SphericalHarmonicSelfAttractionAndLoading)
    simulation.callbacks[:self_attraction_and_loading] =
        Callback(compute_self_attraction_and_loading!; parameters = sal, callsite = UpdateStateCallsite())
    return nothing
end
