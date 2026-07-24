#####
##### Drag-partition roughness-sublayer closure
#####
##### Momentum roughness length z0 and zero-plane displacement d0 from canopy area
##### index Λ and canopy height h, following the drag-partition roughness sublayer of
##### Raupach (1994) as parameterized by Jasinski et al. (2005) and compiled by
##### Borak et al. (2025). Pure, allocation-free and kernel-safe.
#####

"""
$(TYPEDEF)

Drag-partition parameters for one vegetation group. `γ ≡ Uh/u★` partitions momentum
between vegetation form drag and substrate friction drag. All fields are dimensionless.

$(TYPEDFIELDS)
"""
struct DragPartitionParameters{FT}
    "form drag coefficient `Cᴿ`"
    form_drag_coefficient :: FT
    "substrate (ground) drag coefficient `Cˢ`"
    substrate_drag_coefficient :: FT
    "maximum friction-to-wind ratio `(u★/Uh)ₘₐₓ`"
    maximum_friction_ratio :: FT
    "roughness-sublayer wind-profile decay coefficient `c`"
    sublayer_decay_coefficient :: FT
    "displacement coefficient `α`"
    displacement_coefficient :: FT
    "critical (skimming) canopy area index `Λₘₐₓ`"
    maximum_area_index :: FT
end

function DragPartitionParameters(FT=Oceananigans.defaults.FloatType;
                                 form_drag_coefficient,
                                 substrate_drag_coefficient,
                                 maximum_friction_ratio,
                                 sublayer_decay_coefficient,
                                 displacement_coefficient,
                                 maximum_area_index)
    return DragPartitionParameters(convert(FT, form_drag_coefficient),
                                   convert(FT, substrate_drag_coefficient),
                                   convert(FT, maximum_friction_ratio),
                                   convert(FT, sublayer_decay_coefficient),
                                   convert(FT, displacement_coefficient),
                                   convert(FT, maximum_area_index))
end

# The five Borak et al. (2025) drag-partition groups. IGBP classes map onto these via
# `drag_partition_group` (canopy_classes.jl); `canopy_drag_parameters(FT, class)` resolves a
# class to its group's parameters.
drag_group_parameters(FT, group::Symbol) = drag_group_parameters(FT, Val(group))
drag_group_parameters(FT, ::Val{:boreal})    = DragPartitionParameters{FT}(0.21, 0.0030, 0.27, 0.28, 1.90, 1.90)
drag_group_parameters(FT, ::Val{:broadleaf}) = DragPartitionParameters{FT}(0.31, 0.0030, 0.31, 0.36, 1.15, 1.70)
drag_group_parameters(FT, ::Val{:grassland}) = DragPartitionParameters{FT}(0.43, 0.0030, 0.32, 0.49, 1.30, 1.30)
drag_group_parameters(FT, ::Val{:cropland})  = DragPartitionParameters{FT}(0.31, 0.0030, 0.29, 0.39, 1.55, 1.50)
drag_group_parameters(FT, ::Val{:shrubland}) = DragPartitionParameters{FT}(0.50, 0.0030, 0.38, 0.48, 1.00, 1.60)

"""
$(TYPEDSIGNATURES)

Wind-to-friction-velocity ratio `γ ≡ Uh/u★` solved by fixed-point iteration of
`γ = (Cˢ + Λ·Cᴿ/2)^(-1/2) · exp(c·Λ·γ/4)`, then capped so `u★/Uh ≤ (u★/Uh)ₘₐₓ`.
`Λ` is limited to `Λₘₐₓ` (the skimming regime).
"""
@inline function canopy_wind_ratio(Λ, p::DragPartitionParameters, iterations)
    Cᴿ = p.form_drag_coefficient
    Cˢ = p.substrate_drag_coefficient
    c  = p.sublayer_decay_coefficient
    Λ  = min(max(Λ, 0), p.maximum_area_index)
    γ₀ = inv(sqrt(Cˢ + Λ * Cᴿ / 2))
    γ  = γ₀
    for _ in 1:iterations
        γ = γ₀ * exp(c * Λ * γ / 4)
    end
    return max(γ, inv(p.maximum_friction_ratio))
end

"""
$(TYPEDSIGNATURES)

Zero-plane displacement height `d0` from `d0/h = (βΛ/(2+βΛ))·(1 − α/(γ√Λ))` with
`β = Cᴿ/Cˢ`. Monotonic and near-linear in `Λ`, so it carries a clean seasonal signal.
"""
@inline function zero_plane_displacement(Λ, γ, h, p::DragPartitionParameters)
    Cᴿ = p.form_drag_coefficient
    Cˢ = p.substrate_drag_coefficient
    α  = p.displacement_coefficient
    Λ  = min(max(Λ, 0), p.maximum_area_index)
    β  = Cᴿ / Cˢ
    invsqrtΛ = ifelse(Λ > 0, inv(sqrt(Λ)), zero(Λ))
    d0h = (β * Λ / (2 + β * Λ)) * (1 - α / γ * invsqrtΛ)
    return h * clamp(d0h, 0, 1)
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0` from `z0/h = (1 − d0/h)·exp(−κγ + ψₕ)`. Logarithmic and
non-monotonic in `Λ` (roughness falls once `Λ > Λₘₐₓ`, the skimming effect).
"""
@inline function canopy_roughness_length(γ, d0, h, κ, ψₕ)
    d0h = ifelse(h > 0, d0 / h, zero(d0))
    return h * (1 - d0h) * exp(-κ * γ + ψₕ)
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0` and zero-plane displacement `d0` (meters) for a canopy of
area index `Λ` and height `h`, sharing the wind ratio `γ`. Returns `(z0, d0)`. `κ` is the
von Kármán constant and `ψₕ` the roughness-sublayer influence function.

```jldoctest
julia> using NumericalEarth.Lands

julia> p = canopy_drag_parameters(Float64, :evergreen_broadleaf_forest);

julia> z0, d0 = canopy_roughness(6.0, 24.72, p, 0.4, 0.193, 20);

julia> round.((z0, d0), digits=2)
(2.32, 17.76)
```
"""
@inline function canopy_roughness(Λ, h, p::DragPartitionParameters, κ, ψₕ, iterations)
    γ  = canopy_wind_ratio(Λ, p, iterations)
    d0 = zero_plane_displacement(Λ, γ, h, p)
    z0 = canopy_roughness_length(γ, d0, h, κ, ψₕ)
    return z0, d0
end

"""
$(TYPEDSIGNATURES)

Semi-empirical displacement height `d0 ≈ ⅔h` from canopy height alone
(Brutsaert 1982; Parlange & Brutsaert 1989) — a height-only estimate, independent of
the drag partition.
"""
@inline semiempirical_displacement(h) = 2h / 3

"""
$(TYPEDSIGNATURES)

Semi-empirical roughness length `z0 ≈ d0/5` from canopy height alone — the height-only
estimate paired with [`semiempirical_displacement`](@ref).
"""
@inline semiempirical_roughness(h) = semiempirical_displacement(h) / 5

"""
$(TYPEDEF)

Drag-partition canopy roughness closure (Raupach 1994 / Jasinski 2005, Borak et al. 2025).
Holds the drag-partition parameters and representative canopy height for a single IGBP
vegetation class (`vegetation_type`, default `:evergreen_broadleaf_forest`; see
[`canopy_drag_parameters`](@ref) and [`representative_canopy_height`](@ref)) plus the shared
closure constants (von Kármán constant, roughness-sublayer influence, fixed-point iteration
count). The per-cell area index and canopy height come from the `cell` passed to
[`aerodynamic_parameters`](@ref); where no measured height is supplied the class's
representative height fills in. A closure evaluates through the
`aerodynamic_parameters(closure, cell)` contract, so other roughness closures can be added
against the same interface.

$(TYPEDFIELDS)
"""
struct DragPartitionRoughness{FT}
    "drag-partition parameters for the canopy vegetation class"
    parameters :: DragPartitionParameters{FT}
    "representative canopy height (m), the fallback where no measured height is supplied"
    representative_height :: FT
    "von Kármán constant `κ`"
    von_karman_constant :: FT
    "roughness-sublayer influence `ψₕ`"
    sublayer_influence :: FT
    "fixed-point iterations for the wind ratio `γ`"
    iterations :: Int
end

DragPartitionRoughness(FT = Oceananigans.defaults.FloatType;
                       vegetation_type = :evergreen_broadleaf_forest,
                       parameters = canopy_drag_parameters(FT, vegetation_type),
                       representative_height = representative_canopy_height(FT, vegetation_type),
                       von_karman_constant = 0.4,
                       sublayer_influence = 0.193,   # ψₕ (Raupach 1995)
                       iterations = 20) =
    DragPartitionRoughness(parameters, convert(FT, representative_height),
                           convert(FT, von_karman_constant),
                           convert(FT, sublayer_influence), Int(iterations))
