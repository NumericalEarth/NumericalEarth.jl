#####
##### Drag-partition roughness-sublayer closure
#####
##### Momentum roughness length ℓᵐ and zero-plane displacement d from the leaf area index 𝒜
##### and canopy height h, following the drag-partition roughness sublayer of
##### Raupach (1994) as parameterized by Jasinski et al. (2005) and compiled by
##### Borak et al. (2025).
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
    "critical (skimming) leaf area index `𝒜ᶜ`"
    critical_leaf_area_index :: FT
end

function DragPartitionParameters(FT=Oceananigans.defaults.FloatType;
                                 form_drag_coefficient,
                                 substrate_drag_coefficient,
                                 maximum_friction_ratio,
                                 sublayer_decay_coefficient,
                                 displacement_coefficient,
                                 critical_leaf_area_index)
    return DragPartitionParameters(convert(FT, form_drag_coefficient),
                                   convert(FT, substrate_drag_coefficient),
                                   convert(FT, maximum_friction_ratio),
                                   convert(FT, sublayer_decay_coefficient),
                                   convert(FT, displacement_coefficient),
                                   convert(FT, critical_leaf_area_index))
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
`γ = (Cˢ + 𝒜Cᴿ/2)^(-1/2) · exp(c𝒜γ/4)` (Eq. 4 of [Borak et al. (2025)](@cite borak2025global),
whose canopy area index is the `leaf_area_index` `𝒜` here and whose `c` is the
`sublayer_decay_coefficient`), then capped so `u★/Uh ≤ (u★/Uh)ₘₐₓ`. The index is limited to the
critical index `𝒜ᶜ` here (the skimming cap on the wind ratio); displacement and roughness
downstream use the full index.
"""
@inline function canopy_wind_ratio(leaf_area_index, p::DragPartitionParameters, iterations)
    Cᴿ = p.form_drag_coefficient
    Cˢ = p.substrate_drag_coefficient
    𝒜  = max(leaf_area_index, 0)
    𝒜ᶜ = p.critical_leaf_area_index
    γ₀ = inv(sqrt(Cˢ + min(𝒜, 𝒜ᶜ) * Cᴿ / 2))
    γ  = γ₀
    for _ in 1:iterations
        γ = γ₀ * exp(p.sublayer_decay_coefficient * min(𝒜, 𝒜ᶜ) * γ / 4)
    end
    return max(γ, inv(p.maximum_friction_ratio))
end

"""
$(TYPEDSIGNATURES)

Zero-plane displacement height `d` from `d/h = 𝒜/(2Cˢ/Cᴿ + 𝒜) · (1 − α/(γ√𝒜))`
(Eq. 5 of [Borak et al. (2025)](@cite borak2025global), with their `β = Cᴿ/Cˢ` written out and
their `α` the `displacement_coefficient`).
Monotonic and near-linear in the leaf area index, so it carries a clean seasonal signal.
Uses the actual index (not capped at `𝒜ᶜ`). `𝒜ᶜ` limits only the wind ratio `γ`
(the skimming cap on `u★/Uh`), while displacement keeps rising toward `h` as the canopy
densifies — so `d` reaches the class-averaged values of Borak et al. (2025, Table 5).
"""
@inline function zero_plane_displacement(leaf_area_index, γ, h, p::DragPartitionParameters)
    Cᴿ = p.form_drag_coefficient
    Cˢ = p.substrate_drag_coefficient
    𝒜  = max(leaf_area_index, 0)
    dh = 𝒜 / (2 * Cˢ / Cᴿ + 𝒜) *
         (1 - p.displacement_coefficient / γ * ifelse(𝒜 > 0, inv(sqrt(𝒜)), zero(𝒜)))
    return h * clamp(dh, 0, 1)
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` from `ℓᵐ/h = (1 − d/h)·exp(−ϰγ + ψ)`
(Eq. 3 of [Borak et al. (2025)](@cite borak2025global)), where `ϰ` is the von Kármán constant
and `ψ` the roughness-sublayer influence. Logarithmic and non-monotonic in the leaf area index:
roughness rises, peaks, then falls as the canopy densifies (the skimming effect).
"""
@inline function canopy_roughness_length(γ, d, h, ϰ, sublayer_influence)
    dh = ifelse(h > 0, d / h, zero(d))
    return h * (1 - dh) * exp(-ϰ * γ + sublayer_influence)
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) for a canopy of
`leaf_area_index` and height `h`, sharing the wind ratio `γ`. Returns `(ℓᵐ, d)`. `ϰ` is the
von Kármán constant and `sublayer_influence` the roughness-sublayer influence function.

```jldoctest
julia> using NumericalEarth.Lands

julia> p = canopy_drag_parameters(Float64, :evergreen_broadleaf_forest);

julia> ℓᵐ, d = canopy_roughness(6.0, 24.72, p, 0.4, 0.193, 20);

julia> round.((ℓᵐ, d), digits=2)
(1.22, 21.05)
```
"""
@inline function canopy_roughness(leaf_area_index, h, p::DragPartitionParameters,
                                  ϰ, sublayer_influence, iterations)
    γ  = canopy_wind_ratio(leaf_area_index, p, iterations)
    d  = zero_plane_displacement(leaf_area_index, γ, h, p)
    ℓᵐ = canopy_roughness_length(γ, d, h, ϰ, sublayer_influence)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Semi-empirical displacement height `d ≈ ⅔h` from canopy height alone
(Brutsaert 1982; Parlange & Brutsaert 1989) — a height-only estimate, independent of
the drag partition.
"""
@inline semiempirical_displacement(h) = 2h / 3

"""
$(TYPEDSIGNATURES)

Semi-empirical roughness length `ℓᵐ ≈ d/5` from canopy height alone — the height-only
estimate paired with [`semiempirical_displacement`](@ref).
"""
@inline semiempirical_roughness(h) = semiempirical_displacement(h) / 5

"""
$(TYPEDEF)

Drag-partition canopy roughness closure of [Raupach (1994)](@cite raupach1994simplified),
parameterized for land-cover classes by [Jasinski et al. (2005)](@cite jasinski2005bulk) and
compiled/updated by [Borak et al. (2025)](@cite borak2025global).
Holds the drag-partition parameters and representative canopy height for a single IGBP
vegetation class (`vegetation_type`, default `:evergreen_broadleaf_forest`; see
[`canopy_drag_parameters`](@ref) and [`representative_canopy_height`](@ref)) plus the shared
closure constants (von Kármán constant, roughness-sublayer influence, fixed-point iteration
count). The per-cell leaf area index and canopy height come from the `cell` passed to
[`aerodynamic_parameters`](@ref); where no measured height is supplied the class's
representative height fills in.

Raupach's theory is posed in terms of a *plant* area index (leaves plus stems and branches) and
depends on it through a frontal area — the `/2` in the wind ratio, for randomly oriented
elements. The per-class drag coefficients are regressed against satellite *leaf* area index, so
`leaf_area_index` is the calibrated input and the woody area is already absorbed into `Cᴿ`.
Adding a stem area index on top would double-count it. Note this also means leaf-off roughness
is only as good as that absorption: a deciduous canopy at `leaf_area_index ≈ 0` still has bare
branches the closure does not see.

A closure evaluates through the
`aerodynamic_parameters(closure, cell)` contract, so other roughness closures can be added
against the same interface.

$(TYPEDFIELDS)
"""
struct DragPartitionRoughness{FT}
    "drag-partition parameters for the canopy vegetation class"
    parameters :: DragPartitionParameters{FT}
    "representative canopy height (m), the fallback where no measured height is supplied"
    representative_height :: FT
    "von Kármán constant `ϰ`"
    von_karman_constant :: FT
    "roughness-sublayer influence (constant 0.193; [Raupach 1995](@cite raupach1995corrigenda))"
    sublayer_influence :: FT
    "data-quality ceiling on the leaf area index; a larger index is treated as fill/artifact and gapped"
    maximum_valid_leaf_area_index :: FT
    "fixed-point iterations for the wind ratio `γ`"
    iterations :: Int
end

DragPartitionRoughness(FT = Oceananigans.defaults.FloatType;
                       vegetation_type = :evergreen_broadleaf_forest,
                       parameters = canopy_drag_parameters(FT, vegetation_type),
                       representative_height = representative_canopy_height(FT, vegetation_type),
                       von_karman_constant = 0.4,
                       sublayer_influence = 0.193,             # Raupach (1995)
                       maximum_valid_leaf_area_index = 10,     # data-quality ceiling; larger is fill/artifact
                       iterations = 20) =
    DragPartitionRoughness(parameters, convert(FT, representative_height),
                           convert(FT, von_karman_constant),
                           convert(FT, sublayer_influence),
                           convert(FT, maximum_valid_leaf_area_index), Int(iterations))
