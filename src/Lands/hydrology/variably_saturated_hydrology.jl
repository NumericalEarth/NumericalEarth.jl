#####
##### `VariablySaturatedHydrology` — conservative variably saturated soil column.
#####
##### One or more slab layers, surface first. Layer `k` holds water `Mₖ` (kg m⁻²) as the
##### augmented liquid fraction `ϑₖ = Mₖ/(ρˡ hₖ)`, so `Mₖ > Mₖ⁺` is admitted as saturated
##### positive-pressure storage (`Πₖ > 0`). With every flux positive upward,
#####
#####     dMₖ/dt = Jₖ − Jₖ₋₁ − fₖ Jᵛ − δₖ₁ (Jˡˢ + Rˡᵃᵗ),     J₀ ≡ 0,
#####
##### where `Jₖ` is the Darcy flux across the bottom of layer `k` (the `deep_liquid_flux`
##### closure under the last layer) and `fₖ ∝ rₖ 𝒮ₖ` is the root-weighted share of the
##### vapor sink. The interlayer exchanges and a Darcy bottom closure are linearized in
##### head and stepped implicitly; surface fluxes and the other bottom closures are explicit.
#####
##### Diagnostics published every step:
#####   * `deep_liquid_flux`          (Jₙ, across the column bottom)
#####   * `interlayer_liquid_flux_k`  (Jₖ for k < N; several layers only)
#####   * `surface_liquid_flux`       (Jˡˢ = −Pˡ + Rˢᶠᶜ)
#####   * `surface_runoff`            (Rˢᶠᶜ ≥ 0)
#####   * `subsurface_runoff`         (Rˡᵃᵗ ≥ 0, from the top layer)
#####   * `water_storage_tendency`    (realized dM₁/dt)
#####
##### `land.saturation` is the root-weighted `Σ rₖ 𝒮ₖ`; the interface humidity formulation
##### maps it to the evaporation efficiency `β`.
#####

"""
    VariablySaturatedHydrology(FT = Oceananigans.defaults.FloatType;
                               slab_depth,
                               porosity,
                               residual_liquid_fraction = 0,
                               storage_height,
                               liquid_density = 1000,
                               retention_curve,
                               hydraulic_conductivity,
                               deep_liquid_flux = NoDeepLiquidFlux(),
                               deep_pressure_head = 0,
                               runoff = NoRunoff(),
                               root_fraction = nothing)

Conservative variably saturated soil column of one or more slab layers with augmented
liquid fractions `ϑₖ`. `slab_depth` is one layer thickness (m) or a tuple of them, surface
first. The top layer's water is the container's `water_storage`; deeper layers are the
prognostics `water_storage_2`, `water_storage_3`, …, which start at half effective saturation.

Neighboring layers exchange water by Darcy's law across the distance `ℓ = (hₖ + hₖ₊₁)/2`
between their centers, `Jₖ = ρˡ K̄ (Πₖ₊₁ − Πₖ − ℓ)/ℓ` with `K̄` the conductivity at their
mean saturation, positive upward. The last layer exchanges with `deep_liquid_flux`; the
surface fluxes and `runoff` act on the first. The vapor sink is drawn from layer `k` in
proportion to `rₖ 𝒮ₖ`, and the `saturation` the atmosphere reads is `Σ rₖ 𝒮ₖ`. The interlayer
exchanges and a Darcy bottom closure are linearized in head and stepped implicitly through a
tridiagonal solve per column; surface fluxes and the other bottom closures are explicit. The
interface conductivity, the root-weighted uptake, and the linearized step follow the Community
Land Model [Oleson et al. (2013)](@cite oleson2013clm45).

`slab_depth`, `porosity`, `residual_liquid_fraction`, and `storage_height` each accept a
scalar (uniform) or a `Field` that varies grid point by grid point.

* `slab_depth` (`hₖ`, m) — layer thickness, or a tuple of thicknesses.
* `porosity` (`ν`) — total pore fraction.
* `residual_liquid_fraction` (`θʳ`) — minimum pore liquid (default 0).
* `storage_height` (`hˢˢ`, m) — saturated storage height; the reciprocal of the
  specific storage (`1/Sₛ`), i.e. the pressure head built per unit fractional
  over-saturation.
* `retention_curve` — e.g. [`VanGenuchtenRetention`](@ref).
* `hydraulic_conductivity` — e.g. [`VanGenuchtenConductivity`](@ref), or a tuple of one
  per layer.
* `deep_liquid_flux` — bottom-boundary closure: [`NoDeepLiquidFlux`](@ref),
  [`FreeDrainageFlux`](@ref), [`DarcyDeepLiquidFlux`](@ref), or
  [`LinearReservoirDrainage`](@ref).
* `deep_pressure_head` — the deep-reservoir pressure head (m) under the last layer,
  passed to the deep-flux closure as `Πᵈ`. Default 0.
* `runoff` — runoff closure: [`NoRunoff`](@ref) or
  [`InfiltrationCapacityRunoff`](@ref).
* `root_fraction` (`rₖ`) — root share of each layer, summing to 1. Default: all in the
  top layer.
"""
struct VariablySaturatedHydrology{FT, SD, P, RL, SH, R, C, DF, PD, RO, RF} <: AbstractHydrology
    slab_depth               :: SD
    porosity                 :: P
    residual_liquid_fraction :: RL
    storage_height           :: SH
    liquid_density           :: FT
    retention_curve          :: R
    hydraulic_conductivity   :: C
    deep_liquid_flux         :: DF
    deep_pressure_head       :: PD
    runoff                   :: RO
    root_fraction            :: RF
end

Adapt.adapt_structure(to, h::VariablySaturatedHydrology) =
    VariablySaturatedHydrology(Adapt.adapt(to, h.slab_depth),
                               Adapt.adapt(to, h.porosity),
                               Adapt.adapt(to, h.residual_liquid_fraction),
                               Adapt.adapt(to, h.storage_height),
                               Adapt.adapt(to, h.liquid_density),
                               Adapt.adapt(to, h.retention_curve),
                               Adapt.adapt(to, h.hydraulic_conductivity),
                               Adapt.adapt(to, h.deep_liquid_flux),
                               Adapt.adapt(to, h.deep_pressure_head),
                               Adapt.adapt(to, h.runoff),
                               h.root_fraction)

function VariablySaturatedHydrology(FT::Type = Oceananigans.defaults.FloatType;
                                    slab_depth,
                                    porosity,
                                    residual_liquid_fraction = 0,
                                    storage_height,
                                    liquid_density = 1000,
                                    retention_curve,
                                    hydraulic_conductivity,
                                    deep_liquid_flux = NoDeepLiquidFlux(),
                                    deep_pressure_head = 0,
                                    runoff = NoRunoff(),
                                    root_fraction = nothing)
    N = slab_depth isa Tuple ? length(slab_depth) : 1
    r = isnothing(root_fraction) ? ntuple(k -> FT(k == 1), N) : map(x -> convert(FT, x), Tuple(root_fraction))
    length(r) == N || throw(ArgumentError("root_fraction needs one entry per layer of slab_depth"))
    sum(r) ≈ 1 || throw(ArgumentError("root_fraction must sum to 1"))
    hydraulic_conductivity isa Tuple && length(hydraulic_conductivity) != N &&
        throw(ArgumentError("hydraulic_conductivity needs one closure per layer of slab_depth"))
    if porosity isa Number && residual_liquid_fraction isa Number
        porosity > residual_liquid_fraction ||
            throw(ArgumentError("porosity must exceed residual_liquid_fraction"))
    end
    thickness = slab_depth isa Tuple ? map(x -> normalize_property(FT, x), slab_depth) :
                                       normalize_property(FT, slab_depth)
    return VariablySaturatedHydrology(thickness,
                                      normalize_property(FT, porosity),
                                      normalize_property(FT, residual_liquid_fraction),
                                      normalize_property(FT, storage_height),
                                      convert(FT, liquid_density),
                                      retention_curve,
                                      hydraulic_conductivity,
                                      deep_liquid_flux,
                                      normalize_property(FT, deep_pressure_head),
                                      runoff,
                                      r)
end

@inline layer(x::Tuple, k) = @inbounds x[k]
@inline layer(x, k) = x

layer_storage_names(h) = ntuple(k -> Symbol(:water_storage_, k + 1), length(h.root_fraction) - 1)
interlayer_flux_names(h) = ntuple(k -> Symbol(:interlayer_liquid_flux_, k), length(h.root_fraction) - 1)

# Coupler writes the signed vapor flux `Jᵛ` (positive upward) and the rainfall `Pˡ`
# (positive downward) into `land.fluxes`.
flux_variables(::VariablySaturatedHydrology) = (:vapor_flux, :liquid_precipitation_flux)

prognostic_variables(h::VariablySaturatedHydrology) =
    merge_unique(prognostic_variables(h.runoff), layer_storage_names(h))

diagnostic_variables(h::VariablySaturatedHydrology) =
    (:deep_liquid_flux, interlayer_flux_names(h)...,
     :surface_liquid_flux, :surface_runoff, :subsurface_runoff, :water_storage_tendency)

function initial_prognostic(h::VariablySaturatedHydrology, name::Symbol, grid)
    field = CenterField(grid)
    k = findfirst(==(name), layer_storage_names(h))
    isnothing(k) || Oceananigans.set!(field, h.liquid_density * layer(h.slab_depth, k + 1) *
                                             (h.porosity + h.residual_liquid_fraction) / 2)
    return field
end

layer_storages(h, land) =
    (land.water_storage, map(name -> getproperty(land.prognostic, name), layer_storage_names(h))...)

interlayer_fluxes(h, land) = map(name -> getproperty(land.diagnostics, name), interlayer_flux_names(h))

#####
##### Per-layer state at one cell
#####

# θ, 𝒮, pressure head Π, its storage derivative dΠ/dM, and K of layer `k` holding water `M`.
@inline function layer_state(h, M, k, i, j)
    FT   = typeof(M)
    hₗ   = convert(FT, property_value(layer(h.slab_depth, k), i, j))
    ν    = convert(FT, property_value(h.porosity, i, j))
    θʳ   = convert(FT, property_value(h.residual_liquid_fraction, i, j))
    hˢˢ  = convert(FT, property_value(h.storage_height, i, j))
    ρˡhₗ = convert(FT, h.liquid_density) * hₗ
    M⁺   = ν * ρˡhₗ
    θ    = min(M / ρˡhₗ, ν)
    𝒮    = clamp((θ - θʳ) / (ν - θʳ), zero(FT), one(FT))
    saturated = M ≥ M⁺
    Π    = ifelse(saturated, (M - M⁺) * hˢˢ / ρˡhₗ, pressure_head(h.retention_curve, 𝒮))
    dΠdM = ifelse(saturated, hˢˢ, pressure_head_derivative(h.retention_curve, 𝒮) / (ν - θʳ)) / ρˡhₗ
    K    = hydraulic_conductivity(layer(h.hydraulic_conductivity, k), 𝒮)
    return (; θ, 𝒮, Π, dΠdM, K)
end

@inline root_weighted_saturation(h, M, i, j) =
    sum(ntuple(k -> h.root_fraction[k] * layer_state(h, M[k], k, i, j).𝒮, Val(length(M))))

# Conductance Λ = ρˡ K̄/ℓ and explicit Darcy flux between layers k and k+1, a distance
# ℓ = (hₖ + hₖ₊₁)/2 apart, with K̄ at their mean saturation (Oleson et al. 2013, eq. 7.89).
@inline function interlayer_exchange(h, s, k, i, j)
    FT = typeof(s[k].Π)
    ℓ  = (convert(FT, property_value(layer(h.slab_depth, k), i, j)) +
          convert(FT, property_value(layer(h.slab_depth, k + 1), i, j))) / 2
    𝒮̄  = (s[k].𝒮 + s[k+1].𝒮) / 2
    K̄  = (hydraulic_conductivity(layer(h.hydraulic_conductivity, k), 𝒮̄) +
          hydraulic_conductivity(layer(h.hydraulic_conductivity, k + 1), 𝒮̄)) / 2
    Λ  = convert(FT, h.liquid_density) * K̄ / ℓ
    return Λ, Λ * (s[k+1].Π - s[k].Π - ℓ)
end

# Thomas algorithm on tuples; `lower[k]` couples row k+1 to k and `upper[k]` row k to k+1.
@inline function solve_tridiagonal(lower, diagonal, upper, rhs)
    N = length(diagonal)
    b = diagonal
    d = rhs
    for k in 2:N
        w = lower[k-1] / b[k-1]
        b = Base.setindex(b, b[k] - w * upper[k-1], k)
        d = Base.setindex(d, d[k] - w * d[k-1], k)
    end
    x = Base.setindex(d, d[N] / b[N], N)
    for k in N-1:-1:1
        x = Base.setindex(x, (d[k] - upper[k] * x[k+1]) / b[k], k)
    end
    return x
end

#####
##### Kernels
#####

@kernel function _variably_saturated_saturation!(saturation, M, h)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Mᵢⱼ = ntuple(k -> M[k][i, j, 1], Val(length(M)))
        saturation[i, j, 1] = root_weighted_saturation(h, Mᵢⱼ, i, j)
    end
end

@kernel function _variably_saturated_step!(M, saturation, interlayer_flux_fields, diagnostics,
                                           Jᵛ, Pˡ, prognostic, h, deep_pressure_head, Δt, grid, time)
    i, j = @index(Global, NTuple)
    N = length(M)
    @inbounds begin
        Mⁿ   = ntuple(k -> M[k][i, j, 1], Val(N))
        Jᵛᵢⱼ = Jᵛ[i, j, 1]
        Pˡᵢⱼ = Pˡ[i, j, 1]
        Πᵈ   = stateindex(deep_pressure_head, i, j, 1, grid, time, (Center, Center, Center))
    end
    FT = typeof(Jᵛᵢⱼ)
    δt = convert(FT, Δt)
    s  = ntuple(k -> layer_state(h, Mⁿ[k], k, i, j), Val(N))
    s₁, sₙ = s[1], s[end]

    # Surface fluxes on the top layer, the deep closure under the last, Darcy exchanges between.
    Jˡˢ, Rˢᶠᶜ = surface_water_balance!(i, j, h.runoff, prognostic, Pˡᵢⱼ, Mⁿ[1], s₁.θ, s₁.𝒮, s₁.Π, s₁.K, δt)
    Rˡᵃᵗ      = subsurface_runoff(h.runoff, Mⁿ[1], s₁.Π, s₁.K)
    Jᵇ        = deep_liquid_flux(h.deep_liquid_flux, Mⁿ[end], sₙ.θ, sₙ.𝒮, sₙ.Π, sₙ.K, Πᵈ, time)
    exchanges = ntuple(k -> interlayer_exchange(h, s, k, i, j), Val(N - 1))
    Λ  = (map(first, exchanges)..., -deep_liquid_flux_head_derivative(h.deep_liquid_flux, sₙ.K))
    J⁰ = (map(last, exchanges)..., Jᵇ)

    # Root-weighted share of the vapor sink, fₖ = rₖ 𝒮ₖ / Σⱼ rⱼ 𝒮ⱼ (Oleson et al. 2013, eqs. 7.107 and
    # 8.26); all from the top layer of a dry column.
    w   = map(*, h.root_fraction, map(x -> x.𝒮, s))
    Σw  = sum(w)
    wet = Σw > 0
    f   = ntuple(k -> ifelse(wet, w[k] / ifelse(wet, Σw, one(FT)), FT(k == 1)), Val(N))

    # Explicit tendencies Fₖ, then the head-linearized implicit correction of the Darcy
    # exchanges, ΔMₖ − δt Σⱼ (∂Fₖ/∂Mⱼ) ΔMⱼ = δt Fₖ, tridiagonal in the layers (Oleson et al.
    # 2013, eq. 7.110).
    flux_above        = (zero(FT), Base.front(J⁰)...)
    conductance_above = (zero(FT), Base.front(Λ)...)
    sink  = (Jˡˢ + Rˡᵃᵗ, ntuple(_ -> zero(FT), Val(N - 1))...)
    F     = ntuple(k -> J⁰[k] - flux_above[k] - f[k] * Jᵛᵢⱼ - sink[k], Val(N))
    dΠdM  = map(x -> x.dΠdM, s)
    lower = ntuple(k -> -δt * Λ[k] * dΠdM[k], Val(N - 1))
    diag  = ntuple(k -> 1 + δt * dΠdM[k] * (Λ[k] + conductance_above[k]), Val(N))
    upper = ntuple(k -> -δt * Λ[k] * dΠdM[k+1], Val(N - 1))
    ΔM    = solve_tridiagonal(lower, diag, upper, map(x -> δt * x, F))

    # Fluxes realized by the implicit step; the fixed deep head under the last layer does not move.
    storage_change_below = (Base.tail(ΔM)..., zero(FT))
    head_slope_below     = (Base.tail(dΠdM)..., zero(FT))
    J     = ntuple(k -> J⁰[k] + Λ[k] * (head_slope_below[k] * storage_change_below[k] - dΠdM[k] * ΔM[k]), Val(N))
    Mⁿ⁺¹  = map((m, δ) -> max(m + δ, 0), Mⁿ, ΔM)

    @inbounds begin
        for k in 1:N
            M[k][i, j, 1] = Mⁿ⁺¹[k]
        end
        for k in 1:N-1
            interlayer_flux_fields[k][i, j, 1] = J[k]
        end
        diagnostics.deep_liquid_flux[i, j, 1]       = J[end]
        diagnostics.surface_liquid_flux[i, j, 1]    = Jˡˢ
        diagnostics.surface_runoff[i, j, 1]         = Rˢᶠᶜ
        diagnostics.subsurface_runoff[i, j, 1]      = Rˡᵃᵗ
        diagnostics.water_storage_tendency[i, j, 1] = (Mⁿ⁺¹[1] - Mⁿ[1]) / δt
        saturation[i, j, 1] = root_weighted_saturation(h, Mⁿ⁺¹, i, j)
    end
end

function time_step!(h::VariablySaturatedHydrology, land, Δt, time)
    launch!(architecture(land.grid), land.grid, :xy, _variably_saturated_step!,
            layer_storages(h, land), land.saturation, interlayer_fluxes(h, land), land.diagnostics,
            land.fluxes.vapor_flux, land.fluxes.liquid_precipitation_flux, land.prognostic,
            h, h.deep_pressure_head, Δt, land.grid, time)
    return nothing
end

function update_diagnostics!(h::VariablySaturatedHydrology, land)
    launch!(architecture(land.grid), land.grid, :xy, _variably_saturated_saturation!,
            land.saturation, layer_storages(h, land), h)
    return nothing
end

saturation(h::VariablySaturatedHydrology, land) = land.saturation

EarthSystemModels.surface_retention_curve(h::VariablySaturatedHydrology) = h.retention_curve

Base.summary(h::VariablySaturatedHydrology) =
    string("VariablySaturatedHydrology(",
           "slab_depth=", h.slab_depth isa Tuple ? string("(", join(map(prettysummary, h.slab_depth), ", "), ")") :
                                                   prettysummary(h.slab_depth),
           ", porosity=", prettysummary(h.porosity),
           ", retention=", summary(h.retention_curve),
           ", deep=", summary(h.deep_liquid_flux),
           ", runoff=", summary(h.runoff), ")")
