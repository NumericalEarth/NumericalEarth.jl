#####
##### Runoff closures for `VariablySaturatedHydrology`.
#####
##### Two diagnostic categories:
#####
##### * `Rᵒᵛᵉʳ`: surface runoff — liquid leaving the column at the surface.
#####   Returned together with the actual surface liquid water flux `Jˡˢ` because
#####   they are coupled (the infiltration-capacity model splits the water
#####   offered at the surface between the two). The hydrology adds saturation
#####   excess (input the remaining pore volume cannot hold) to `Rᵒᵛᵉʳ` outside
#####   the closure.
##### * `Rˡᵃᵗ`: lateral / subsurface runoff — true storage export. Carries
#####   internal energy with it.
#####
##### Each closure implements
#####
#####     surface_liquid_water_flux_and_runoff(i, j, grid, runoff, Pˡ, M, θˡ, 𝒮, Π, K)
#####         -> (Jˡˢ, Rᵒᵛᵉʳ)
#####
#####     subsurface_runoff(runoff, M, Π, K) -> Rˡᵃᵗ
#####
##### with the sign convention `Jˡˢ > 0` upward, `Pˡ > 0` downward,
##### `Rᵒᵛᵉʳ ≥ 0`, `Rˡᵃᵗ ≥ 0`. A closure with prognostic surface storage
##### declares it through `prognostic_variables` and steps it in
#####
#####     surface_water_balance!(i, j, grid, runoff, prognostic, Pˡ, M, M⁺, θˡ, 𝒮, Π, K, Δt)
#####         -> (Jˡˢ, Rᵒᵛᵉʳ)
#####

"""
    NoRunoff()

No runoff from the closure. All precipitation infiltrates (`Jˡˢ = −Pˡ`) up to
the pore volume the column has left; no subsurface export.
"""
struct NoRunoff end

prognostic_variables(::NoRunoff) = ()

@inline function surface_liquid_water_flux_and_runoff(i, j, grid, ::NoRunoff, Pˡ, M, θˡ, 𝒮, Π, K)
    return -Pˡ, zero(Pˡ)
end

@inline subsurface_runoff(::NoRunoff, M, Π, K) = zero(M)

Base.summary(::NoRunoff) = "NoRunoff"

"""
    InfiltrationCapacityRunoff(FT = Oceananigans.defaults.FloatType;
                               infiltration_capacity,
                               drainage_time_scale = nothing)

Cap the downward infiltration rate at `infiltration_capacity` (kg m⁻² s⁻¹,
positive magnitude; a number or a per-cell `Field`):

```math
J^{ls} = \\max(-P^l, -J^l_{cap}), \\qquad R^{\\mathrm{over}} = J^{ls} + P^l \\ge 0.
```

By default the excess `Rᵒᵛᵉʳ` leaves as surface runoff within the step. With a
`drainage_time_scale` `τ` (s) it instead ponds in the prognostic surface water store
`Mᵖᵒⁿᵈ` (kg m⁻²), which infiltrates at capacity while it stands and drains to surface
runoff as a linear reservoir, `dMᵖᵒⁿᵈ/dt = Pˡ − I − Mᵖᵒⁿᵈ/τ`, integrated exactly over each
step. The store exchanges mass only.

No subsurface runoff.

```jldoctest
julia> using NumericalEarth

julia> summary(InfiltrationCapacityRunoff(infiltration_capacity = 1e-3, drainage_time_scale = 3600))
"InfiltrationCapacityRunoff(infiltration_capacity=0.001, drainage_time_scale=3600.0)"
```
"""
struct InfiltrationCapacityRunoff{C, T}
    infiltration_capacity :: C
    drainage_time_scale   :: T
end

InfiltrationCapacityRunoff(FT::Type = Oceananigans.defaults.FloatType;
                           infiltration_capacity,
                           drainage_time_scale = nothing) =
    InfiltrationCapacityRunoff(normalize_property(FT, infiltration_capacity),
                               isnothing(drainage_time_scale) ? nothing : convert(FT, drainage_time_scale))

Adapt.adapt_structure(to, c::InfiltrationCapacityRunoff) =
    InfiltrationCapacityRunoff(Adapt.adapt(to, c.infiltration_capacity), c.drainage_time_scale)

prognostic_variables(c::InfiltrationCapacityRunoff) =
    isnothing(c.drainage_time_scale) ? () : (:surface_water_storage,)

@inline function surface_liquid_water_flux_and_runoff(i, j, grid, c::InfiltrationCapacityRunoff,
                                                      Pˡ, M, θˡ, 𝒮, Π, K)
    FT   = typeof(Pˡ)
    Jcap = convert(FT, stateindex(c.infiltration_capacity, i, j, 1))
    # Available downward flux is -Pˡ. Cap its downward magnitude at Jcap.
    Jˡs  = max(-Pˡ, -Jcap)
    Rsfc = Jˡs - (-Pˡ)   # ≥ 0
    return Jˡs, Rsfc
end

@inline subsurface_runoff(::InfiltrationCapacityRunoff, M, Π, K) = zero(M)

# Infiltration cannot exceed the pore volume `M⁺ − M` the top layer has left; the surplus
# is saturation-excess runoff.
@inline function pore_limited_surface_flux(Jˡˢ, Rᵒᵛᵉʳ, M, M⁺, Δt)
    pore_limited_flux = -max(M⁺ - M, 0) / Δt
    saturation_excess = max(pore_limited_flux - Jˡˢ, 0)
    return Jˡˢ + saturation_excess, Rᵒᵛᵉʳ + saturation_excess
end

@inline function surface_water_balance!(i, j, grid, runoff, prognostic, Pˡ, M, M⁺, θˡ, 𝒮, Π, K, Δt)
    Jˡˢ, Rᵒᵛᵉʳ = surface_liquid_water_flux_and_runoff(i, j, grid, runoff, Pˡ, M, θˡ, 𝒮, Π, K)
    return pore_limited_surface_flux(Jˡˢ, Rᵒᵛᵉʳ, M, M⁺, Δt)
end

# Exact solution of the pond balance dMᵖᵒⁿᵈ/dt = Pˡ − I − Mᵖᵒⁿᵈ/τᵖᵒⁿᵈ over the step. Infiltration is
# I = Iᶜᵃᵖ while a pond stands and I = Iᵈʳʸ = min(Pˡ, Iᶜᵃᵖ) on unponded ground, where the capacity
# Iᶜᵃᵖ is limited by the pore volume M⁺ − M the slab has left. With the net pond inflow
# Jᵖᵒⁿᵈ = Pˡ − I and its equilibrium pond Mᵉ = τᵖᵒⁿᵈ Jᵖᵒⁿᵈ,
#
#     Mᵖᵒⁿᵈ(t) = Mᵉ + (Mᵖᵒⁿᵈⁿ − Mᵉ) e^{−t/τᵖᵒⁿᵈ}.
#
# If the pond empties, at t⁰ = τᵖᵒⁿᵈ log((Mᵖᵒⁿᵈⁿ − Mᵉ)/(−Mᵉ)) < Δt, the rest of the step infiltrates Iᵈʳʸ.
# The step-mean runoff Rᵒᵛᵉʳ (the drained ∫ Mᵖᵒⁿᵈ/τᵖᵒⁿᵈ dt over Δt) and infiltration Ī close the balance
# Pˡ Δt + Mᵖᵒⁿᵈⁿ = (Ī + Rᵒᵛᵉʳ) Δt + Mᵖᵒⁿᵈⁿ⁺¹.
@inline function surface_water_balance!(i, j, grid, c::InfiltrationCapacityRunoff{<:Any, <:Number},
                                        prognostic, Pˡ, M, M⁺, θˡ, 𝒮, Π, K, Δt)
    FT   = typeof(Pˡ)
    Mᵖᵒⁿᵈ = prognostic.surface_water_storage
    @inbounds Mᵖᵒⁿᵈⁿ = Mᵖᵒⁿᵈ[i, j, 1]
    τᵖᵒⁿᵈ = convert(FT, c.drainage_time_scale)

    Iᶜᵃᵖ = min(convert(FT, stateindex(c.infiltration_capacity, i, j, 1)), max(M⁺ - M, 0) / Δt)
    Iᵈʳʸ = min(Pˡ, Iᶜᵃᵖ)
    I    = ifelse(Mᵖᵒⁿᵈⁿ > 0, Iᶜᵃᵖ, Iᵈʳʸ)
    Jᵖᵒⁿᵈ = Pˡ - I
    Mᵉ   = τᵖᵒⁿᵈ * Jᵖᵒⁿᵈ
    M̃ᵖᵒⁿᵈ = Mᵉ + (Mᵖᵒⁿᵈⁿ - Mᵉ) * exp(-Δt / τᵖᵒⁿᵈ)

    empties = M̃ᵖᵒⁿᵈ < 0
    ratio   = ifelse(empties, (Mᵖᵒⁿᵈⁿ - Mᵉ) / -Mᵉ, one(FT))
    t⁰      = ifelse(empties, τᵖᵒⁿᵈ * log(ratio), Δt)

    Mᵖᵒⁿᵈⁿ⁺¹ = max(M̃ᵖᵒⁿᵈ, 0)
    Rᵒᵛᵉʳ    = (Jᵖᵒⁿᵈ * t⁰ + Mᵖᵒⁿᵈⁿ - Mᵖᵒⁿᵈⁿ⁺¹) / Δt
    Ī       = (I * t⁰ + Iᵈʳʸ * (Δt - t⁰)) / Δt

    @inbounds Mᵖᵒⁿᵈ[i, j, 1] = Mᵖᵒⁿᵈⁿ⁺¹
    return -Ī, Rᵒᵛᵉʳ
end

Base.summary(c::InfiltrationCapacityRunoff) =
    string("InfiltrationCapacityRunoff(infiltration_capacity=", prettysummary(c.infiltration_capacity),
           isnothing(c.drainage_time_scale) ? "" : string(", drainage_time_scale=", prettysummary(c.drainage_time_scale)),
           ")")
