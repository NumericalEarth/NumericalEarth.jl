#####
##### Runoff closures for `VariablySaturatedHydrology`.
#####
##### Two diagnostic categories:
#####
##### * `Rˢᶠᶜ`: surface runoff — liquid leaving the column at the surface.
#####   Returned together with the actual surface liquid flux `Jˡˢ` because
#####   they are coupled (the infiltration-capacity model splits the water
#####   offered at the surface between the two). The hydrology adds saturation
#####   excess (input the remaining pore volume cannot hold) to `Rˢᶠᶜ` outside
#####   the closure.
##### * `Rˡᵃᵗ`: lateral / subsurface runoff — true storage export. Carries
#####   internal energy with it.
#####
##### Each closure implements
#####
#####     surface_liquid_flux_and_runoff(i, j, grid, runoff, Pˡ, M, θˡ, 𝒮, Π, K)
#####         -> (Jˡˢ, Rˢᶠᶜ)
#####
#####     subsurface_runoff(runoff, M, Π, K) -> Rˡᵃᵗ
#####
##### with the sign convention `Jˡˢ > 0` upward, `Pˡ > 0` downward,
##### `Rˢᶠᶜ ≥ 0`, `Rˡᵃᵗ ≥ 0`. A closure with prognostic surface storage
##### declares it through `prognostic_variables` and steps it in
#####
#####     surface_water_balance!(i, j, grid, runoff, prognostic, Pˡ, M, M⁺, θˡ, 𝒮, Π, K, Δt)
#####         -> (Jˡˢ, Rˢᶠᶜ)
#####

"""
    NoRunoff()

No runoff from the closure. All precipitation infiltrates (`Jˡˢ = −Pˡ`) up to
the pore volume the column has left; no subsurface export.
"""
struct NoRunoff end

prognostic_variables(::NoRunoff) = ()

@inline function surface_liquid_flux_and_runoff(i, j, grid, ::NoRunoff, Pˡ, M, θˡ, 𝒮, Π, K)
    return -Pˡ, zero(Pˡ)
end

@inline subsurface_runoff(::NoRunoff, M, Π, K) = zero(M)

Base.summary(::NoRunoff) = "NoRunoff"

"""
    InfiltrationCapacityRunoff(FT = Oceananigans.defaults.FloatType;
                               infiltration_capacity,
                               drainage_timescale = nothing)

Cap the downward infiltration rate at `infiltration_capacity` (kg m⁻² s⁻¹,
positive magnitude; a number or a per-cell `Field`):

```math
J^{ls} = \\max(-P^l, -J^l_{cap}), \\qquad R^{\\mathrm{sfc}} = J^{ls} + P^l \\ge 0.
```

By default the excess `Rˢᶠᶜ` leaves as surface runoff within the step. With a
`drainage_timescale` `τ` (s) it instead ponds in the prognostic surface water store
`Sˢᶠᶜ` (kg m⁻²), which is offered back to infiltration together with the rain each
step and drains to surface runoff as a linear reservoir, `dSˢᶠᶜ/dt = −Sˢᶠᶜ/τ`
(the water ponded during the step decays with it). The store exchanges mass only.

No subsurface runoff.

```jldoctest
julia> using NumericalEarth

julia> summary(InfiltrationCapacityRunoff(infiltration_capacity = 1e-3, drainage_timescale = 3600))
"InfiltrationCapacityRunoff(infiltration_capacity=0.001, drainage_timescale=3600.0)"
```
"""
struct InfiltrationCapacityRunoff{C, T}
    infiltration_capacity :: C
    drainage_timescale    :: T
end

InfiltrationCapacityRunoff(FT::Type = Oceananigans.defaults.FloatType;
                           infiltration_capacity,
                           drainage_timescale = nothing) =
    InfiltrationCapacityRunoff(normalize_property(FT, infiltration_capacity),
                               isnothing(drainage_timescale) ? nothing : convert(FT, drainage_timescale))

Adapt.adapt_structure(to, c::InfiltrationCapacityRunoff) =
    InfiltrationCapacityRunoff(Adapt.adapt(to, c.infiltration_capacity), c.drainage_timescale)

prognostic_variables(c::InfiltrationCapacityRunoff) =
    isnothing(c.drainage_timescale) ? () : (:surface_water_storage,)

@inline function surface_liquid_flux_and_runoff(i, j, grid, c::InfiltrationCapacityRunoff,
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
@inline function pore_limited_surface_flux(Jˡˢ, Rˢᶠᶜ, M, M⁺, Δt)
    pore_limited_flux = -max(M⁺ - M, 0) / Δt
    saturation_excess = max(pore_limited_flux - Jˡˢ, 0)
    return Jˡˢ + saturation_excess, Rˢᶠᶜ + saturation_excess
end

@inline function surface_water_balance!(i, j, grid, runoff, prognostic, Pˡ, M, M⁺, θˡ, 𝒮, Π, K, Δt)
    Jˡˢ, Rˢᶠᶜ = surface_liquid_flux_and_runoff(i, j, grid, runoff, Pˡ, M, θˡ, 𝒮, Π, K)
    return pore_limited_surface_flux(Jˡˢ, Rˢᶠᶜ, M, M⁺, Δt)
end

# The pond is offered to infiltration with the rain; the excess ponds again, and the
# share `f` of it that survives the step's linear drain stays in the store.
@inline function surface_water_balance!(i, j, grid, c::InfiltrationCapacityRunoff{<:Any, <:Number},
                                        prognostic, Pˡ, M, M⁺, θˡ, 𝒮, Π, K, Δt)
    S = prognostic.surface_water_storage
    @inbounds Sⁿ = S[i, j, 1]
    Jˡˢ, Rˢᶠᶜ = surface_liquid_flux_and_runoff(i, j, grid, c, Pˡ + Sⁿ / Δt, M, θˡ, 𝒮, Π, K)
    Jˡˢ, Rˢᶠᶜ = pore_limited_surface_flux(Jˡˢ, Rˢᶠᶜ, M, M⁺, Δt)
    f = exp(-Δt / c.drainage_timescale)
    @inbounds S[i, j, 1] = f * Δt * Rˢᶠᶜ
    return Jˡˢ, (1 - f) * Rˢᶠᶜ
end

Base.summary(c::InfiltrationCapacityRunoff) =
    string("InfiltrationCapacityRunoff(infiltration_capacity=", prettysummary(c.infiltration_capacity),
           isnothing(c.drainage_timescale) ? "" : string(", drainage_timescale=", prettysummary(c.drainage_timescale)),
           ")")
