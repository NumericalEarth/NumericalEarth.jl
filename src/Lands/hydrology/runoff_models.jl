#####
##### Runoff closures for `VariablySaturatedHydrology`.
#####
##### Two diagnostic categories:
#####
##### * `Rˢᶠᶜ`: surface runoff — rejected liquid input. Returned together with
#####   the actual surface liquid flux `Jˡˢ` because they are coupled (the
#####   infiltration-capacity model splits incoming precipitation between the
#####   two).
##### * `Rˡᵃᵗ`: lateral / subsurface runoff — true storage export. Carries
#####   internal energy with it.
#####
##### Each closure implements
#####
#####     surface_liquid_flux_and_runoff(runoff, Pˡ, M, θˡ, 𝒮, Π, K, i, j)
#####         -> (Jˡˢ, Rˢᶠᶜ)
#####
#####     subsurface_runoff(runoff, M, Π, K) -> Rˡᵃᵗ
#####
##### with the sign convention `Jˡˢ > 0` upward, `Pˡ > 0` downward,
##### `Rˢᶠᶜ ≥ 0`, `Rˡᵃᵗ ≥ 0`.
#####

"""
    NoRunoff()

No runoff. All precipitation infiltrates (`Jˡˢ = −Pˡ`), no subsurface export.
"""
struct NoRunoff end

@inline function surface_liquid_flux_and_runoff(::NoRunoff, Pˡ, M, θˡ, 𝒮, Π, K, i, j)
    return -Pˡ, zero(Pˡ)
end

@inline subsurface_runoff(::NoRunoff, M, Π, K) = zero(M)

Base.summary(::NoRunoff) = "NoRunoff"

"""
    InfiltrationCapacityRunoff(infiltration_capacity)

Cap the downward infiltration rate at `infiltration_capacity` (kg m⁻² s⁻¹,
positive magnitude; a number or a per-cell `Field`). Any precipitation exceeding
the cap becomes surface runoff:

```math
J^{ls} = \\max(-P^l, -J^l_{cap}), \\qquad R^{\\mathrm{sfc}} = J^{ls} - (-P^l) \\ge 0.
```

No subsurface runoff.
"""
struct InfiltrationCapacityRunoff{C}
    infiltration_capacity :: C
end

InfiltrationCapacityRunoff(FT::Type = Oceananigans.defaults.FloatType;
                           infiltration_capacity) =
    InfiltrationCapacityRunoff(normalize_property(FT, infiltration_capacity))

Adapt.adapt_structure(to, c::InfiltrationCapacityRunoff) =
    InfiltrationCapacityRunoff(Adapt.adapt(to, c.infiltration_capacity))

@inline function surface_liquid_flux_and_runoff(c::InfiltrationCapacityRunoff,
                                                Pˡ, M, θˡ, 𝒮, Π, K, i, j)
    FT   = typeof(Pˡ)
    Jcap = convert(FT, property_value(c.infiltration_capacity, i, j))
    # Available downward flux is -Pˡ. Cap its downward magnitude at Jcap.
    Jˡs  = max(-Pˡ, -Jcap)
    Rsfc = Jˡs - (-Pˡ)   # ≥ 0
    return Jˡs, Rsfc
end

@inline subsurface_runoff(::InfiltrationCapacityRunoff, M, Π, K) = zero(M)

Base.summary(c::InfiltrationCapacityRunoff) =
    string("InfiltrationCapacityRunoff(infiltration_capacity=",
           prettysummary(c.infiltration_capacity), ")")
