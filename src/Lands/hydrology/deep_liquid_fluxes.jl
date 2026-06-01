#####
##### Deep liquid flux closures for `VariablySaturatedBucketHydrology`.
#####
##### Each closure implements
#####
#####     deep_liquid_flux(closure, M, θˡ, 𝒮, Π, K, ψ_D, time) -> Jˡ_b  (kg m⁻² s⁻¹)
#####
##### with `Jˡ_b` positive upward (capillary rise / groundwater return) and
##### negative downward (drainage). All scalars are per-cell; the kernel
##### passes them in.
#####

"""
    NoDeepLiquidFlux()

Zero deep liquid flux. The slab is closed at the bottom.
"""
struct NoDeepLiquidFlux end

@inline deep_liquid_flux(::NoDeepLiquidFlux, M, θˡ, 𝒮, Π, K, ψ_D, time) = zero(M)

Base.summary(::NoDeepLiquidFlux) = "NoDeepLiquidFlux"

"""
    FreeDrainageFlux(ρˡ)

Free-drainage bottom boundary. `∂z Π = 0` so `∂z h = 1` and `Jˡ = -ρˡ K`.
`ρˡ` is the intrinsic liquid-water density (kg m⁻³).
"""
struct FreeDrainageFlux{FT}
    liquid_density :: FT
end

FreeDrainageFlux(FT::Type = Oceananigans.defaults.FloatType; liquid_density = 1000) =
    FreeDrainageFlux(convert(FT, liquid_density))

@inline deep_liquid_flux(c::FreeDrainageFlux, M, θˡ, 𝒮, Π, K, ψ_D, time) =
    -convert(typeof(M), c.liquid_density) * K

Base.summary(c::FreeDrainageFlux) =
    string("FreeDrainageFlux(liquid_density=", prettysummary(c.liquid_density), ")")

"""
    DarcyDeepLiquidFlux(exchange_length, deep_pressure_head, deep_depth, liquid_density)

Darcy exchange between the slab bottom and a deep reservoir held at pressure
head `Π_D`. The exchange length `ℓ_l` (m) separates the slab bottom from the
deep reservoir; the deep height is `z_D = z_b − ℓ_l`. With hydraulic heads
`h_b = z_b + Π_b` and `h_D = z_D + Π_D`,

```math
J^l_b = \\rho^l K_b\\,\\frac{h_D - h_b}{\\ell_l}.
```

`Π_D` may be a scalar, `Field`, or `FieldTimeSeries`.
"""
struct DarcyDeepLiquidFlux{FT, H}
    exchange_length    :: FT
    deep_pressure_head :: H
    liquid_density     :: FT
end

DarcyDeepLiquidFlux(FT::Type = Oceananigans.defaults.FloatType;
                    exchange_length, deep_pressure_head, liquid_density = 1000) =
    DarcyDeepLiquidFlux(convert(FT, exchange_length),
                        normalize_property(FT, deep_pressure_head),
                        convert(FT, liquid_density))

@inline function deep_liquid_flux(c::DarcyDeepLiquidFlux, M, θˡ, 𝒮, Π, K, ψ_D, time)
    FT = typeof(M)
    ℓ  = convert(FT, c.exchange_length)
    ρˡ = convert(FT, c.liquid_density)
    # h_D − h_b = (z_D + Π_D) − (z_b + Π) = -ℓ + Π_D − Π
    return ρˡ * K * (ψ_D - Π - ℓ) / ℓ
end

Base.summary(c::DarcyDeepLiquidFlux) =
    string("DarcyDeepLiquidFlux(exchange_length=", prettysummary(c.exchange_length),
           ", deep_pressure_head=", prettysummary(c.deep_pressure_head),
           ", liquid_density=", prettysummary(c.liquid_density), ")")

"""
    LinearReservoirDrainage(drainage_time_scale, equilibrium_storage)

Cheap linear-reservoir drainage:

```math
J^l_b = -\\frac{\\max(M^{la} - M_{eq}, 0)}{\\tau_l}.
```

Storage above `equilibrium_storage` drains out the bottom on time scale `τ_l`.
"""
struct LinearReservoirDrainage{FT}
    drainage_time_scale :: FT
    equilibrium_storage :: FT
end

LinearReservoirDrainage(FT::Type = Oceananigans.defaults.FloatType;
                        drainage_time_scale, equilibrium_storage = 0) =
    LinearReservoirDrainage(convert(FT, drainage_time_scale),
                            convert(FT, equilibrium_storage))

@inline function deep_liquid_flux(c::LinearReservoirDrainage, M, θˡ, 𝒮, Π, K, ψ_D, time)
    FT = typeof(M)
    return -max(M - convert(FT, c.equilibrium_storage), zero(FT)) / convert(FT, c.drainage_time_scale)
end

Base.summary(c::LinearReservoirDrainage) =
    string("LinearReservoirDrainage(drainage_time_scale=", prettysummary(c.drainage_time_scale),
           ", equilibrium_storage=", prettysummary(c.equilibrium_storage), ")")
