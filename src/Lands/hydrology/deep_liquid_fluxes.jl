#####
##### Deep liquid flux closures for `VariablySaturatedHydrology`.
#####
##### Each closure implements
#####
#####     deep_liquid_flux(closure, M, θˡ, 𝒮, Π, K, Πᵈ, time) -> Jˡᵇ  (kg m⁻² s⁻¹)
#####
##### with `Jˡᵇ` positive upward (capillary rise / groundwater return) and
##### negative downward (drainage). All scalars are per-cell; the kernel
##### passes them in.
#####

"""
    NoDeepLiquidFlux()

Zero deep liquid flux. The slab is closed at the bottom.
"""
struct NoDeepLiquidFlux end

@inline deep_liquid_flux(::NoDeepLiquidFlux, M, θˡ, 𝒮, Π, K, Πᵈ, time) = zero(M)

Base.summary(::NoDeepLiquidFlux) = "NoDeepLiquidFlux"

"""
    FreeDrainageFlux(liquid_density)

Free-drainage bottom boundary. `∂z Π = 0` so `∂z h = 1` and `Jˡ = -ρˡ K`.
`ρˡ` is the intrinsic liquid-water density (kg m⁻³).
"""
struct FreeDrainageFlux{FT}
    liquid_density :: FT
end

FreeDrainageFlux(FT::Type = Oceananigans.defaults.FloatType; liquid_density = 1000) =
    FreeDrainageFlux(convert(FT, liquid_density))

@inline deep_liquid_flux(c::FreeDrainageFlux, M, θˡ, 𝒮, Π, K, Πᵈ, time) =
    -convert(typeof(M), c.liquid_density) * K

Base.summary(c::FreeDrainageFlux) =
    string("FreeDrainageFlux(liquid_density=", prettysummary(c.liquid_density), ")")

"""
    DarcyDeepLiquidFlux(exchange_length, liquid_density)

Darcy exchange between the slab bottom and a deep reservoir held at pressure
head `Πᵈ`, set via `VariablySaturatedHydrology(deep_pressure_head = …)`. The
exchange length `ℓ_l` (m) separates the slab bottom from the deep reservoir;
the deep height is `z_D = z_b − ℓ_l`. With hydraulic heads `h_b = z_b + Π_b`
and `h_D = z_D + Πᵈ`,

```math
J^{lb} = \\rho^l K_b\\,\\frac{h_D - h_b}{\\ell_l}.
```
"""
struct DarcyDeepLiquidFlux{FT}
    exchange_length :: FT
    liquid_density  :: FT
end

DarcyDeepLiquidFlux(FT::Type = Oceananigans.defaults.FloatType;
                    exchange_length, liquid_density = 1000) =
    DarcyDeepLiquidFlux(convert(FT, exchange_length),
                        convert(FT, liquid_density))

@inline function deep_liquid_flux(c::DarcyDeepLiquidFlux, M, θˡ, 𝒮, Π, K, Πᵈ, time)
    FT = typeof(M)
    ℓ  = convert(FT, c.exchange_length)
    ρˡ = convert(FT, c.liquid_density)
    # h_D − h_b = (z_D + Πᵈ) − (z_b + Π) = -ℓ + Πᵈ − Π
    return ρˡ * K * (Πᵈ - Π - ℓ) / ℓ
end

Base.summary(c::DarcyDeepLiquidFlux) =
    string("DarcyDeepLiquidFlux(exchange_length=", prettysummary(c.exchange_length),
           ", liquid_density=", prettysummary(c.liquid_density), ")")

"""
    LinearReservoirDrainage(drainage_time_scale, equilibrium_storage)

Cheap linear-reservoir drainage:

```math
J^{lb} = -\\frac{\\max(M^{la} - M_{eq}, 0)}{\\tau_l}.
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

@inline function deep_liquid_flux(c::LinearReservoirDrainage, M, θˡ, 𝒮, Π, K, Πᵈ, time)
    FT = typeof(M)
    return -max(M - convert(FT, c.equilibrium_storage), 0) / convert(FT, c.drainage_time_scale)
end

Base.summary(c::LinearReservoirDrainage) =
    string("LinearReservoirDrainage(drainage_time_scale=", prettysummary(c.drainage_time_scale),
           ", equilibrium_storage=", prettysummary(c.equilibrium_storage), ")")
