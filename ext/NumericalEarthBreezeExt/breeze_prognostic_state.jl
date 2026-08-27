#####
##### Breeze prognostic maps: moist thermodynamic state → CompressibleDynamics prognostics
#####
#
# Pointwise `@inline` maps from a moist thermodynamic state (temperature `T`, vapor `qᵛ`, total
# liquid `qˡ` = cloud liquid + rain, total ice `qⁱ` = cloud ice + snow, pressure `p`) to the
# variables Breeze integrates. Written on bare scalars so they are GPU-safe and reusable both over
# whole `Field`s (`breeze_prognostic_state`) and per boundary-face node (the on-the-fly nesting
# boundary conditions / relaxation).
#
# `air_density` and `total_water_specific_humidity` are pure arithmetic, so `Field` arguments compose
# lazily via AbstractOperations. `liquid_ice_potential_temperature` builds a Breeze thermodynamic
# state, so it is applied through a `KernelFunctionOperation` instead.

using Oceananigans.Fields: Field, compute!, location
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Breeze: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant
using Breeze.Thermodynamics: MoistureMassFractions, LiquidIcePotentialTemperatureState, with_temperature

# Total moist-air density from Breeze's equation of state `ρ = p / (Rᵐ T)`, with the moist-air
# mixture gas constant `Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ` and dry-air mass fraction `qᵈ = 1 − qᵛ − qˡ − qⁱ` —
# exactly Breeze's `mixture_gas_constant`/`density`. `qˡ`/`qⁱ` are the TOTAL liquid/ice mass fractions
# (cloud + precipitation): every hydrometeor is mass that is NOT dry gas, so it loads the mixture. A
# vapor-only `qᵈ = 1 − qᵛ` (plain virtual temperature) over-counts the dry-air contribution and biases
# ρ — and hence ρᵈ, ρθ, ρu, ρv — low wherever condensate is present.
@inline air_density(T, qᵛ, qˡ, qⁱ, p, Rᵈ, Rᵛ) = p / (((1 - qᵛ - qˡ - qⁱ) * Rᵈ + qᵛ * Rᵛ) * T)

# `with_temperature` is the exact inverse of the `temperature` the child reconstructs with, so θˡⁱ is
# taken from it rather than restated here.
#
# `qᵛ` has no latent-heat term of its own, but it sets the answer twice. It fixes the mixture gas constant
# `Rᵐ = qᵈRᵈ + qᵛRᵛ` and the mixture heat capacity `cᵖᵐ = qᵈcᵖᵈ + qᵛcᵖᵛ + qˡcˡ + qⁱcⁱ` (both through
# `qᵈ = 1 − qᵛ − qˡ − qⁱ` as well), whose ratio is the Exner exponent in `Π = (p/pˢᵗ)^(Rᵐ/cᵖᵐ)`; and `cᵖᵐ`
# divides the latent term again in `θˡⁱ = (T − (ℒˡqˡ + ℒⁱqⁱ)/cᵖᵐ)/Π`. `pˢᵗ` is the caller's
# `standard_pressure`, never hardcoded.
@inline function liquid_ice_potential_temperature(T, qᵛ, qˡ, qⁱ, p, pˢᵗ, constants)
    FT = typeof(T)
    q  = MoistureMassFractions(convert(FT, qᵛ), convert(FT, qˡ), convert(FT, qⁱ))
    𝒰  = LiquidIcePotentialTemperatureState(zero(FT), q, convert(FT, pˢᵗ), convert(FT, p))
    return with_temperature(𝒰, T, constants).potential_temperature
end

# An absent species arrives as the scalar `0` from `full_snapshot(::Nothing, t)`, so accept a number
# alongside anything indexable. The arithmetic maps get this from broadcasting.
@inline value_at(q, i, j, k) = @inbounds q[i, j, k]
@inline value_at(q::Number, i, j, k) = q

# `KernelFunctionOperation` form of the same map.
@inline liquid_ice_potential_temperature(i, j, k, grid, T, qᵛ, qˡ, qⁱ, p, pˢᵗ, constants) =
    liquid_ice_potential_temperature(value_at(T, i, j, k), value_at(qᵛ, i, j, k), value_at(qˡ, i, j, k),
                                     value_at(qⁱ, i, j, k), value_at(p, i, j, k), pˢᵗ, constants)

@inline total_water_specific_humidity(qᵛ, qˡ, qⁱ) = qᵛ + qˡ + qⁱ

"""
$(TYPEDSIGNATURES)

Map a moist thermodynamic column state — temperature `T`, water-vapor specific
humidity `qᵛ`, total liquid `qˡ` (cloud liquid + rain), total ice `qⁱ` (cloud
ice + snow), and pressure `p` (all `Field`s on a common grid) — to the
prognostic fields integrated by Breeze's `CompressibleDynamics`:

  - density,                          `ρ   = p / (Rᵐ T)`,   with `Rᵐ = (1 − qᵗ) Rᵈ + qᵛ Rᵛ`
  - liquid-ice potential temperature, `θˡⁱ = (T − (Lᵥ qˡ + Lₛ qⁱ) / cₚᵐ) / Π`,
    with the Exner function `Π = (p/pˢᵗ)^(Rᵐ/cₚᵐ)` — Breeze's `with_temperature`,
    the exact inverse of the `temperature` the child reads it back with
  - total-water specific humidity,    `qᵗ  = qᵛ + qˡ + qⁱ`

`qˡ`/`qⁱ` load the density through Breeze's mixture gas constant and set `qᵗ`
(hence the dry density `ρᵈ = ρ(1 − qᵗ)`), so they must be the *total* liquid/ice
mass fractions — every hydrometeor the parent carries, not just cloud condensate.

`constants` (`Breeze.ThermodynamicConstants`) supplies the gas constants, dry-air
heat capacity, and latent heats; `pˢᵗ` is the potential-temperature reference
pressure (the child dynamics' `standard_pressure`). Returns a `NamedTuple`
`(; ρ, θˡⁱ, qᵗ)` of computed `Field`s. Both the ERA5 parent `FieldTimeSeries`
population and the child initial condition in `examples/era5_breeze.jl` route
through this so the conversion lives in one place.
"""
function NumericalEarth.Atmospheres.breeze_prognostic_state(constants::ThermodynamicConstants, pˢᵗ,
                                                            T, qᵛ, qˡ, qⁱ, p)
    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)

    ρ   = Field(air_density(T, qᵛ, qˡ, qⁱ, p, Rᵈ, Rᵛ))
    qᵗ  = Field(total_water_specific_humidity(qᵛ, qˡ, qⁱ))

    LX, LY, LZ = location(T)
    θˡⁱ = Field(KernelFunctionOperation{LX, LY, LZ}(liquid_ice_potential_temperature, T.grid,
                                                    T, qᵛ, qˡ, qⁱ, p, pˢᵗ, constants))

    compute!(ρ)
    compute!(θˡⁱ)
    compute!(qᵗ)

    return (; ρ, θˡⁱ, qᵗ)
end
