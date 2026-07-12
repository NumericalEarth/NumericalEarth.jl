#####
##### Breeze prognostic maps: moist thermodynamic state → CompressibleDynamics prognostics
#####
#
# Pointwise `@inline` maps from a moist thermodynamic state (temperature `T`, vapor `qᵛ`, total
# liquid `qˡ` = cloud liquid + rain, total ice `qⁱ` = cloud ice + snow, pressure `p`) to the
# variables Breeze integrates. Written on bare scalars + gas-constant/heat-capacity/latent-heat
# numbers so they are GPU-safe and reusable both broadcast over `Field`s (`breeze_prognostic_state`)
# and per boundary-face node (the on-the-fly nesting boundary conditions / relaxation). `Field`-valued
# arguments compose lazily via AbstractOperations, so the same definitions materialize fields too.

using Oceananigans.Fields: Field, compute!
using Breeze: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant

# Total moist-air density from Breeze's equation of state `ρ = p / (Rᵐ T)`, with the moist-air
# mixture gas constant `Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ` and dry-air mass fraction `qᵈ = 1 − qᵛ − qˡ − qⁱ` —
# exactly Breeze's `mixture_gas_constant`/`density`. `qˡ`/`qⁱ` are the TOTAL liquid/ice mass fractions
# (cloud + precipitation): every hydrometeor is mass that is NOT dry gas, so it loads the mixture. A
# vapor-only `qᵈ = 1 − qᵛ` (plain virtual temperature) over-counts the dry-air contribution and biases
# ρ — and hence ρᵈ, ρθ, ρu, ρv — low wherever condensate is present.
@inline air_density(T, qᵛ, qˡ, qⁱ, p, Rᵈ, Rᵛ) = p / (((1 - qᵛ - qˡ - qⁱ) * Rᵈ + qᵛ * Rᵛ) * T)

# `pˢᵗ` is the potential-temperature reference pressure, supplied by the caller from the child
# dynamics' `standard_pressure` (Breeze's `ReferenceState` value) — never hardcoded.
@inline potential_temperature(T, p, pˢᵗ, Rᵈ, cᵖᵈ) = T * (pˢᵗ / p)^(Rᵈ / cᵖᵈ)

@inline liquid_ice_potential_temperature(T, qˡ, qⁱ, p, pˢᵗ, Rᵈ, cᵖᵈ, ℒˡ, ℒⁱ) =
    potential_temperature(T, p, pˢᵗ, Rᵈ, cᵖᵈ) * (1 - (ℒˡ * qˡ + ℒⁱ * qⁱ) / (cᵖᵈ * T))

@inline total_water_specific_humidity(qᵛ, qˡ, qⁱ) = qᵛ + qˡ + qⁱ

"""
$(TYPEDSIGNATURES)

Map a moist thermodynamic column state — temperature `T`, water-vapor specific
humidity `qᵛ`, total liquid `qˡ` (cloud liquid + rain), total ice `qⁱ` (cloud
ice + snow), and pressure `p` (all `Field`s on a common grid) — to the
prognostic fields integrated by Breeze's `CompressibleDynamics`:

  - density,                          `ρ   = p / (Rᵐ T)`,   with `Rᵐ = (1 − qᵗ) Rᵈ + qᵛ Rᵛ`
  - liquid-ice potential temperature, `θˡⁱ = θ (1 − (Lᵥ qˡ + Lₛ qⁱ) / (cₚᵈ T))`,
    with `θ = T (pˢᵗ/p)^(Rᵈ/cₚᵈ)`
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
    cᵖᵈ = constants.dry_air.heat_capacity
    ℒˡ  = constants.liquid.reference_latent_heat
    ℒⁱ  = constants.ice.reference_latent_heat

    ρ   = Field(air_density(T, qᵛ, qˡ, qⁱ, p, Rᵈ, Rᵛ))
    θˡⁱ = Field(liquid_ice_potential_temperature(T, qˡ, qⁱ, p, pˢᵗ, Rᵈ, cᵖᵈ, ℒˡ, ℒⁱ))
    qᵗ  = Field(total_water_specific_humidity(qᵛ, qˡ, qⁱ))

    compute!(ρ)
    compute!(θˡⁱ)
    compute!(qᵗ)

    return (; ρ, θˡⁱ, qᵗ)
end
