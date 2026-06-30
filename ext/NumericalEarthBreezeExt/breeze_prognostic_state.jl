#####
##### Breeze prognostic maps: moist thermodynamic state → CompressibleDynamics prognostics
#####
#
# Pointwise `@inline` maps from a moist thermodynamic state (temperature `T`, vapor `qᵛ`, cloud
# liquid `qᶜˡ`, cloud ice `qᶜⁱ`, pressure `p`) to the variables Breeze integrates. Written on bare
# scalars + gas-constant/heat-capacity/latent-heat numbers so they are GPU-safe and reusable both
# broadcast over `Field`s (`breeze_prognostic_state`) and per boundary-face node (the on-the-fly
# nesting boundary conditions / relaxation). `Field`-valued arguments compose lazily via
# AbstractOperations, so the same definitions materialize fields too.

using Oceananigans.Fields: Field, compute!
using Breeze: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant

@inline virtual_temperature(T, qᵛ, Rᵈ, Rᵛ) = T * (1 + (Rᵛ / Rᵈ - 1) * qᵛ)

@inline air_density(T, qᵛ, p, Rᵈ, Rᵛ) = p / (Rᵈ * virtual_temperature(T, qᵛ, Rᵈ, Rᵛ))

# `pˢᵗ` is the potential-temperature reference pressure, supplied by the caller from the child
# dynamics' `standard_pressure` (Breeze's `ReferenceState` value) — never hardcoded.
@inline potential_temperature(T, p, pˢᵗ, Rᵈ, cᵖᵈ) = T * (pˢᵗ / p)^(Rᵈ / cᵖᵈ)

@inline liquid_ice_potential_temperature(T, qᶜˡ, qᶜⁱ, p, pˢᵗ, Rᵈ, cᵖᵈ, ℒˡ, ℒⁱ) =
    potential_temperature(T, p, pˢᵗ, Rᵈ, cᵖᵈ) * (1 - (ℒˡ * qᶜˡ + ℒⁱ * qᶜⁱ) / (cᵖᵈ * T))

@inline total_water_specific_humidity(qᵛ, qᶜˡ, qᶜⁱ) = qᵛ + qᶜˡ + qᶜⁱ

"""
$(TYPEDSIGNATURES)

Map a moist thermodynamic column state — temperature `T`, water-vapor specific
humidity `qᵛ`, cloud-liquid `qᶜ`, cloud-ice `qⁱ`, and pressure `p` (all `Field`s
on a common grid) — to the prognostic fields integrated by Breeze's
`CompressibleDynamics`:

  - density,                          `ρ   = p / (Rᵈ Tᵛ)`,  with `Tᵛ = T (1 + (Rᵛ/Rᵈ − 1) qᵛ)`
  - liquid-ice potential temperature, `θˡⁱ = θ (1 − (Lᵥ qᶜ + Lₛ qⁱ) / (cₚᵈ T))`,
    with `θ = T (pˢᵗ/p)^(Rᵈ/cₚᵈ)`
  - total-water specific humidity,    `qᵗ  = qᵛ + qᶜ + qⁱ`

`constants` (`Breeze.ThermodynamicConstants`) supplies the gas constants, dry-air
heat capacity, and latent heats; `pˢᵗ` is the potential-temperature reference
pressure (the child dynamics' `standard_pressure`). Returns a `NamedTuple`
`(; ρ, θˡⁱ, qᵗ)` of computed `Field`s. Both the ERA5 parent `FieldTimeSeries`
population and the child initial condition in `examples/era5_breeze.jl` route
through this so the conversion lives in one place.
"""
function NumericalEarth.Atmospheres.breeze_prognostic_state(constants::ThermodynamicConstants, pˢᵗ,
                                                            T, qᵛ, qᶜˡ, qᶜⁱ, p)
    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    ℒˡ  = constants.liquid.reference_latent_heat
    ℒⁱ  = constants.ice.reference_latent_heat

    ρ   = Field(air_density(T, qᵛ, p, Rᵈ, Rᵛ))
    θˡⁱ = Field(liquid_ice_potential_temperature(T, qᶜˡ, qᶜⁱ, p, pˢᵗ, Rᵈ, cᵖᵈ, ℒˡ, ℒⁱ))
    qᵗ  = Field(total_water_specific_humidity(qᵛ, qᶜˡ, qᶜⁱ))

    compute!(ρ)
    compute!(θˡⁱ)
    compute!(qᵗ)

    return (; ρ, θˡⁱ, qᵗ)
end
