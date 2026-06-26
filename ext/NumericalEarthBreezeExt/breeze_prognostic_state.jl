#####
##### breeze_prognostic_state: moist thermodynamic state → Breeze prognostics
#####

using Oceananigans.Fields: Field, compute!
using Breeze: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant

"""
$(TYPEDSIGNATURES)

Map a moist thermodynamic column state — temperature `T`, water-vapor specific
humidity `qᵛ`, cloud-liquid `qᶜ`, cloud-ice `qⁱ`, and pressure `p` (all `Field`s
on a common grid) — to the prognostic fields integrated by Breeze's
`CompressibleDynamics`:

  - density,                          `ρ   = p / (Rᵈ Tᵛ)`,  with `Tᵛ = T (1 + (Rᵛ/Rᵈ − 1) qᵛ)`
  - liquid-ice potential temperature, `θˡⁱ = θ (1 − (Lᵥ qᶜ + Lₛ qⁱ) / (cₚᵈ T))`,
    with `θ = T (pˢᵗ/p)^(Rᵈ/cₚᵈ)` and `pˢᵗ = 10⁵ Pa`
  - total-water specific humidity,    `qᵗ  = qᵛ + qᶜ + qⁱ`

`constants` (`Breeze.ThermodynamicConstants`) supplies the gas constants, dry-air
heat capacity, and latent heats. Returns a `NamedTuple` `(; ρ, θˡⁱ, qᵗ)` of
computed `Field`s. Both the ERA5 parent `FieldTimeSeries` population and the
child initial condition in `examples/era5_breeze.jl` route through this so the
conversion lives in one place.
"""
function NumericalEarth.Atmospheres.breeze_prognostic_state(constants::ThermodynamicConstants,
                                                            T, qᵛ, qᶜ, qⁱ, p)
    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cₚᵈ = constants.dry_air.heat_capacity
    Lᵥ  = constants.liquid.reference_latent_heat
    Lₛ  = constants.ice.reference_latent_heat

    κ    = Rᵈ / cₚᵈ
    εfac = Rᵛ / Rᵈ - 1
    pˢᵗ  = 1e5  # Pa

    Tᵛ  = T * (1 + εfac * qᵛ)                 # virtual temperature (vapor only)
    θ   = T * (pˢᵗ / p)^κ                     # potential temperature

    ρ   = Field(p / (Rᵈ * Tᵛ))                                   # moist ideal gas law
    θˡⁱ = Field(θ * (1 - (Lᵥ * qᶜ + Lₛ * qⁱ) / (cₚᵈ * T)))       # Breeze diagnostic form
    qᵗ  = Field(qᵛ + qᶜ + qⁱ)

    compute!(ρ)
    compute!(θˡⁱ)
    compute!(qᵗ)

    return (; ρ, θˡⁱ, qᵗ)
end
