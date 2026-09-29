using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

"""
$(TYPEDEF)

The TEOS-10 freezing point of seawater: the Conservative Temperature at which seawater of
Absolute Salinity ``Sᴬ`` (g/kg) freezes at sea pressure ``p`` (dbar), from the polynomial fit
`gsw_CT_freezing_poly` of the Gibbs SeaWater toolbox (McDougall et al., 2014), which is
accurate to 6 × 10⁻⁴ K. The sea pressure at depth ``z ≤ 0`` is ``p = - ρ₀ g z``.

It is the default liquidus of [`IceShelfOceanInterface`](@ref) when the ocean buoyancy uses
`TEOS10EquationOfState` (see [`ice_shelf_liquidus`](@ref)). The liquidus is nonlinear in ``Sᴬ``, so the three-equation solve is
repeated `iterations` times, each with the liquidus linearized about the previous interface salinity.

$(TYPEDFIELDS)
"""
struct TEOS10Liquidus{FT}
    "reference density ``ρ₀`` converting depth to pressure (kg/m³)"
    reference_density :: FT
    "gravitational acceleration ``g`` (m/s²)"
    gravitational_acceleration :: FT
    "saturation fraction of dissolved air, between 0 (air-free) and 1 (air-saturated)"
    saturation_fraction :: FT
    "number of relinearizations of the liquidus in the three-equation solve"
    iterations :: Int
    "coefficients ``c₀, …, c₂₂`` of the freezing polynomial"
    polynomial_coefficients :: NTuple{23, FT}
    "coefficients ``(2.4, a, b, S_SO)`` of the dissolved-air correction"
    air_coefficients :: NTuple{4, FT}
end

const TEOS10_freezing_polynomial_coefficients =
    ( 0.017947064327968736, -6.076099099929818,    4.883198653547851,   -11.88081601230542,
     13.34658511480257,     -8.722761043208607,    2.082038908808201,    -7.389420998107497,
     -2.110913185058476,     0.2295491578006229,  -0.9891538123307282,   -0.08987150128406496,
      0.3831132432071728,    1.054318231187074,    1.065556599652796,    -0.7997496801694032,
      0.3850133554097069,   -2.078616693017569,    0.8756340772729538,   -2.079022768390933,
      1.596435439942262,     0.1338002171109174,   1.242891021876471)

const TEOS10_freezing_air_coefficients = (2.4, 0.014289763856964, 0.057000649899720, 35.16504)

"""
$(TYPEDSIGNATURES)

Construct a [`TEOS10Liquidus`](@ref).

```jldoctest
using NumericalEarth.IceShelves: TEOS10Liquidus, melting_temperature

liquidus = TEOS10Liquidus()
round(melting_temperature(liquidus, 34.5, -500), digits = 4)

# output

-2.2671
```
"""
function TEOS10Liquidus(FT::DataType = Oceananigans.defaults.FloatType;
                        reference_density = 1020,
                        gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                        saturation_fraction = 1,
                        iterations = 3)

    polynomial_coefficients = map(c -> convert(FT, c), TEOS10_freezing_polynomial_coefficients)
    air_coefficients = map(c -> convert(FT, c), TEOS10_freezing_air_coefficients)

    return TEOS10Liquidus(convert(FT, reference_density),
                          convert(FT, gravitational_acceleration),
                          convert(FT, saturation_fraction),
                          iterations,
                          polynomial_coefficients,
                          air_coefficients)
end

# Sea pressure in units of 10⁴ dbar
@inline reduced_pressure(liquidus::TEOS10Liquidus, z) =
    - liquidus.reference_density * liquidus.gravitational_acceleration * z / 100_000_000

@inline function melting_temperature(liquidus::TEOS10Liquidus, Sᴬ, z)
    c₀, c₁, c₂, c₃, c₄, c₅, c₆, c₇, c₈, c₉, c₁₀, c₁₁, c₁₂, c₁₃, c₁₄, c₁₅,
        c₁₆, c₁₇, c₁₈, c₁₉, c₂₀, c₂₁, c₂₂ = liquidus.polynomial_coefficients

    a₀, a, b, Sₛₒ = liquidus.air_coefficients

    Sᵣ = Sᴬ / 100
    x  = sqrt(Sᵣ)
    pᵣ = reduced_pressure(liquidus, z)

    Θᶠ = c₀ + Sᵣ * (c₁ + x * (c₂ + x * (c₃ + x * (c₄ + x * (c₅ + c₆ * x))))) +
         pᵣ * (c₇ + pᵣ * (c₈ + c₉ * pᵣ)) +
         Sᵣ * pᵣ * (c₁₀ + pᵣ * (c₁₂ + pᵣ * (c₁₅ + c₂₁ * Sᵣ)) + Sᵣ * (c₁₃ + c₁₇ * pᵣ + c₁₉ * Sᵣ) +
                    x * (c₁₁ + pᵣ * (c₁₄ + c₁₈ * pᵣ) + Sᵣ * (c₁₆ + c₂₀ * pᵣ + c₂₂ * Sᵣ)))

    air_correction = liquidus.saturation_fraction * (a₀ - a * Sᴬ) * (1 + b * (1 - Sᴬ / Sₛₒ)) / 1000

    return Θᶠ - air_correction
end

"""
$(TYPEDSIGNATURES)

Return ``∂Θᶠ/∂Sᴬ``, the derivative of the TEOS-10 freezing temperature with respect to
Absolute Salinity (K kg/g) at depth `z`.
"""
@inline function melting_temperature_salinity_derivative(liquidus::TEOS10Liquidus, Sᴬ, z)
    c₀, c₁, c₂, c₃, c₄, c₅, c₆, c₇, c₈, c₉, c₁₀, c₁₁, c₁₂, c₁₃, c₁₄, c₁₅,
        c₁₆, c₁₇, c₁₈, c₁₉, c₂₀, c₂₁, c₂₂ = liquidus.polynomial_coefficients

    a₀, a, b, Sₛₒ = liquidus.air_coefficients

    x  = sqrt(Sᴬ / 100)
    pᵣ = reduced_pressure(liquidus, z)

    ∂Θᶠ∂Sᵣ = c₁ + x * (3c₂/2 + x * (2c₃ + x * (5c₄/2 + x * (3c₅ + 7c₆/2 * x)))) +
             pᵣ * (c₁₀ + x * (3c₁₁/2 + x * (2c₁₃ + x * (5c₁₆/2 + x * (3c₁₉ + 7c₂₂/2 * x)))) +
             pᵣ * (c₁₂ + x * (3c₁₄/2 + x * (2c₁₇ + 5c₂₀/2 * x)) +
             pᵣ * (c₁₅ + x * (3c₁₈/2 + 2c₂₁ * x))))

    ∂air∂Sᴬ = liquidus.saturation_fraction * (- a * (1 + b * (1 - Sᴬ / Sₛₒ)) - b * (a₀ - a * Sᴬ) / Sₛₒ) / 1000

    return ∂Θᶠ∂Sᵣ / 100 - ∂air∂Sᴬ
end

# Tangent to the liquidus at (Sᴬ, z), so the linearization is exact at Sᴬ.
@inline function linearize(liquidus::TEOS10Liquidus, Sᴬ, z)
    Sᴬ⁺ = max(Sᴬ, zero(Sᴬ))
    m = - melting_temperature_salinity_derivative(liquidus, Sᴬ⁺, z)
    T₀ = melting_temperature(liquidus, Sᴬ⁺, z) + m * Sᴬ⁺
    return LinearLiquidus(T₀, m)
end

"""
$(TYPEDSIGNATURES)

Solve the three-equation balance at the ice base at depth `zᵈ`, returning the interface heat
flux, temperature and salinity. A linear liquidus is solved once at depth `zᵈ`; a
[`TEOS10Liquidus`](@ref) is linearized about the ocean salinity and then about each new
interface salinity, `liquidus.iterations` times.
"""
@inline ice_shelf_interface_heat_flux(flux_formulation, ocean_state, ice_state, liquidus, zᵈ, properties, ℰ, u★) =
    compute_interface_heat_flux(flux_formulation, ocean_state, ice_state, at_depth(liquidus, zᵈ), properties, ℰ, u★)

@inline function ice_shelf_interface_heat_flux(flux_formulation, ocean_state, ice_state, liquidus::TEOS10Liquidus,
                                               zᵈ, properties, ℰ, u★)
    FT = typeof(ocean_state.S)
    𝒬  = zero(FT)
    Tᵦ = zero(FT)
    Sᵦ = ocean_state.S

    for _ in 1:liquidus.iterations
        𝒬, Tᵦ, Sᵦ = compute_interface_heat_flux(flux_formulation, ocean_state, ice_state,
                                                linearize(liquidus, Sᵦ, zᵈ), properties, ℰ, u★)
    end

    return 𝒬, Tᵦ, Sᵦ
end

"""
$(TYPEDSIGNATURES)

Return the liquidus used by `interface` for the ocean `model`: `interface.liquidus`, or, when
that is `nothing`, a [`TEOS10Liquidus`](@ref) with the equation of state's reference density
and the buoyancy's gravitational acceleration if `model.buoyancy` uses `TEOS10EquationOfState`,
and the ISOMIP+ [`PressureDependentLiquidus`](@ref) otherwise.
"""
ice_shelf_liquidus(interface, model) = ice_shelf_liquidus(interface.liquidus, eltype(model.grid), model.buoyancy)

ice_shelf_liquidus(liquidus, FT, buoyancy) = liquidus
ice_shelf_liquidus(::Nothing, FT, buoyancy) = PressureDependentLiquidus(FT)
ice_shelf_liquidus(::Nothing, FT, buoyancy::BuoyancyForce) = ice_shelf_liquidus(nothing, FT, buoyancy.formulation)
ice_shelf_liquidus(::Nothing, FT, buoyancy::SeawaterBuoyancy) =
    equation_of_state_liquidus(FT, buoyancy.equation_of_state, buoyancy.gravitational_acceleration)

equation_of_state_liquidus(FT, equation_of_state, gravitational_acceleration) = PressureDependentLiquidus(FT)
equation_of_state_liquidus(FT, equation_of_state::TEOS10EquationOfState, gravitational_acceleration) =
    TEOS10Liquidus(FT; reference_density = equation_of_state.reference_density, gravitational_acceleration)

Base.summary(::TEOS10Liquidus{FT}) where FT = "TEOS10Liquidus{$FT}"

function Base.show(io::IO, liquidus::TEOS10Liquidus)
    print(io, summary(liquidus), '\n')
    print(io, "├── reference_density: ", liquidus.reference_density, " kg/m³", '\n')
    print(io, "├── gravitational_acceleration: ", liquidus.gravitational_acceleration, " m/s²", '\n')
    print(io, "├── saturation_fraction: ", liquidus.saturation_fraction, '\n')
    print(io, "└── iterations: ", liquidus.iterations)
end
