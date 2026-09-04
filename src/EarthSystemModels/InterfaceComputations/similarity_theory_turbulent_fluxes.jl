using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Utils: prettysummary
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

#####
##### Bulk turbulent fluxes based on similarity theory
#####

struct SimilarityTheoryFluxes{FT, UF, R, D, B, S, SV}
    von_karman_constant :: FT        # parameter
    turbulent_prandtl_number :: FT   # parameter
    subgrid_velocities :: SV         # empirical velocity enhancements of the bulk wind
    stability_functions :: UF        # functions for turbulent fluxes
    roughness_lengths :: R           # parameterization for turbulent fluxes
    zero_plane_displacement :: D     # displacement of the similarity profile
    similarity_form :: B             # similarity profile relating atmosphere to interface state
    solver_stop_criteria :: S        # stop criteria for compute_interface_state
end

Adapt.@adapt_structure SimilarityTheoryFluxes

#####
##### Subgrid velocity corrections: empirical enhancements of the bulk velocity
##### representing motions unresolved by the atmosphere state, added in quadrature
##### to the resolved velocity difference.
#####

"""
    ConvectiveGustiness{FT}(; gustiness_parameter = 1.2, minimum_gustiness = 0.01)

Beljaars (1995)-type convective gustiness: in unstable conditions ``(Jᵇ > 0)``
the bulk velocity is enhanced by ``Uᴳ = β (Jᵇ h_{bℓ})^{1/3}``, where ``Jᵇ = -u★b★``
is the surface buoyancy flux, ``h_{bℓ}`` the boundary layer height, and
``β`` the `gustiness_parameter`. In stable conditions the enhancement falls back
to the `minimum_gustiness` floor [m/s].
"""
@kwdef struct ConvectiveGustiness{FT}
    gustiness_parameter :: FT = 1.2   # β
    minimum_gustiness   :: FT = 0.01  # velocity floor [m/s]
end

Base.summary(::ConvectiveGustiness{FT}) where FT = "ConvectiveGustiness{$FT}"
Base.show(io::IO, g::ConvectiveGustiness) = print(io, summary(g))

"""
    SubgridVelocityCorrection(FT = Float64;
                              convective = ConvectiveGustiness{FT}(),
                              mesoscale = nothing)

Composition of the empirical subgrid velocity enhancements applied to the bulk
velocity, ``U² = Δu² + Δv² + Uᶜ² + Uᵐ²``:

- `convective`: a state-dependent convective gustiness, by default [`ConvectiveGustiness`](@ref).
- `mesoscale`: a static velocity scale [m/s] representing mesoscale variability
  unresolved on coarse grids, e.g. [`mahrt_sun_subgrid_velocity`](@ref)`(Δx)`.
  Default `nothing` (no contribution).

Either slot may be `nothing`, a `Number` [m/s], or a formulation implementing
`vsgs²(correction, u★, b★, h_bℓ)`.
"""
struct SubgridVelocityCorrection{C, M}
    convective :: C
    mesoscale :: M
end

function SubgridVelocityCorrection(FT::DataType = Oceananigans.defaults.FloatType;
                                   convective = ConvectiveGustiness{FT}(),
                                   mesoscale = nothing)
    convective isa Number && (convective = convert(FT, convective))
    mesoscale  isa Number && (mesoscale  = convert(FT, mesoscale))
    return SubgridVelocityCorrection(convective, mesoscale)
end

Base.summary(sv::SubgridVelocityCorrection) =
    string("SubgridVelocityCorrection(convective=", prettysummary(sv.convective),
           ", mesoscale=", prettysummary(sv.mesoscale), ")")

Base.show(io::IO, sv::SubgridVelocityCorrection) = print(io, summary(sv))

@inline vsgs²(::Nothing, u★, b★, h_bℓ) = 0
@inline vsgs²(v::Number, u★, b★, h_bℓ) = v^2

@inline function vsgs²(g::ConvectiveGustiness, u★, b★, h_bℓ)
    Jᵇ = - u★ * b★
    Uᴳ = max(g.minimum_gustiness, g.gustiness_parameter * cbrt(max(zero(Jᵇ), Jᵇ) * h_bℓ))
    return Uᴳ^2
end

@inline vsgs²(sv::SubgridVelocityCorrection, u★, b★, h_bℓ) =
    vsgs²(sv.convective, u★, b★, h_bℓ) + vsgs²(sv.mesoscale, u★, b★, h_bℓ)

"""
    mahrt_sun_subgrid_velocity(Δx; threshold = 5e3)

Return the mesoscale subgrid velocity [m/s] for grid spacing `Δx` [m] following
[Mahrt and Sun (1995)](https://doi.org/10.1175/1520-0493(1995)123<3032:TSVSIT>2.0.CO;2),
as implemented in the revised MM5 surface layer scheme of [Jiménez et al. (2012)](https://doi.org/10.1175/MWR-D-11-00056.1):

```math
V_{sg} = 0.32 \\, [\\max(Δx / 5000 - 1, 0)]^{0.33}
```

The enhancement is zero for `Δx ≤ threshold` (default 5 km). Pass the result as
the `mesoscale` slot of [`SubgridVelocityCorrection`](@ref).
"""
function mahrt_sun_subgrid_velocity(Δx; threshold = 5e3)
    δ = max(Δx / threshold - 1, 0)
    return 0.32 * δ^0.33
end


Base.summary(::SimilarityTheoryFluxes{FT}) where FT = "SimilarityTheoryFluxes{$FT}"

function Base.show(io::IO, fluxes::SimilarityTheoryFluxes)
    print(io, summary(fluxes), '\n',
          "├── von_karman_constant: ",        prettysummary(fluxes.von_karman_constant), '\n',
          "├── turbulent_prandtl_number: ",   prettysummary(fluxes.turbulent_prandtl_number), '\n',
          "├── subgrid_velocities: ",         summary(fluxes.subgrid_velocities), '\n',
          "├── stability_functions: ",        summary(fluxes.stability_functions), '\n',
          "├── roughness_lengths: ",          summary(fluxes.roughness_lengths), '\n',
          "├── zero_plane_displacement: ",    prettysummary(fluxes.zero_plane_displacement), '\n',
          "├── similarity_form: ",            summary(fluxes.similarity_form), '\n',
          "└── solver_stop_criteria: ",       summary(fluxes.solver_stop_criteria))
end

"""
    SimilarityTheoryFluxes(FT::DataType = Float64;
                           gravitational_acceleration = 9.81,
                           von_karman_constant = 0.4,
                           turbulent_prandtl_number = 1,
                           subgrid_velocities = ConvectiveGustiness{FT}(),
                           stability_functions = default_stability_functions(FT),
                           roughness_lengths = default_roughness_lengths(FT),
                           zero_plane_displacement = 0,
                           similarity_form = LogarithmicSimilarityProfile(),
                           solver_stop_criteria = nothing,
                           solver_tolerance = 1e-8,
                           solver_maxiter = 100)

`SimilarityTheoryFluxes` contains parameters and settings to calculate
air-interface turbulent fluxes using Monin--Obukhov similarity theory.

Keyword Arguments
==================

- `von_karman_constant`: The von Karman constant. Default: 0.4.
- `turbulent_prandtl_number`: The turbulent Prandtl number. Default: 1.
- `subgrid_velocities`: Empirical subgrid velocity enhancement of the bulk velocity: `nothing`,
                        a `Number` [m/s], a formulation implementing `vsgs²`, or a
                        [`SubgridVelocityCorrection`](@ref) composing several. Default:
                        [`ConvectiveGustiness`](@ref)`{FT}()`.
- `stability_functions`: The stability functions. Default: `default_stability_functions(FT)` that follow the
                         formulation of [edson2013exchange](@citet).
- `roughness_lengths`: The roughness lengths used to calculate the characteristic scales for momentum, temperature and
                       water vapor. Each may be a formulation, a `Number`, or — at the
                       atmosphere--land interface only — a `Field{Center, Center, Nothing}`
                       of per-cell values.
                       Default: `default_roughness_lengths(FT)`, formulation taken from [edson2013exchange](@citet).
- `zero_plane_displacement`: The zero-plane displacement `d` [m] of surfaces with tall roughness
                             elements (buildings, plant canopy): the similarity profiles are evaluated
                             at the height `Δh - d` above the interface. A `Number`, or — at the
                             atmosphere--land interface only — a `Field{Center, Center, Nothing}`
                             of per-cell displacements. Default: 0 (undisplaced).
- `similarity_form`: The type of similarity profile used to relate the atmospheric state to the
                             interface fluxes / characteristic scales.
- `solver_tolerance`: The tolerance for convergence. Default: 1e-8.
- `solver_maxiter`: The maximum number of iterations. Default: 100.
"""
function SimilarityTheoryFluxes(FT::DataType = Oceananigans.defaults.FloatType;
                                von_karman_constant = 0.4,
                                turbulent_prandtl_number = 1,
                                subgrid_velocities = ConvectiveGustiness{FT}(),
                                stability_functions = atmosphere_ocean_stability_functions(FT),
                                momentum_roughness_length = MomentumRoughnessLength(FT),
                                temperature_roughness_length = ScalarRoughnessLength(FT),
                                water_vapor_roughness_length = ScalarRoughnessLength(FT),
                                zero_plane_displacement = 0,
                                similarity_form = LogarithmicSimilarityProfile(),
                                solver_stop_criteria = nothing,
                                solver_tolerance = 1e-8,
                                solver_maxiter = 100)

    roughness_lengths = SimilarityScales(convert_if_number(FT, momentum_roughness_length),
                                         convert_if_number(FT, temperature_roughness_length),
                                         convert_if_number(FT, water_vapor_roughness_length))

    zero_plane_displacement = convert_if_number(FT, zero_plane_displacement)

    if isnothing(solver_stop_criteria)
        solver_tolerance = convert(FT, solver_tolerance)
        solver_stop_criteria = ConvergenceStopCriteria(solver_tolerance, solver_maxiter)
    end

    if isnothing(stability_functions)
        returns_zero = Returns(zero(FT))
        stability_functions = SimilarityScales(returns_zero, returns_zero, returns_zero)
    end

    return SimilarityTheoryFluxes(convert(FT, von_karman_constant),
                                  convert(FT, turbulent_prandtl_number),
                                  subgrid_velocities,
                                  stability_functions,
                                  roughness_lengths,
                                  zero_plane_displacement,
                                  similarity_form,
                                  solver_stop_criteria)
end

#####
##### Similarity profile types
#####

"""
    LogarithmicSimilarityProfile()

Represent the classic Monin--Obukhov similarity profile, which finds that

```math
ϕ(z) = Π(z) ϕ_★ / ϰ
```

where ``ϰ`` is the Von Karman constant, ``ϕ_★`` is the characteristic scale for ``ϕ``,
and ``Π`` is the "similarity profile",

```math
Π(h) = \\log(h / ℓ) - ψ(h / L) + ψ(ℓ / L)
```

which is a logarithmic profile adjusted by the stability function ``ψ`` and dependent on
the Monin--Obukhov length ``L`` and the roughness length ``ℓ``.
"""
struct LogarithmicSimilarityProfile end
struct COARELogarithmicSimilarityProfile end

@inline function similarity_profile(::LogarithmicSimilarityProfile, stability_function, h, ℓ, L)
    ζ = h / L
    ψh = stability_profile(stability_function, ζ)
    ψℓ = inner_stability_profile(stability_function, ℓ / L)
    return log(h / ℓ) - ψh + ψℓ
end

@inline function similarity_profile(::COARELogarithmicSimilarityProfile, stability_function, h, ℓ, L)
    ζ = h / L
    ψh = stability_profile(stability_function, ζ)
    return log(h / ℓ) - ψh
end

# Localize the flux closure to cell (i, j) before the index-free MOST iteration:
# `Field`-valued roughness lengths and displacement collapse to the cell's values,
# `Number`s and formulations pass through.
@inline local_flux_formulation(flux_formulation, i, j) = flux_formulation

@inline function local_flux_formulation(fluxes::SimilarityTheoryFluxes, i, j)
    ℓ = fluxes.roughness_lengths
    roughness_lengths = SimilarityScales(state2dindex(ℓ.momentum, i, j),
                                         state2dindex(ℓ.temperature, i, j),
                                         state2dindex(ℓ.water_vapor, i, j))

    return SimilarityTheoryFluxes(fluxes.von_karman_constant,
                                  fluxes.turbulent_prandtl_number,
                                  fluxes.subgrid_velocities,
                                  fluxes.stability_functions,
                                  roughness_lengths,
                                  state2dindex(fluxes.zero_plane_displacement, i, j),
                                  fluxes.similarity_form,
                                  fluxes.solver_stop_criteria)
end

# A zero-plane displacement at or above the surface layer height leaves no room
# for the similarity profiles.
validate_zero_plane_displacement(flux_formulation, zᵃᵗ) = nothing

function validate_zero_plane_displacement(fluxes::SimilarityTheoryFluxes, zᵃᵗ)
    Δhᵈ = minimum(zᵃᵗ - fluxes.zero_plane_displacement)
    Δhᵈ > 0 || throw(ArgumentError("zero_plane_displacement must be below the surface layer height, found a displaced profile height of $Δhᵈ m"))
    return nothing
end

#####
##### Layout of `Field`-valued roughness lengths and displacement
#####

function validate_interface_field(f::AbstractField, name, grid)
    location(f) === (Center, Center, Nothing) &&
        architecture(f) === architecture(grid) && f.grid == grid ||
        throw(ArgumentError("$name must be a Field{Center, Center, Nothing} on the interface grid, got $(summary(f))"))

    return nothing
end

validate_interface_field(f, name, grid) = nothing

"""
$(TYPEDSIGNATURES)

Check that any `Field`-valued roughness length or zero-plane displacement of
`flux_formulation` is laid out so the flux kernel can read it per cell on `grid`,
and throw an `ArgumentError` naming the offending field otherwise.
"""
function validate_flux_formulation(fluxes::SimilarityTheoryFluxes, grid)
    ℓ = fluxes.roughness_lengths

    validate_interface_field(ℓ.momentum,    "momentum_roughness_length",    grid)
    validate_interface_field(ℓ.temperature, "temperature_roughness_length", grid)
    validate_interface_field(ℓ.water_vapor, "water_vapor_roughness_length", grid)
    validate_interface_field(fluxes.zero_plane_displacement, "zero_plane_displacement", grid)

    return nothing
end

validate_flux_formulation(flux_formulation, grid) = nothing

function iterate_interface_fluxes(flux_formulation::SimilarityTheoryFluxes,
                                  Tₛ, qₛ, Δθ, Δq, Δh,
                                  approximate_interface_state,
                                  atmosphere_state,
                                  interface_properties,
                                  atmosphere_properties)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    g  = atmosphere_properties.gravitational_acceleration
    pᵃᵗ = atmosphere_state.p

    # "initial" scales because we will recompute them
    u★ = approximate_interface_state.fluxes.u★
    θ★ = approximate_interface_state.fluxes.θ★
    q★ = approximate_interface_state.fluxes.q★

    # Stability functions for momentum, heat, and vapor
    ψu = flux_formulation.stability_functions.momentum
    ψθ = flux_formulation.stability_functions.temperature
    ψq = flux_formulation.stability_functions.water_vapor

    ℓu = flux_formulation.roughness_lengths.momentum
    ℓθ = flux_formulation.roughness_lengths.temperature
    ℓq = flux_formulation.roughness_lengths.water_vapor

    # Compute Monin--Obukhov length scale depending on a `buoyancy flux`
    b★ = buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)

    # Squared subgrid velocity enhancements: convective gustiness and, on coarse
    # grids, an optional mesoscale contribution (see `SubgridVelocityCorrection`).
    h_bℓ = atmosphere_state.h_bℓ
    Uˢᵍ² = vsgs²(flux_formulation.subgrid_velocities, u★, b★, h_bℓ)

    # Velocity difference accounting for subgrid velocity enhancements
    Δu, Δv = velocity_difference(interface_properties.velocity_formulation,
                                 atmosphere_state,
                                 approximate_interface_state)

    U = sqrt(Δu^2 + Δv^2 + Uˢᵍ²)

    # Compute roughness length scales (pass surface temperature for viscosity calculation)
    ℓu₀ = roughness_length(ℓu, u★, U, ℂᵃᵗ, Tₛ)
    ℓq₀ = roughness_length(ℓq, ℓu₀, u★, U, ℂᵃᵗ, Tₛ)
    ℓθ₀ = roughness_length(ℓθ, ℓu₀, u★, U, ℂᵃᵗ, Tₛ)

    # Tall roughness elements displace the similarity profiles upward by `d`.
    d = flux_formulation.zero_plane_displacement
    Δhᵈ = Δh - d

    # Transfer coefficients at height `h`
    ϰ = flux_formulation.von_karman_constant
    L★ = ifelse(b★ == 0, Inf, u★^2 / (ϰ * b★))
    form = flux_formulation.similarity_form

    χu = ϰ / similarity_profile(form, ψu, Δhᵈ, ℓu₀, L★)
    χθ = ϰ / similarity_profile(form, ψθ, Δhᵈ, ℓθ₀, L★)
    χq = ϰ / similarity_profile(form, ψq, Δhᵈ, ℓq₀, L★)

    # Recompute
    u★ = χu * U
    θ★ = χθ * Δθ
    q★ = χq * Δq

    return u★, θ★, q★, χθ, χq
end

"""
    buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)

Return the characteristic buoyancy scale `b★` associated with
the characteristic temperature `θ★`, specific humidity scale `q★`,
surface temperature `Tₛ`, surface specific humidity `qₛ`,
atmosphere thermodynamic parameters `ℂᵃᵗ`, and gravitational acceleration `g`.

The buoyancy scale is defined in terms of the interface buoyancy flux,

```math
u★ b★ ≡ w'b',
```

where `u_★` is the friction velocity.
Using the definition of buoyancy for clear air without condensation, we find that

```math
b★ = (g / 𝒯ₛ) [θ★ (1 + δ qₛ) + δ 𝒯ₛ q★] ,
```
where ``𝒯ₛ`` is the virtual temperature at the surface, and ``δ = Rᵛ / Rᵈ - 1``,
where ``Rᵛ`` is the molar mass of water vapor and ``Rᵈ`` is the molar mass of dry air.

Note that the Monin--Obukhov characteristic length scale is defined
in terms of ``b★`` and additionally the Von Karman constant ``ϰ``,

```math
L★ = u★² / ϰ b★ .
```
"""
@inline function buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)
    𝒯ₛ = AtmosphericThermodynamics.virtual_temperature(ℂᵃᵗ, Tₛ, qₛ)
    ε  = AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)
    δ  = ε - 1 # typically equal to 0.608

    b★ = g / 𝒯ₛ * (θ★ * (1 + δ * qₛ) + δ * 𝒯ₛ * q★)

    return b★
end


#####
##### Struct that represents a 3-tuple of momentum, heat, and water vapor
#####

struct SimilarityScales{U, T, Q}
    momentum :: U
    temperature :: T
    water_vapor :: Q
end

Adapt.@adapt_structure SimilarityScales

Base.summary(ss::SimilarityScales) =
    string("SimilarityScales(momentum=", prettysummary(ss.momentum),
           ", temperature=", prettysummary(ss.temperature),
           ", water_vapor=", prettysummary(ss.water_vapor), ")")

Base.show(io::IO, ss::SimilarityScales) = print(io, summary(ss))

@inline stability_profile(ψ, ζ) = ψ(ζ)

# Convenience
abstract type AbstractStabilityFunction end
@inline (ψ::AbstractStabilityFunction)(ζ) = stability_profile(ψ, ζ)

"""
    EdsonMomentumStabilityFunction{FT}

A struct representing the momentum stability function detailed by [edson2013exchange](@citet).
The formulation hinges on the definition of three different functions:
one for stable atmospheric conditions ``(ζ > 0)``, named ``ψₛ`` and two for unstable conditions,
named ``ψᵤ₁`` and ``ψᵤ₂``.
These stability functions are obtained by regression to experimental data.

The stability parameter for stable atmospheric conditions is defined as
```math
\\begin{align*}
dζ &= \\min(ζ_{\\max}, A⁺ ζ) \\\\
ψ⁺ &= - B⁺ ζ⁺ - C⁺ (ζ⁺ - D⁺) \\exp(- dζ) - C⁺ D⁺
\\end{align*}
```

While the stability parameter for unstable atmospheric conditions is calculated
as a function of the two individual stability functions as follows

```math
\\begin{align*}
f⁻₁ &= (1 - A⁻ζ)^{1/4} \\\\
ψ⁻₁ &= (B⁻ / 2) \\log[(1 + f⁻₁ + f⁻₁² + f⁻₁³) / B⁻] - √B⁻ \\mathrm{atan}(f⁻₁) - C⁻ \\\\
\\\\
f⁻₂ &= ∛(1 - D⁻ζ) \\\\
ψ⁻₂ &= (E⁻ / 2) \\log[(1 + f⁻₂ + f⁻₂²) / E⁻]- √E⁻ \\mathrm{atan}[(1 + 2f⁻₂) / √E⁻] + F⁻ \\\\
\\\\
f   &= ζ² / (1 + ζ²) \\\\
ψ⁻  &= (1 - f) ψ⁻₁ + f ψ⁻₂
\\end{align*}
```

The superscripts ``+`` and ``-`` indicate if the parameter applies to the
stability function for _stable_ or _unstable_ atmospheric conditions, respectively.
"""
@kwdef struct EdsonMomentumStabilityFunction{FT} <: AbstractStabilityFunction
    ζmax :: FT = 50.0
    A⁺   :: FT = 0.35
    B⁺   :: FT = 0.7
    C⁺   :: FT = 0.75
    D⁺   :: FT = 5/0.35
    A⁻   :: FT = 15.0
    B⁻   :: FT = 2.0
    C⁻   :: FT = π/2
    D⁻   :: FT = 10.15
    E⁻   :: FT = 3.0
    F⁻   :: FT = π / sqrt(3)
end

@inline function stability_profile(ψ::EdsonMomentumStabilityFunction, ζ)
    ζmax = ψ.ζmax
    A⁺   = ψ.A⁺
    B⁺   = ψ.B⁺
    C⁺   = ψ.C⁺
    D⁺   = ψ.D⁺
    A⁻   = ψ.A⁻
    B⁻   = ψ.B⁻
    C⁻   = ψ.C⁻
    D⁻   = ψ.D⁻
    E⁻   = ψ.E⁻
    F⁻   = ψ.F⁻

    ζ⁻ = min(zero(ζ), ζ)
    ζ⁺ = max(zero(ζ), ζ)
    dζ = min(ζmax, A⁺ * ζ⁺)

    # Stability parameter for _stable_ atmospheric conditions
    ψ⁺ = - B⁺ * ζ⁺ - C⁺ * (ζ⁺ - D⁺) * exp(- dζ) - C⁺ * D⁺

    # Stability parameter for _unstable_ atmospheric conditions
    f⁻₁ = sqrt(sqrt(1 - A⁻ * ζ⁻))
    ψ⁻₁ = B⁻ * log((1 + f⁻₁) / B⁻) + log((1 + f⁻₁^2) / B⁻) - B⁻ * atan(f⁻₁) + C⁻

    f⁻₂ = cbrt(1 - D⁻ * ζ⁻)
    ψ⁻₂ = E⁻ / 2 * log((1 + f⁻₂ + f⁻₂^2) / E⁻) - sqrt(E⁻) * atan( (1 + 2f⁻₂) / sqrt(E⁻)) + F⁻

    f  = ζ⁻^2 / (1 + ζ⁻^2)
    ψ⁻ = (1 - f) * ψ⁻₁ + f * ψ⁻₂

    return ifelse(ζ < 0, ψ⁻, ψ⁺)
end

"""
    EdsonScalarStabilityFunction{FT}

A struct representing the scalar stability function detailed by [edson2013exchange](@citet).
The formulation hinges on the definition of two different functions:
one for stable atmospheric conditions ``(ζ > 0)``, named ``ψ⁺`` and one for unstable conditions,
named ``ψ⁻``.

These stability functions are obtained by regression to experimental data.

The stability parameter for stable atmospheric conditions is defined as

```math
\\begin{align*}
dζ &= \\min(ζ_{\\max}, A⁺ζ) \\\\
ψ⁺ &= - (1 + B⁺ ζ)^{C⁺} - B⁺ (ζ - D⁺) \\exp( - dζ) - E⁺
\\end{align*}
```

While the stability parameter for unstable atmospheric conditions is calculated
as a function of the two individual stability functions as follows
```math
\\begin{align*}
f⁻₁ &= √(1 - A⁻ζ) \\\\
ψ⁻₁ &= B⁻ \\log[(1 + f⁻₁) / B⁻] + C⁻ \\\\
\\\\
f⁻₂ &= ∛(1 - D⁻ζ) \\\\
ψ⁻₂ &= (E⁻ / 2) \\log[(1 + f⁻₂ + f⁻₂²) / E⁻] - √E⁻ \\mathrm{atan}[(1 + 2f⁻₂) / √E⁻] + F⁻ \\\\
\\\\
f   &= ζ² / (1 + ζ²) \\\\
ψ⁻  &= (1 - f) ψ⁻₁ + f ψ⁻₂
\\end{align*}
```

The superscripts ``+`` and ``-`` indicate if the parameter applies to the
stability function for _stable_ or _unstable_ atmospheric conditions, respectively.
"""
@kwdef struct EdsonScalarStabilityFunction{FT} <: AbstractStabilityFunction
    ζmax :: FT = 50.0
    A⁺   :: FT = 0.35
    B⁺   :: FT = 2/3
    C⁺   :: FT = 3/2
    D⁺   :: FT = 14.28
    E⁺   :: FT = 8.525
    A⁻   :: FT = 15.0
    B⁻   :: FT = 2.0
    C⁻   :: FT = 0.0
    D⁻   :: FT = 34.15
    E⁻   :: FT = 3.0
    F⁻   :: FT = π / sqrt(3)
end

@inline function stability_profile(ψ::EdsonScalarStabilityFunction, ζ)
    ζmax = ψ.ζmax
    A⁺   = ψ.A⁺
    B⁺   = ψ.B⁺
    C⁺   = ψ.C⁺
    D⁺   = ψ.D⁺
    E⁺   = ψ.E⁺
    A⁻   = ψ.A⁻
    B⁻   = ψ.B⁻
    C⁻   = ψ.C⁻
    D⁻   = ψ.D⁻
    E⁻   = ψ.E⁻
    F⁻   = ψ.F⁻

    ζ⁻ = min(zero(ζ), ζ)
    ζ⁺ = max(zero(ζ), ζ)
    dζ = min(ζmax, A⁺ * ζ⁺)

    # stability function for stable atmospheric conditions
    ψ⁺ = - (1 + B⁺ * ζ⁺)^C⁺ - B⁺ * (ζ⁺ - D⁺) * exp(-dζ) - E⁺

    # Stability parameter for _unstable_ atmospheric conditions
    f⁻₁ = sqrt(1 - A⁻ * ζ⁻)
    ψ⁻₁ = B⁻ * log((1 + f⁻₁) / B⁻) + C⁻

    f⁻₂ = cbrt(1 - D⁻ * ζ⁻)
    ψ⁻₂ = E⁻ / 2 * log((1 + f⁻₂ + f⁻₂^2) / E⁻) - sqrt(E⁻) * atan((1 + 2f⁻₂) / sqrt(E⁻)) + F⁻

    f  = ζ⁻^2 / (1 + ζ⁻^2)
    ψ⁻ = (1 - f) * ψ⁻₁ + f * ψ⁻₂

    return ifelse(ζ < 0, ψ⁻, ψ⁺)
end

# Edson et al. (2013)
function atmosphere_ocean_stability_functions(FT=Oceananigans.defaults.FloatType)
    ψu = EdsonMomentumStabilityFunction{FT}()
    ψc = EdsonScalarStabilityFunction{FT}()
    return SimilarityScales(ψu, ψc, ψc)
end

Base.summary(::EdsonMomentumStabilityFunction{FT}) where FT = "EdsonMomentumStabilityFunction{$FT}"
Base.summary(::EdsonScalarStabilityFunction{FT}) where FT = "EdsonScalarStabilityFunction{$FT}"

Base.show(io, ::EdsonMomentumStabilityFunction{FT}) where FT = print(io, "EdsonMomentumStabilityFunction{$FT}")
Base.show(io, ::EdsonScalarStabilityFunction{FT}) where FT = print(io, "EdsonScalarStabilityFunction{$FT}")

#####
##### From Grachev et al. (2007), for stable boundary layers
#####

@kwdef struct ShebaMomentumStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 6.5
    b :: FT = 1.3
end

# @inline (ψ::ShebaMomentumStabilityFunction)(ζ) = 1 + ψ.a * ζ * cbrt(1 + ζ) / (ψ.b + ζ)
@inline function stability_profile(ψ::ShebaMomentumStabilityFunction, ζ)
    a = ψ.a
    b = ψ.b
    ζ⁺ = max(zero(ζ), ζ)
    z = cbrt(1 + ζ⁺)
    B = cbrt((1 - b) / b)

    rt3 = sqrt(3)
    Ψ₁ = - 3 * a * (z - 1) / b
    Ψ₂ = a * B / 2b * (2 * log((z + B) / (1 + B))
                       - log((z^2 - B * z + B^2) / (1 - B + B^2))
                       + 2 * rt3 * (atan((2z - B) / (rt3 * B)) - atan((2 - B) / (rt3 * B))))

    return Ψ₁ + Ψ₂
end

@kwdef struct ShebaScalarStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 5.0
    b :: FT = 5.0
    c :: FT = 3.0
end

@inline function stability_profile(ψ::ShebaScalarStabilityFunction, ζ)
    a = ψ.a
    b = ψ.b
    c = ψ.c
    B = sqrt(c^2 - 4)
    ζ⁺ = max(zero(ζ), ζ)

    Ψ₁ = - b/2 * log(1 + c * ζ⁺ + ζ⁺^2)
    Ψ₂ = (b * c / 2B - a / B) *
        (log((2ζ⁺ + c - B) / (2ζ⁺ + c + B)) - log((c - B) / (c + B)))

    return Ψ₁ + Ψ₂
end

#####
##### From Paulson (1970), for unstable boundary layers
#####

# Integrated Businger--Dyer profiles, shared with the free-convection matched
# functions below.
@inline function businger_dyer_momentum_profile(a, b, ζ⁻)
    z = sqrt(sqrt(1 - a * ζ⁻))
    return 2 * log((1 + z) / 2) + log((1 + z^2) / 2) - 2 * atan(z) + b
end

@inline businger_dyer_scalar_profile(a, ζ⁻) = 2 * log((1 + sqrt(1 - a * ζ⁻)) / 2)

@kwdef struct PaulsonMomentumStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 16.0
    b :: FT = π/2
end

@inline stability_profile(ψ::PaulsonMomentumStabilityFunction, ζ) =
    businger_dyer_momentum_profile(ψ.a, ψ.b, min(zero(ζ), ζ))

@kwdef struct PaulsonScalarStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 16.0
end

@inline stability_profile(ψ::PaulsonScalarStabilityFunction, ζ) =
    businger_dyer_scalar_profile(ψ.a, min(zero(ζ), ζ))

#####
##### From Zeng et al. (1998), matching the unstable branch to free convection
#####

"""
    FreeConvectionMomentumStabilityFunction{FT}

Unstable momentum stability function of [zeng1998intercomparison](@citet): the
Businger--Dyer profile up to a matching point ``ζ_m``, and the free-convection
profile beyond it,

```math
ψ(ζ) = ψ_{BD}(-ζ_m) + \\log \\left ( \\frac{|ζ|}{ζ_m} \\right )
       - C \\left ( |ζ|^{1/3} - ζ_m^{1/3} \\right ) , \\qquad ζ < -ζ_m ,
```

which is continuously differentiable at ``-ζ_m``. The default
`matching_stability_parameter` and `free_convection_coefficient` produce that
``C^1`` match for `a = 16`; retune them together with `a`.

The unmatched Businger--Dyer form is ill-posed in the free-convection limit: ``ψ``
grows like ``\\log(4 |ζ|)``, so ``ψ(h / L_★) - ψ(ℓ / L_★) → \\log(h / ℓ)`` and the
similarity profile [`similarity_profile`](@ref) collapses to zero, with unbounded
transfer coefficients. [businger1971flux](@citet) note the underlying problem: as
``u_★ → 0`` the profile needs a velocity scale of the large convective eddies,
which Monin--Obukhov similarity does not supply (here, [`ConvectiveGustiness`](@ref)
does). The matched profile grows like ``|ζ|^{1/3}`` instead.

`maximum_stability_parameter` bounds ``|ζ|`` in the profile, as CLM bounds the
unstable stability parameter within its flux iteration. It is load-bearing: the
matched momentum profile sends ``χ_u → 0`` like ``|ζ|^{-1/3}``, which admits a
spurious turbulent-shutoff solution (``u_★ → 0`` with the surface far warmer than
the air) in dead-calm convection under a shallow boundary layer. Freezing the
profile beyond the bound removes that solution.
"""
@kwdef struct FreeConvectionMomentumStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 16.0
    b :: FT = π/2
    matching_stability_parameter :: FT = 1.574
    free_convection_coefficient :: FT = 1.14
    maximum_stability_parameter :: FT = 100.0
end

@inline function stability_profile(ψ::FreeConvectionMomentumStabilityFunction, ζ)
    ζₘ = ψ.matching_stability_parameter
    ζ⁻ = min(zero(ζ), max(ζ, -ψ.maximum_stability_parameter))

    # |ζ| clipped from below at the match, so both matched terms vanish above it
    ζᶠ = max(-ζ⁻, ζₘ)
    ψᴮᴰ = businger_dyer_momentum_profile(ψ.a, ψ.b, max(ζ⁻, -ζₘ))

    return ψᴮᴰ + log(ζᶠ / ζₘ) - ψ.free_convection_coefficient * (cbrt(ζᶠ) - cbrt(ζₘ))
end

"""
    FreeConvectionScalarStabilityFunction{FT}

Unstable scalar stability function of [zeng1998intercomparison](@citet), matched to
free convection at ``ζ_t`` as in [`FreeConvectionMomentumStabilityFunction`](@ref) with the
scalar free-convection increment,

```math
ψ(ζ) = ψ_{BD}(-ζ_t) + \\log \\left ( \\frac{|ζ|}{ζ_t} \\right )
       - C \\left ( ζ_t^{-1/3} - |ζ|^{-1/3} \\right ) , \\qquad ζ < -ζ_t .
```
"""
@kwdef struct FreeConvectionScalarStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 16.0
    matching_stability_parameter :: FT = 0.465
    free_convection_coefficient :: FT = 0.8
    maximum_stability_parameter :: FT = 100.0
end

@inline function stability_profile(ψ::FreeConvectionScalarStabilityFunction, ζ)
    ζₜ = ψ.matching_stability_parameter
    ζ⁻ = min(zero(ζ), max(ζ, -ψ.maximum_stability_parameter))

    ζᶠ = max(-ζ⁻, ζₜ)
    ψᴮᴰ = businger_dyer_scalar_profile(ψ.a, max(ζ⁻, -ζₜ))

    return ψᴮᴰ + log(ζᶠ / ζₜ) - ψ.free_convection_coefficient * (1 / cbrt(ζₜ) - 1 / cbrt(ζᶠ))
end

Base.summary(::FreeConvectionMomentumStabilityFunction{FT}) where FT = "FreeConvectionMomentumStabilityFunction{$FT}"
Base.summary(::FreeConvectionScalarStabilityFunction{FT}) where FT = "FreeConvectionScalarStabilityFunction{$FT}"

Base.show(io::IO, ψ::FreeConvectionMomentumStabilityFunction) = print(io, summary(ψ))
Base.show(io::IO, ψ::FreeConvectionScalarStabilityFunction) = print(io, summary(ψ))

struct SplitStabilityFunction{S, U}
    stable :: S
    unstable :: U
end

Base.summary(ss::SplitStabilityFunction) = "SplitStabilityFunction"
Base.show(io::IO, ss::SplitStabilityFunction) = print(io, "SplitStabilityFunction")

@inline function stability_profile(ψ::SplitStabilityFunction, ζ)
    Ψ_stable = stability_profile(ψ.stable, ζ)
    Ψ_unstable = stability_profile(ψ.unstable, ζ)
    stable = ζ > 0
    return ifelse(stable, Ψ_stable, Ψ_unstable)
end

"""
$(TYPEDSIGNATURES)

Stability function evaluated at the roughness height by [`similarity_profile`](@ref).

The free-convection matched functions split the profile at ``z = -ζ_m L_★`` and keep
the Businger--Dyer form below the match, so their roughness-height term is the
*unmatched* function. Every other stability function evaluates identically at both
heights.
"""
@inline inner_stability_profile(ψ, ζ) = stability_profile(ψ, ζ)

@inline inner_stability_profile(ψ::FreeConvectionMomentumStabilityFunction, ζ) =
    businger_dyer_momentum_profile(ψ.a, ψ.b, min(zero(ζ), ζ))

@inline inner_stability_profile(ψ::FreeConvectionScalarStabilityFunction, ζ) =
    businger_dyer_scalar_profile(ψ.a, min(zero(ζ), ζ))

@inline function inner_stability_profile(ψ::SplitStabilityFunction, ζ)
    Ψ_stable = stability_profile(ψ.stable, ζ)
    Ψ_unstable = inner_stability_profile(ψ.unstable, ζ)
    return ifelse(ζ > 0, Ψ_stable, Ψ_unstable)
end

#####
##### Linear stable stability function (ψ = -c ζ, bounded)
#####

"""
    LinearStableStabilityFunction{FT}

A simple linear stability function for stable conditions: ``ψ = -c ζ``,
bounded at ``|ζ| ≤ ζ_{max}``.

Used by the NCAR/Large-Yeager (2004) bulk formulae with ``c = 5`` and ``ζ_{max} = 10``.

References:
- Large, W.G. & Yeager, S.G. (2004): NCAR/TN-460+STR
"""
@kwdef struct LinearStableStabilityFunction{FT} <: AbstractStabilityFunction
    coefficient :: FT = 5.0
    maximum_stability_parameter :: FT = 10.0
end

@inline function stability_profile(ψ::LinearStableStabilityFunction, ζ)
    c = ψ.coefficient
    ζmax = ψ.maximum_stability_parameter
    ζ⁺ = max(zero(ζ), ζ)
    return -c * min(ζ⁺, ζmax)
end

Base.summary(::LinearStableStabilityFunction{FT}) where FT = "LinearStableStabilityFunction{$FT}"
Base.show(io::IO, ::LinearStableStabilityFunction{FT}) where FT = print(io, "LinearStableStabilityFunction{$FT}")

"""
    large_yeager_stability_functions(FT = Float64)

NCAR/Large-Yeager (2004) stability functions combining:
- Unstable: Paulson (1970) with γ = 16
- Stable: linear ψ = -5ζ, bounded at |ζ| ≤ 10

Used for OMIP-2 protocol compliance.
"""
function large_yeager_stability_functions(FT=Oceananigans.defaults.FloatType)
    stable   = LinearStableStabilityFunction{FT}()
    momentum = SplitStabilityFunction(stable, PaulsonMomentumStabilityFunction{FT}())
    scalar   = SplitStabilityFunction(stable, PaulsonScalarStabilityFunction{FT}())
    return SimilarityScales(momentum, scalar, scalar)
end

"""
$(TYPEDSIGNATURES)

Stability functions for the atmosphere--land interface: the free-convection matched
unstable functions of [zeng1998intercomparison](@citet)
([`FreeConvectionMomentumStabilityFunction`](@ref), [`FreeConvectionScalarStabilityFunction`](@ref)) and
the linear stable function ``ψ = -5 ζ``, bounded at ``ζ ≤`` `stable_maximum_stability_parameter`.

Both branches of the plain Businger--Dyer form used over the ocean misbehave over
land, where calm transitions push ``|ζ|`` far outside its fitted range:

  - unstable: ``ψ`` grows like ``\\log(4|ζ|)``, so the similarity profile
    ``\\log(h / ℓ) - ψ(h / L_★) + ψ(ℓ / L_★)`` collapses toward zero and the transfer
    coefficients grow without bound. The matched form removes the collapse.
  - stable: extrapolating ``ψ = -5 ζ`` to the Large--Yeager ocean bound ``ζ ≤ 10``
    cuts each transfer coefficient by ``\\sim 10 ×``, a near-decoupled state that
    drives excessive surface cooling [louis1979parametric](@citep) and suppresses
    turbulence that observations sustain to ``Ri_b ≥ 1`` [best2001modelling](@citep).
    Bounding the stable branch is standard land practice (CLM4.5 at ``ζ = 2``,
    CLM5 at ``0.5``, Noah-MP at ``1``).

Ocean and sea-ice stability functions are unchanged.
"""
function atmosphere_land_stability_functions(FT=Oceananigans.defaults.FloatType;
                                             maximum_stability_parameter = 100,
                                             stable_maximum_stability_parameter = 2)
    ζᵐᵃˣ = convert(FT, maximum_stability_parameter)
    ζᵐᵃˣ⁺ = convert(FT, stable_maximum_stability_parameter)
    stable   = LinearStableStabilityFunction{FT}(coefficient = 5,
                                                 maximum_stability_parameter = ζᵐᵃˣ⁺)
    momentum = SplitStabilityFunction(stable,
                   FreeConvectionMomentumStabilityFunction{FT}(; maximum_stability_parameter = ζᵐᵃˣ))
    scalar   = SplitStabilityFunction(stable,
                   FreeConvectionScalarStabilityFunction{FT}(; maximum_stability_parameter = ζᵐᵃˣ))
    return SimilarityScales(momentum, scalar, scalar)
end

function atmosphere_sea_ice_stability_functions(FT=Oceananigans.defaults.FloatType)
    unstable_momentum = PaulsonMomentumStabilityFunction{FT}()
    stable_momentum = ShebaMomentumStabilityFunction{FT}()
    momentum = SplitStabilityFunction(stable_momentum, unstable_momentum)

    unstable_scalar = PaulsonScalarStabilityFunction{FT}()
    stable_scalar = ShebaScalarStabilityFunction{FT}()
    scalar = SplitStabilityFunction(stable_scalar, unstable_scalar)

    return SimilarityScales(momentum, scalar, scalar)
end

function atmosphere_sea_ice_similarity_theory(FT=Oceananigans.defaults.FloatType)
    stability_functions = atmosphere_sea_ice_stability_functions(FT)
    return SimilarityTheoryFluxes(FT; stability_functions)
end
