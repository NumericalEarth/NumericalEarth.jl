using DocStringExtensions

#####
##### Polynomial neutral drag coefficient (Large & Yeager 2004)
#####

"""
    PolynomialNeutralDragCoefficient{FT}

Neutral 10-m drag coefficient from the Large & Yeager (2004) polynomial:

    C_dN = (a/U + b + c*U - d*U^6) × 10⁻³   for U < high_wind_speed_threshold
    C_dN = high_wind_drag_coefficient         otherwise

Called as a functor: `C_dN = drag(U)`.

References:
- Large, W.G. & Yeager, S.G. (2004): NCAR/TN-460+STR
"""
struct PolynomialNeutralDragCoefficient{FT}
    a :: FT
    b :: FT
    c :: FT
    d :: FT
    high_wind_speed_threshold :: FT
    high_wind_drag_coefficient :: FT
    minimum_wind_speed :: FT
end

function PolynomialNeutralDragCoefficient(FT = Oceananigans.defaults.FloatType;
                                          a = 2.7,
                                          b = 0.142,
                                          c = 1 / 13.09,
                                          d = 3.14807e-10,
                                          high_wind_speed_threshold = 33,
                                          high_wind_drag_coefficient = 2.34e-3,
                                          minimum_wind_speed = 0.5)

    return PolynomialNeutralDragCoefficient(convert(FT, a),
                                            convert(FT, b),
                                            convert(FT, c),
                                            convert(FT, d),
                                            convert(FT, high_wind_speed_threshold),
                                            convert(FT, high_wind_drag_coefficient),
                                            convert(FT, minimum_wind_speed))
end

@inline function (p::PolynomialNeutralDragCoefficient)(U)
    U = max(U, p.minimum_wind_speed)
    Cd = ifelse(U < p.high_wind_speed_threshold,
                (p.a / U + p.b + p.c * U - p.d * U^6) / 1000,
                p.high_wind_drag_coefficient)
    return Cd
end

Base.summary(::PolynomialNeutralDragCoefficient{FT}) where FT =
    "PolynomialNeutralDragCoefficient{$FT}"

#####
##### Large & Yeager transfer coefficients (L&Y 2004 with stability corrections)
#####

"""
    LargeYeagerTransferCoefficients{FT, D, SF}

Wind-dependent bulk transfer coefficients following the NCAR/Large & Yeager (2004)
algorithm as used in OMIP-2 simulations.

Computes all three transfer coefficients (drag, heat, moisture) from a neutral drag
polynomial with stability corrections via Monin-Obukhov similarity functions. The empirical
heat and moisture coefficients correspond to the Stanton and Dalton numbers of L&Y eq. 6c-6d.

Pass as `transfer_coefficients` to `CoefficientBasedFluxes`:

    ly = LargeYeagerTransferCoefficients()
    fluxes = CoefficientBasedFluxes(transfer_coefficients = ly,
                                    solver_stop_criteria = FixedIterations(5))

References:
- Large, W.G. & Yeager, S.G. (2004): NCAR/TN-460+STR
"""
struct LargeYeagerTransferCoefficients{FT, D, SF}
    von_karman_constant :: FT
    neutral_drag_coefficient :: D
    stability_functions :: SF
    reference_height :: FT
    stable_heat_transfer_coefficient :: FT
    unstable_heat_transfer_coefficient :: FT
    moisture_transfer_coefficient :: FT
end

function LargeYeagerTransferCoefficients(FT = Oceananigans.defaults.FloatType;
                                         von_karman_constant = 0.4,
                                         neutral_drag_coefficient = PolynomialNeutralDragCoefficient(FT),
                                         stability_functions = large_yeager_stability_functions(FT),
                                         reference_height = 10,
                                         stable_heat_transfer_coefficient = 18,
                                         unstable_heat_transfer_coefficient = 32.7,
                                         moisture_transfer_coefficient = 34.6)

    return LargeYeagerTransferCoefficients(convert(FT, von_karman_constant),
                                                   neutral_drag_coefficient,
                                                   stability_functions,
                                                   convert(FT, reference_height),
                                                   convert(FT, stable_heat_transfer_coefficient),
                                                   convert(FT, unstable_heat_transfer_coefficient),
                                                   convert(FT, moisture_transfer_coefficient))
end

Adapt.adapt_structure(to, n::LargeYeagerTransferCoefficients) =
    LargeYeagerTransferCoefficients(Adapt.adapt(to, n.von_karman_constant),
                                    Adapt.adapt(to, n.neutral_drag_coefficient),
                                    Adapt.adapt(to, n.stability_functions),
                                    Adapt.adapt(to, n.reference_height),
                                    Adapt.adapt(to, n.stable_heat_transfer_coefficient),
                                    Adapt.adapt(to, n.unstable_heat_transfer_coefficient),
                                    Adapt.adapt(to, n.moisture_transfer_coefficient))

Base.summary(::LargeYeagerTransferCoefficients{FT}) where FT = "LargeYeagerTransferCoefficients{$FT}"

function Base.show(io::IO, n::LargeYeagerTransferCoefficients{FT}) where FT
    print(io, "LargeYeagerTransferCoefficients{$FT}", '\n')
    print(io, "├── von_karman_constant: ", n.von_karman_constant, '\n')
    print(io, "├── neutral_drag_coefficient: ", summary(n.neutral_drag_coefficient), '\n')
    print(io, "├── stability_functions: ", summary(n.stability_functions), '\n')
    print(io, "├── reference_height: ", n.reference_height, '\n')
    print(io, "├── heat transfer coefficient (stable/unstable): ", n.stable_heat_transfer_coefficient, " / ", n.unstable_heat_transfer_coefficient, '\n')
    print(io, "└── moisture transfer coefficient: ", n.moisture_transfer_coefficient)
end

#####
##### CoefficientBasedFluxes
#####

"""
    struct CoefficientBasedFluxes{C, S}

A structure for computing turbulent fluxes using bulk transfer coefficients.

$(TYPEDFIELDS)
"""
struct CoefficientBasedFluxes{C, S}
    transfer_coefficients :: C  # `SimilarityScales` with constant or callable coefficients, or an `LargeYeagerTransferCoefficients`."
    solver_stop_criteria  :: S  # "Criteria for iterative solver convergence."
end

Adapt.adapt_structure(to, f::CoefficientBasedFluxes) = CoefficientBasedFluxes(Adapt.adapt(to, f.transfer_coefficients), f.solver_stop_criteria)

Base.summary(flux_formulation::CoefficientBasedFluxes) = "CoefficientBasedFluxes"

function Base.show(io::IO, flux_formulation::CoefficientBasedFluxes)
    print(io, summary(flux_formulation), '\n')
    print(io, "├── transfer_coefficients: ", summary(flux_formulation.transfer_coefficients), '\n')
    print(io, "└── solver_stop_criteria: ",  summary(flux_formulation.solver_stop_criteria))
end

convert_if_number(FT, a::Number) = convert(FT, a)
convert_if_number(FT, a) = a

convert_transfer_coefficients(FT, c) = c
convert_transfer_coefficients(FT, c::Tuple) = convert_if_number.(FT, c)
convert_transfer_coefficients(FT, c::NamedTuple) = (; momentum    = convert_if_number(FT, c.momentum),
                                                      temperature = convert_if_number(FT, c.temperature),
                                                      water_vapor = convert_if_number(FT, c.water_vapor))

convert_transfer_coefficients(FT, c::SimilarityScales) = SimilarityScales(convert_if_number(FT, c.momentum),
                                                                          convert_if_number(FT, c.temperature),
                                                                          convert_if_number(FT, c.water_vapor))

"""
    CoefficientBasedFluxes(FT = Oceananigans.defaults.FloatType;
                           transfer_coefficients = (1e-3, 1e-3, 1e-3),
                           solver_stop_criteria = nothing,
                           solver_tolerance = 1e-8,
                           solver_maxiter = 20)

Return the structure for computing turbulent fluxes using bulk transfer coefficients.
Used in bulk flux calculations to determine the exchange of momentum, heat, and moisture
between the ocean/ice surface and the atmosphere.

Arguments
=========

- `FT`: (optional) Float type for the coefficients, defaults to Oceananigans.defaults.FloatType

Keyword Arguments
=================

- `transfer_coefficients`: Transfer coefficients for momentum, heat, and moisture.
  Can be a `SimilarityScales`, a `Tuple`, or a `NamedTuple` with constant or callable entries, 
  or an `LargeYeagerTransferCoefficients`. Defaults to `(1e-3, 1e-3, 1e-3)`.
- `solver_stop_criteria`: Criteria for iterative solver convergence. If `nothing`,
                          creates new criteria using `solver_tolerance` and `solver_maxiter`.
- `solver_tolerance`: Tolerance for solver convergence when creating new stop criteria, defaults to 1e-8.
- `solver_maxiter`: Maximum iterations for solver when creating new stop criteria, defaults to 20

Example
========

```jldoctest
using Oceananigans
using NumericalEarth

grid = RectilinearGrid(size=3, z=(-1, 0), topology=(Flat, Flat, Bounded))
ocean = ocean_simulation(grid; timestepper = :QuasiAdamsBashforth2)

ao_fluxes = CoefficientBasedFluxes(transfer_coefficients = (1e-2, 1e-3, 1e-3))

interfaces = ComponentInterfaces(nothing, ocean; atmosphere_ocean_fluxes=ao_fluxes)

# output
ComponentInterfaces
```
"""
function CoefficientBasedFluxes(FT = Oceananigans.defaults.FloatType;
                                transfer_coefficients = (1e-3, 1e-3, 1e-3),
                                solver_stop_criteria = nothing,
                                solver_tolerance = 1e-8,
                                solver_maxiter = 20)

    transfer_coefficients = validate_coefficients(FT, transfer_coefficients)
    
    if isnothing(solver_stop_criteria)
        solver_tolerance = convert(FT, solver_tolerance)
        solver_stop_criteria = ConvergenceStopCriteria(solver_tolerance, solver_maxiter)
    end

    return CoefficientBasedFluxes(transfer_coefficients, solver_stop_criteria)
end

validate_coefficients(FT, ly::LargeYeagerTransferCoefficients) = ly
validate_coefficients(FT, sc::SimilarityScales) = convert_transfer_coefficients(FT, sc)

function validate_coefficients(FT, nt::NamedTuple)
    required = (:momentum, :temperature, :water_vapor)
    missing = filter(k -> !haskey(nt, k), required)

    if !isempty(missing)
        throw(ArgumentError(
            "Transfer coefficients NamedTuple must contain keys $(required). " *
            "Missing keys: $(missing). Received: $(keys(nt))"
        ))
    end

    return convert_transfer_coefficients(FT, nt)
end

function validate_coefficients(FT, tc::Tuple)
    if length(tc) != 3
        throw(ArgumentError(
            "Transfer coefficients must be a tuple of length 3: " *
            "(momentum, temperature, water_vapor). " *
            "Got length $(length(tc)) with value $(tc)."
        ))
    end

    return convert_transfer_coefficients(FT, tc)
end

#####
##### Evaluate transfer coefficients (dispatch on coefficient type)
#####

# Minimum wind speed floor: only LargeYeager needs it; constant coefficients don't.
@inline minimum_wind_speed(::Union{Tuple, NTuple}) = 0
@inline minimum_wind_speed(::SimilarityScales) = 0
@inline minimum_wind_speed(ly::LargeYeagerTransferCoefficients) = ly.neutral_drag_coefficient.minimum_wind_speed

@inline evaluate_coefficient(C::Number, args...) = C
@inline evaluate_coefficient(C::Function, args...) = C(args...)
@inline evaluate_coefficient(C::PolynomialNeutralDragCoefficient, ΔU, args...) = C(ΔU)

@inline function evaluate_coefficients(coeffs::Union{SimilarityScales, NamedTuple}, args...)
    Cd = evaluate_coefficient(coeffs.momentum,    args...)
    Ch = evaluate_coefficient(coeffs.temperature, args...)
    Cq = evaluate_coefficient(coeffs.water_vapor, args...)
    return Cd, Ch, Cq
end

@inline function evaluate_coefficients(coeffs::Tuple, args...)
    Cd = evaluate_coefficient(coeffs[1], args...)
    Ch = evaluate_coefficient(coeffs[2], args...)
    Cq = evaluate_coefficient(coeffs[3], args...)
    return Cd, Ch, Cq
end

# NCAR transfer coefficients: full L&Y stability correction (eqs. 6c-6d, 10a-10c)
@inline function evaluate_coefficients(ly::LargeYeagerTransferCoefficients,
                                       ΔU, Tₛ, qₛ, Δh,
                                       approximate_interface_state,
                                       atmosphere_properties)

    FT  = eltype(approximate_interface_state)
    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    g   = atmosphere_properties.gravitational_acceleration

    κ  = ly.von_karman_constant
    u★ = approximate_interface_state.u★
    θ★ = approximate_interface_state.θ★
    q★ = approximate_interface_state.q★

    neutral_drag = ly.neutral_drag_coefficient
    h₀   = ly.reference_height
    Umin = neutral_drag.minimum_wind_speed
    ΔU   = max(ΔU, Umin)

    # Monin-Obukhov length from previous iteration scales
    b★ = buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)
    L★ = ifelse(b★ == 0, convert(FT, Inf), u★^2 / (κ * b★))
    ζ  = Δh / L★

    # Stability functions
    ψₘ = stability_profile(ly.stability_functions.momentum, ζ)
    ψₕ = stability_profile(ly.stability_functions.temperature, ζ)

    # Neutral 10-m wind speed
    Cdp  = ifelse(u★ == 0, neutral_drag(ΔU), u★^2 / ΔU^2)
    UN10 = ΔU / (1 + sqrt(Cdp) / κ * (log(Δh / h₀) - ψₘ))
    UN10 = max(UN10, Umin)

    # Neutral transfer coefficients
    CdN = neutral_drag(UN10)

    # Stability-dependent neutral heat (Stanton) and moisture (Dalton) numbers (L&Y eq. 6c-6d)
    stable = ζ > 0
    ChN = sqrt(CdN) / 1000 * ifelse(stable, ly.stable_heat_transfer_coefficient, ly.unstable_heat_transfer_coefficient)
    CqN = sqrt(CdN) / 1000 * ly.moisture_transfer_coefficient

    # Stability corrections (L&Y eq. 10a-10c)
    ξₘ = sqrt(CdN) / κ * (log(Δh / h₀) - ψₘ)
    Cd = CdN / (1 + ξₘ)^2

    ξₕ = sqrt(CdN) / κ * (log(Δh / h₀) - ψₕ)
    ratio = sqrt(Cd) / sqrt(CdN)

    Ch = ChN * ratio / (1 + ChN * ξₕ)
    Cq = CqN * ratio / (1 + CqN * ξₕ)

    return Cd, Ch, Cq
end

#####
##### Iteration
#####

@inline function iterate_interface_fluxes(flux_formulation::CoefficientBasedFluxes,
                                          Tₛ, qₛ, Δθ, Δq, Δh,
                                          approximate_interface_state,
                                          atmosphere_state,
                                          interface_properties,
                                          atmosphere_properties)

    Δu, Δv = velocity_difference(interface_properties.velocity_formulation,
                                 atmosphere_state,
                                 approximate_interface_state)

    Umin = minimum_wind_speed(flux_formulation.transfer_coefficients)
    ΔU   = max(sqrt(Δu^2 + Δv^2), Umin)

    Cd, Ch, Cq = evaluate_coefficients(flux_formulation.transfer_coefficients,
                                       ΔU, Tₛ, qₛ, Δh,
                                       approximate_interface_state,
                                       atmosphere_properties)

    u★ = sqrt(Cd) * ΔU
    θ★ = ifelse(Cd == 0, zero(Δθ), Ch / sqrt(Cd) * Δθ)
    q★ = ifelse(Cd == 0, zero(Δq), Cq / sqrt(Cd) * Δq)

    return u★, θ★, q★
end
