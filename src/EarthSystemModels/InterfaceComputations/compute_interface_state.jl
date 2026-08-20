#####
##### Solver stop criteria
#####

struct ConvergenceStopCriteria{FT}
    tolerance :: FT
    maxiter :: Int
end

@inline function iterating(Ψⁿ, Ψ⁻, iteration, convergence::ConvergenceStopCriteria)
    maxiter = convergence.maxiter
    tolerance = convergence.tolerance
    hasnt_started = iteration == 0
    reached_maxiter = iteration ≥ maxiter
    drift = abs(Ψⁿ.fluxes.u★ - Ψ⁻.fluxes.u★) + abs(Ψⁿ.fluxes.θ★ - Ψ⁻.fluxes.θ★) + abs(Ψⁿ.fluxes.q★ - Ψ⁻.fluxes.q★)
    converged = drift < tolerance
    return !(converged | reached_maxiter) | hasnt_started
end

"""
    FixedIterations(iterations; relaxation = 1)

Stop criteria running the interface fixed point for a fixed number of
`iterations` — every GPU thread follows the same path, unlike the
tolerance-based `ConvergenceStopCriteria`. `relaxation` under-relaxes
the characteristic scales across iterates,

```math
u★ⁿ ← λ u★ⁿ + (1 - λ) u★ⁿ⁻¹,
```

(likewise `θ★`, `q★`, and the transfer coefficients). The composed
similarity–surface map is steep at calm stability transitions, where convective
gustiness switches on and off with the sign of the buoyancy flux and plain
alternation falls into a limit cycle that exits at arbitrary phase; `λ = 1/2`
restores contraction there without changing the fixed point. The default
`λ = 1` keeps plain alternation.
"""
struct FixedIterations{I, FT}
    iterations :: I
    relaxation :: FT
end

FixedIterations(iterations; relaxation = 1) = FixedIterations(iterations, relaxation)

@inline iterating(Ψⁿ, Ψ⁻, iteration, fixed::FixedIterations) = iteration < fixed.iterations

# Under-relaxation of the characteristic scales between iterates (see
# `FixedIterations`). The tolerance-based criteria keep plain alternation.
@inline relax_interface_state(stop_criteria, Ψⁿ, Ψ⁻) = Ψⁿ

@inline function relax_interface_state(fixed::FixedIterations, Ψⁿ, Ψ⁻)
    FT = eltype(Ψⁿ)
    λ  = convert(FT, fixed.relaxation)
    fⁿ = Ψⁿ.fluxes
    f⁻ = Ψ⁻.fluxes
    fluxes = InterfaceFluxScales(λ * fⁿ.u★ + (1 - λ) * f⁻.u★,
                                 λ * fⁿ.θ★ + (1 - λ) * f⁻.θ★,
                                 λ * fⁿ.q★ + (1 - λ) * f⁻.q★,
                                 λ * fⁿ.χθ + (1 - λ) * f⁻.χθ,
                                 λ * fⁿ.χq + (1 - λ) * f⁻.χq)
    return rebuild_interface_state(Ψⁿ, fluxes, Ψⁿ.temperature, Ψⁿ.specific_humidity)
end

#####
##### The solver
#####

# Iterating condition for the characteristic scales solvers
@inline function compute_interface_state(flux_formulation,
                                         initial_interface_state,
                                         atmosphere_state,
                                         interior_state,
                                         radiation_state,
                                         interface_properties,
                                         atmosphere_properties,
                                         interior_properties)

    Ψₐ = atmosphere_state
    Ψᵢ = interior_state
    Ψₛⁿ = Ψₛ⁻ = initial_interface_state
    stop_criteria = flux_formulation.solver_stop_criteria
    iteration = 0

    while iterating(Ψₛⁿ, Ψₛ⁻, iteration, stop_criteria)
        Ψₛ⁻ = Ψₛⁿ
        Ψₛⁿ = iterate_interface_state(flux_formulation,
                                      Ψₛ⁻, Ψₐ, Ψᵢ,
                                      radiation_state,
                                      interface_properties,
                                      atmosphere_properties,
                                      interior_properties)
        Ψₛⁿ = relax_interface_state(stop_criteria, Ψₛⁿ, Ψₛ⁻)
        iteration += 1
    end

    return Ψₛⁿ
end

# Interface temperature and specific humidity for one iterate. Split formulations
# compute them in sequence (humidity from the just-updated temperature): diagnostic
# formulations (`ImpureSaturationSpecificHumidity`, `BulkHumidity`) evaluate qₛ explicitly,
# while `SkinHumidity` solves a vapor-flux balance using the previous iterate's turbulent
# fluxes (analog of `SkinTemperature`). A combined formulation (a `CanopyAirSpace` in both
# interface slots) overrides this to solve the coupled node once and return both, instead
# of running the inner solve twice.
@inline function interface_temperature_and_humidity(temperature_formulation, humidity_formulation,
                                                    Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ)
    Tₛ = compute_interface_temperature(temperature_formulation, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ)
    qₛ = compute_interface_humidity(humidity_formulation, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    return Tₛ, qₛ
end

"""
    iterate_interface_state(flux_formulation, Ψₛⁿ⁻¹, Ψₐ, Ψᵢ, Qᵣ, ℙₛ, ℙₐ, ℙᵢ)

Return the nth iterate of the interface state `Ψₛⁿ` computed according to the
`flux_formulation`, given the interface state at the previous iterate `Ψₛⁿ⁻¹`,
as well as the atmosphere state `Ψₐ`, the interior state `Ψᵢ`,
downwelling radiation `Qᵣ`, and the interface, atmosphere,
and interior properties `ℙₛ`, `ℙₐ`, and `ℙᵢ`.
"""
@inline function iterate_interface_state(flux_formulation,
                                         approximate_interface_state,
                                         atmosphere_state,
                                         interior_state,
                                         radiation_state,
                                         interface_properties,
                                         atmosphere_properties,
                                         interior_properties)

    FT = eltype(approximate_interface_state)

    qᵃᵗ = atmosphere_state.q
    Tₛ, qₛ = interface_temperature_and_humidity(interface_properties.temperature_formulation,
                                                interface_properties.specific_humidity_formulation,
                                                approximate_interface_state,
                                                atmosphere_state,
                                                interior_state,
                                                radiation_state,
                                                interface_properties,
                                                atmosphere_properties,
                                                interior_properties)

    # Compute the specific humidity increment
    Δq = qᵃᵗ - qₛ

    θᵃᵗ = surface_atmosphere_temperature(atmosphere_state, atmosphere_properties)
    Δθ = θᵃᵗ - Tₛ
    Δh = atmosphere_state.z # Assumption! The surface is at z = 0 -> Δh = zᵃᵗ - 0

    u★, θ★, q★, χθ, χq = iterate_interface_fluxes(flux_formulation,
                                                  Tₛ, qₛ, Δθ, Δq, Δh,
                                                  approximate_interface_state,
                                                  atmosphere_state,
                                                  interface_properties,
                                                  atmosphere_properties,
                                                  interior_properties)

    fluxes = InterfaceFluxScales(convert(FT, u★), convert(FT, θ★), convert(FT, q★),
                                 convert(FT, χθ), convert(FT, χq))

    return rebuild_interface_state(approximate_interface_state,
                                   fluxes,
                                   convert(FT, Tₛ),
                                   convert(FT, qₛ))
end
