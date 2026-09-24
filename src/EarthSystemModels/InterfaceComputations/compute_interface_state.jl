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

struct FixedIterations{I}
    iterations :: I
end

@inline iterating(Ψⁿ, Ψ⁻, iteration, fixed::FixedIterations) = iteration < fixed.iterations

#####
##### The solver
#####

# Iterating condition for the characteristic scales solvers
@inline function compute_interface_state(interface_formulation,
                                         initial_interface_state,
                                         atmosphere_state,
                                         interior_state,
                                         radiation_state,
                                         atmosphere_properties,
                                         interior_properties)

    Ψₐ = atmosphere_state
    Ψᵢ = interior_state
    Ψₛⁿ = Ψₛ⁻ = initial_interface_state
    stop_criteria = interface_formulation.turbulent_fluxes.solver_stop_criteria
    iteration = 0

    while iterating(Ψₛⁿ, Ψₛ⁻, iteration, stop_criteria)
        Ψₛ⁻ = Ψₛⁿ
        Ψₛⁿ = iterate_interface_state(interface_formulation,
                                      Ψₛ⁻, Ψₐ, Ψᵢ,
                                      radiation_state,
                                      atmosphere_properties,
                                      interior_properties)
        iteration += 1
    end

    return Ψₛⁿ
end

"""
    iterate_interface_state(interface_formulation, Ψₛⁿ⁻¹, Ψₐ, Ψᵢ, Qᵣ, ℙₐ, ℙᵢ)

Return the nth iterate of the interface state `Ψₛⁿ` computed according to the
`interface_formulation`, given the interface state at the previous iterate `Ψₛⁿ⁻¹`,
as well as the atmosphere state `Ψₐ`, the interior state `Ψᵢ`,
downwelling radiation `Qᵣ`, and the atmosphere and interior properties `ℙₐ` and `ℙᵢ`.
"""
@inline function iterate_interface_state(interface_formulation,
                                         approximate_interface_state,
                                         atmosphere_state,
                                         interior_state,
                                         radiation_state,
                                         atmosphere_properties,
                                         interior_properties)

    Tₛ = compute_interface_temperature(interface_formulation.temperature,
                                       approximate_interface_state,
                                       atmosphere_state,
                                       interior_state,
                                       radiation_state,
                                       interface_formulation,
                                       atmosphere_properties,
                                       interior_properties)

    FT = eltype(approximate_interface_state)

    # Recompute the interface specific humidity from the just-updated temperature.
    # Diagnostic formulations (`ImpureSaturationSpecificHumidity`, `BulkHumidity`)
    # evaluate qₛ explicitly; `SkinHumidity` solves a vapor-flux balance for qₛ
    # using the previous iterate's turbulent fluxes (analog of `SkinTemperature`).
    qᵃᵗ = atmosphere_state.q
    qₛ = compute_interface_humidity(interface_formulation.specific_humidity, Tₛ,
                                    approximate_interface_state,
                                    atmosphere_state,
                                    interior_state,
                                    atmosphere_properties)

    # Compute the specific humidity increment
    Δq = qᵃᵗ - qₛ

    θᵃᵗ = surface_atmosphere_temperature(atmosphere_state, atmosphere_properties)
    Δθ = θᵃᵗ - Tₛ
    Δh = atmosphere_state.z # Assumption! The surface is at z = 0 -> Δh = zᵃᵗ - 0

    u★, θ★, q★ = iterate_interface_fluxes(interface_formulation.turbulent_fluxes,
                                          Tₛ, qₛ, Δθ, Δq, Δh,
                                          approximate_interface_state,
                                          atmosphere_state,
                                          interface_formulation,
                                          atmosphere_properties)

    fluxes = InterfaceFluxScales(convert(FT, u★), convert(FT, θ★), convert(FT, q★))

    return rebuild_interface_state(approximate_interface_state,
                                   fluxes,
                                   convert(FT, Tₛ),
                                   convert(FT, qₛ))
end
