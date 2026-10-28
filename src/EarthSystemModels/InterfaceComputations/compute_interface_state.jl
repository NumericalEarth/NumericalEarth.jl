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
    FixedIterations(iterations)

Stop criteria running the interface fixed point for a fixed number of
`iterations` — every GPU thread follows the same path, unlike the
tolerance-based `ConvergenceStopCriteria`.
"""
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
                                         interior_properties,
                                         Δt = 0)

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
                                      interior_properties, Δt)
        iteration += 1
    end

    return Ψₛⁿ
end

@inline function interface_temperature_and_humidity(state::NamedTuple,
                                                    Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ, Δt)
    Tₛ = compute_interface_temperature(state.temperature, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ)
    qₛ = compute_interface_humidity(state.specific_humidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    return Tₛ, qₛ
end

"""
$(TYPEDSIGNATURES)

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
                                         interior_properties,
                                         Δt)

    FT = eltype(approximate_interface_state)

    qᵃᵗ = atmosphere_state.q
    Tₛ, qₛ = interface_temperature_and_humidity(interface_formulation.state,
                                                approximate_interface_state,
                                                atmosphere_state,
                                                interior_state,
                                                radiation_state,
                                                interface_formulation,
                                                atmosphere_properties,
                                                interior_properties, Δt)

    # Compute the specific humidity increment
    Δq = qᵃᵗ - qₛ

    θᵃᵗ = surface_atmosphere_temperature(atmosphere_state, atmosphere_properties)
    Δθ = θᵃᵗ - Tₛ
    Δh = atmosphere_state.z # Assumption! The surface is at z = 0 -> Δh = zᵃᵗ - 0

    u★, θ★, q★, Ξθ, Ξq = iterate_interface_fluxes(interface_formulation.turbulent_fluxes,
                                                  Tₛ, qₛ, Δθ, Δq, Δh,
                                                  approximate_interface_state,
                                                  atmosphere_state,
                                                  interface_formulation,
                                                  atmosphere_properties)

    fluxes = InterfaceFluxScales(convert(FT, u★), convert(FT, θ★), convert(FT, q★),
                                 convert(FT, Ξθ), convert(FT, Ξq))

    return rebuild_interface_state(approximate_interface_state,
                                   fluxes,
                                   convert(FT, Tₛ),
                                   convert(FT, qₛ))
end
