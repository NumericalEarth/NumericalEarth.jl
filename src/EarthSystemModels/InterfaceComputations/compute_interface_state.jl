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
    drift = abs(Ψⁿ.u★ - Ψ⁻.u★) + abs(Ψⁿ.θ★ - Ψ⁻.θ★) + abs(Ψⁿ.q★ - Ψ⁻.q★)
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
@inline function compute_interface_state(flux_formulation,
                                         initial_interface_state,
                                         atmosphere_state,
                                         interior_state,
                                         downwelling_radiation,
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
                                      downwelling_radiation,
                                      interface_properties,
                                      atmosphere_properties,
                                      interior_properties)

        iteration += 1
    end

    return Ψₛⁿ
end

#####
##### Solver with unrolled iterations
#####

struct TenUnrolledIterations end

@inline function compute_interface_state(flux_formulation::AbstractTurbulentFluxFormulation{<:TenUnrolledIterations},
                                         initial_interface_state,
                                         atmosphere_state,
                                         interior_state,
                                         downwelling_radiation,
                                         interface_properties,
                                         atmosphere_properties,
                                         interior_properties)

    args = (downwelling_radiation,
            interface_properties,
            atmosphere_properties,
            interior_properties)

    Ψₐ = atmosphere_state
    Ψᵢ = interior_state
    Ψₛ⁰ = initial_interface_state
    Ψₛ¹ = iterate_interface_state(flux_formulation, Ψₛ⁰, Ψₐ, Ψᵢ, args...)
    Ψₛ² = iterate_interface_state(flux_formulation, Ψₛ¹, Ψₐ, Ψᵢ, args...)
    Ψₛ³ = iterate_interface_state(flux_formulation, Ψₛ², Ψₐ, Ψᵢ, args...)
    Ψₛ⁴ = iterate_interface_state(flux_formulation, Ψₛ³, Ψₐ, Ψᵢ, args...)
    Ψₛ⁵ = iterate_interface_state(flux_formulation, Ψₛ⁴, Ψₐ, Ψᵢ, args...)
    Ψₛ⁶ = iterate_interface_state(flux_formulation, Ψₛ⁵, Ψₐ, Ψᵢ, args...)
    Ψₛ⁷ = iterate_interface_state(flux_formulation, Ψₛ⁶, Ψₐ, Ψᵢ, args...)
    Ψₛ⁸ = iterate_interface_state(flux_formulation, Ψₛ⁷, Ψₐ, Ψᵢ, args...)
    Ψₛ⁹ = iterate_interface_state(flux_formulation, Ψₛ⁸, Ψₐ, Ψᵢ, args...)

    return iterate_interface_state(flux_formulation, Ψₛ⁹, Ψₐ, Ψᵢ, args...)
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
                                         downwelling_radiation,
                                         interface_properties,
                                         atmosphere_properties,
                                         interior_properties)

    Tₛ = compute_interface_temperature(interface_properties.temperature_formulation,
                                       approximate_interface_state,
                                       atmosphere_state,
                                       interior_state,
                                       downwelling_radiation,
                                       interface_properties,
                                       atmosphere_properties,
                                       interior_properties)

    FT = eltype(approximate_interface_state)
    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters

    # Recompute the saturation specific humidity at the interface based on the new temperature
    q_formulation = interface_properties.specific_humidity_formulation
    Sₛ = approximate_interface_state.S
    Tᵃᵗ = atmosphere_state.T
    pᵃᵗ = atmosphere_state.p
    qᵃᵗ = atmosphere_state.q
    qₛ = surface_specific_humidity(q_formulation, ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, Tₛ, Sₛ)

    # Compute the specific humidity increment
    Δq = qᵃᵗ - qₛ

    θᵃᵗ = surface_atmosphere_temperature(atmosphere_state, atmosphere_properties)
    Δθ = θᵃᵗ - Tₛ
    Δh = atmosphere_state.z # Assumption! The surface is at z = 0 -> Δh = zᵃᵗ - 0

    u★, θ★, q★ = iterate_interface_fluxes(flux_formulation,
                                          Tₛ, qₛ, Δθ, Δq, Δh,
                                          approximate_interface_state,
                                          atmosphere_state,
                                          interface_properties,
                                          atmosphere_properties)

    u = approximate_interface_state.u
    v = approximate_interface_state.v
    S = approximate_interface_state.S

    return InterfaceState(convert(FT, u★),
                          convert(FT, θ★),
                          convert(FT, q★), 
                          convert(FT, u), 
                          convert(FT, v), 
                          convert(FT, Tₛ), 
                          convert(FT, S), 
                          convert(FT, qₛ))
end
