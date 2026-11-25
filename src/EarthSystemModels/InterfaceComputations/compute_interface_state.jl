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
##### Under-relaxation of an iteration that has not converged
#####

# The interface iteration is a Picard iteration of the similarity scales, together with the skin temperature.
# It does not contract everywhere: when the reference height is large compared with the Obukhov length and the
# heat and vapor fluxes contribute to the buoyancy flux with opposite signs, the stability parameter changes sign
# from one iterate to the next and the iterates wander without converging, at fixed interface temperature too.
# What the iteration then returns depends on where it started.
#
# An iteration that has used half of its `maxiter` is therefore under-relaxed for the rest: each iterate moves
# halfway from the previous one to its image. A solve that converges within `maxiter ÷ 2` iterations is
# untouched, bit for bit. Under-relaxation by 1/2 contracts an oscillation whose amplification is below 3; it is
# a safeguard, not a guarantee. `FixedIterations` is not relaxed: every thread follows the same path.
@inline relax_interface_state(stop_criteria, Ψ⁻, Ψⁿ, iteration) = Ψⁿ

@inline function relax_interface_state(convergence::ConvergenceStopCriteria, Ψ⁻, Ψⁿ, iteration)
    u★ = (Ψ⁻.fluxes.u★ + Ψⁿ.fluxes.u★) / 2
    θ★ = (Ψ⁻.fluxes.θ★ + Ψⁿ.fluxes.θ★) / 2
    q★ = (Ψ⁻.fluxes.q★ + Ψⁿ.fluxes.q★) / 2
    fluxes = InterfaceFluxScales(u★, θ★, q★, Ψⁿ.fluxes.Ξθ, Ψⁿ.fluxes.Ξq)

    Tⁱⁿ = (Ψ⁻.temperature + Ψⁿ.temperature) / 2
    qⁱⁿ = (Ψ⁻.specific_humidity + Ψⁿ.specific_humidity) / 2
    relaxed = rebuild_interface_state(Ψⁿ, fluxes, Tⁱⁿ, qⁱⁿ)

    return ifelse(iteration > convergence.maxiter ÷ 2, relaxed, Ψⁿ)
end

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
    Ψⁱⁿⁿ = Ψⁱⁿ⁻ = initial_interface_state
    stop_criteria = interface_formulation.turbulent_fluxes.solver_stop_criteria
    iteration = 0

    while iterating(Ψⁱⁿⁿ, Ψⁱⁿ⁻, iteration, stop_criteria)
        Ψⁱⁿ⁻ = Ψⁱⁿⁿ
        Ψⁱⁿⁿ = iterate_interface_state(interface_formulation,
                                      Ψⁱⁿ⁻, Ψₐ, Ψᵢ,
                                      radiation_state,
                                      atmosphere_properties,
                                      interior_properties, Δt)
        iteration += 1
        Ψⁱⁿⁿ = relax_interface_state(stop_criteria, Ψⁱⁿ⁻, Ψⁱⁿⁿ, iteration)
    end

    return Ψⁱⁿⁿ
end

@inline function interface_temperature_and_humidity(state::NamedTuple,
                                                    Ψⁱⁿ, Ψₐ, Ψᵢ, Ψᵣ, ℙⁱⁿ, ℙₐ, ℙᵢ, Δt)
    Tⁱⁿ = compute_interface_temperature(state.temperature, Ψⁱⁿ, Ψₐ, Ψᵢ, Ψᵣ, ℙⁱⁿ, ℙₐ, ℙᵢ)
    qⁱⁿ = compute_interface_humidity(state.specific_humidity, Tⁱⁿ, Ψⁱⁿ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    return Tⁱⁿ, qⁱⁿ
end

"""
$(TYPEDSIGNATURES)

Return the nth iterate of the interface state `Ψⁱⁿⁿ` computed according to the
`interface_formulation`, given the interface state at the previous iterate `Ψⁱⁿⁿ⁻¹`,
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
    Tⁱⁿ, qⁱⁿ = interface_temperature_and_humidity(interface_formulation.state,
                                                approximate_interface_state,
                                                atmosphere_state,
                                                interior_state,
                                                radiation_state,
                                                interface_formulation,
                                                atmosphere_properties,
                                                interior_properties, Δt)

    # Compute the specific humidity increment
    Δq = qᵃᵗ - qⁱⁿ

    θᵃᵗ = surface_atmosphere_temperature(atmosphere_state, atmosphere_properties)
    Δθ = θᵃᵗ - Tⁱⁿ
    Δh = atmosphere_state.z # Assumption! The surface is at z = 0 -> Δh = zᵃᵗ - 0

    u★, θ★, q★, Ξθ, Ξq = iterate_interface_fluxes(interface_formulation.turbulent_fluxes,
                                                  Tⁱⁿ, qⁱⁿ, Δθ, Δq, Δh,
                                                  approximate_interface_state,
                                                  atmosphere_state,
                                                  interface_formulation,
                                                  atmosphere_properties)

    fluxes = InterfaceFluxScales(convert(FT, u★), convert(FT, θ★), convert(FT, q★),
                                 convert(FT, Ξθ), convert(FT, Ξq))

    return rebuild_interface_state(approximate_interface_state,
                                   fluxes,
                                   convert(FT, Tⁱⁿ),
                                   convert(FT, qⁱⁿ))
end
