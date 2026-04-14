using Oceananigans.Utils: prettysummary

"""
    NCARBulkFluxes{FT, S}

Bulk turbulent flux formulation following the NCAR/Large & Yeager (2004, 2009) algorithm
as used in OMIP-2 simulations.

Unlike `SimilarityTheoryFluxes`, this formulation iterates directly on the transfer
coefficients (Cd, Ch, Ce) rather than on roughness lengths. Key features:

- Neutral 10-m drag coefficient `CdN` from empirical polynomial of neutral wind speed `UN10`
- Stability-dependent neutral Stanton number: `ChN = 18sqrtCdN` (stable) or `32.7sqrtCdN` (unstable)
- Neutral Dalton number: `CeN = 34.6sqrtCdN` (both regimes)
- Paulson (1970) / Kansas stability functions: γ = 16, stable branch = -5ζ (bounded at 10)
- 5 fixed iterations (no convergence check)
- Wind speed floor at 0.5 m/s

References:
- Large, W.G. & Yeager, S.G. (2004): NCAR/TN-460+STR
- Large, W.G. & Yeager, S.G. (2009): Climate Dynamics, 33, 341-364
- AeroBulk reference: `mod_blk_ncar.f90` (Brodeau et al.)
"""
struct NCARBulkFluxes{FT, S}
    von_karman_constant :: FT
    reference_height :: FT
    minimum_wind_speed :: FT
    solver_stop_criteria :: S
end

Adapt.adapt_structure(to, f::NCARBulkFluxes) =
    NCARBulkFluxes(Adapt.adapt(to, f.von_karman_constant),
                   Adapt.adapt(to, f.reference_height),
                   Adapt.adapt(to, f.minimum_wind_speed),
                   f.solver_stop_criteria)

Base.summary(::NCARBulkFluxes{FT}) where FT = "NCARBulkFluxes{$FT}"

function Base.show(io::IO, f::NCARBulkFluxes{FT}) where FT
    print(io, "NCARBulkFluxes{$FT}", '\n')
    print(io, "├── von_karman_constant: ", f.von_karman_constant, '\n')
    print(io, "├── reference_height: ", f.reference_height, '\n')
    print(io, "└── minimum_wind_speed: ", f.minimum_wind_speed)
end

"""
    NCARBulkFluxes(FT = Float64;
                   von_karman_constant = 0.4,
                   reference_height = 10,
                   minimum_wind_speed = 0.5,
                   solver_stop_criteria = FixedIterations(5))

Construct an NCAR/Large-Yeager bulk flux formulation with the specified parameters.
"""
function NCARBulkFluxes(FT = Oceananigans.defaults.FloatType;
                        von_karman_constant = 0.4,
                        reference_height = 10,
                        minimum_wind_speed = 0.5,
                        solver_stop_criteria = FixedIterations(5))

    return NCARBulkFluxes(convert(FT, von_karman_constant),
                          convert(FT, reference_height),
                          convert(FT, minimum_wind_speed),
                          solver_stop_criteria)
end

#####
##### NCAR helper functions (all @inline for GPU compatibility)
#####

"""Neutral 10-m drag coefficient from Large & Yeager (2004) polynomial."""
@inline function ncar_neutral_drag_coefficient(U::FT) where FT
    U = max(U, FT(0.5))
    Cd = ifelse(U < FT(33),
                (FT(2.7) / U + FT(0.142) + U / FT(13.09) - FT(3.14807e-10) * U^6) * FT(1e-3),
                FT(2.34e-3))
    return Cd
end

"""Paulson (1970) momentum stability function with γ=16, stable branch -5ζ."""
@inline function ncar_momentum_stability(ζ::FT) where FT
    if ζ < 0
        x = (1 - FT(16) * ζ) ^ FT(0.25)
        return 2 * log((1 + x) / 2) + log((1 + x^2) / 2) - 2 * atan(x) + FT(π) / 2
    else
        return -FT(5) * min(ζ, FT(10))
    end
end

"""Paulson (1970) scalar stability function with γ=16, stable branch -5ζ."""
@inline function ncar_scalar_stability(ζ::FT) where FT
    if ζ < 0
        x = sqrt(1 - FT(16) * ζ)
        return 2 * log((1 + x) / 2)
    else
        return -FT(5) * min(ζ, FT(10))
    end
end

"""Compute UN10 (neutral wind at 10m) from current Cd and stability correction."""
@inline function neutral_10m_wind(U, Cd, κ, h, ψₘ)
    # From: sqrt(Cd) = κ / [log(h/z0) - ψ_m]
    # and:  z0 = 10 * exp(-κ / sqrt(CdN))
    # UN10 = U * sqrt(CdN/Cd) * [1 + sqrt(Cd)/κ * (log(h/10) - ψ_m)]^{-1}
    # Simplified from AeroBulk: UN10 = U / (1 + sqrt(Cd)/κ * (log(h/10) - ψ_m))
    return U / (1 + sqrt(Cd) / κ * (log(h / 10) - ψₘ))
end

#####
##### The core iteration
#####

@inline function iterate_interface_fluxes(flux_formulation::NCARBulkFluxes,
                                          Tₛ, qₛ, Δθ, Δq, Δh,
                                          approximate_interface_state,
                                          atmosphere_state,
                                          interface_properties,
                                          atmosphere_properties)

    FT = eltype(approximate_interface_state)
    κ  = flux_formulation.von_karman_constant
    h₀ = flux_formulation.reference_height
    Umin = flux_formulation.minimum_wind_speed

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    g  = atmosphere_properties.gravitational_acceleration

    # Previous iteration scales
    u★ = approximate_interface_state.u★
    θ★ = approximate_interface_state.θ★
    q★ = approximate_interface_state.q★

    # Wind speed (with floor)
    Δu, Δv = velocity_difference(interface_properties.velocity_formulation,
                                 atmosphere_state,
                                 approximate_interface_state)
    U = max(sqrt(Δu^2 + Δv^2), Umin)

    # --- Compute Monin-Obukhov length from previous scales ---
    b★ = buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)
    L★ = ifelse(b★ == 0, convert(FT, Inf), u★^2 / (κ * b★))
    ζ  = Δh / L★

    # --- Stability functions ---
    ψₘ = ncar_momentum_stability(ζ)
    ψₕ = ncar_scalar_stability(ζ)

    # --- Neutral 10-m wind speed (derived from current state) ---
    # First iteration: if u★ ≈ 0 (initial guess), UN10 ≈ U
    # Subsequent: back-compute UN10 from current Cd and stability
    Cd_prev = ifelse(u★ == 0, ncar_neutral_drag_coefficient(U), u★^2 / U^2)
    UN10 = neutral_10m_wind(U, Cd_prev, κ, Δh, ψₘ)
    UN10 = max(UN10, Umin)

    # --- Neutral transfer coefficients ---
    CdN = ncar_neutral_drag_coefficient(UN10)
    sqrtCdN = sqrt(CdN)

    # Stability-dependent neutral Stanton number (L&Y eq. 6c-6d)
    stable = ζ > 0
    ChN = FT(1e-3) * sqrtCdN * ifelse(stable, FT(18), FT(32.7))
    CeN = FT(1e-3) * FT(34.6) * sqrtCdN

    # --- Apply stability corrections (L&Y eq. 10a-10c) ---
    # Cd = CdN / [1 + sqrtCdN/κ * (log(Δh/h₀) - ψₘ)]²
    ξₘ = sqrtCdN / κ * (log(Δh / h₀) - ψₘ)
    Cd = CdN / (1 + ξₘ)^2

    sqrtCd = sqrt(Cd)
    ξₕ = sqrtCdN / κ * (log(Δh / h₀) - ψₕ)
    ratio = sqrtCd / sqrtCdN

    # Ch = ChN * ratio / (1 + ChN * ξₕ)
    Ch = ChN * ratio / (1 + ChN * ξₕ)

    # Ce = CeN * ratio / (1 + CeN * ξₕ)
    Ce = CeN * ratio / (1 + CeN * ξₕ)

    # --- Derive turbulent scales from transfer coefficients ---
    u★ = sqrtCd * U
    θ★ = Ch / sqrtCd * Δθ
    q★ = Ce / sqrtCd * Δq

    return u★, θ★, q★
end
