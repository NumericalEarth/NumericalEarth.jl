# Boundary-layer diffusivity within hbl: shape function G(σ), matching to
# interior K and dK/dz at σ = 1, and nonlocal transport coefficient γ
# (tracers only, destabilizing forcing only).
@inline function shape_function(σ, G1, dG1)
    a₁ = σ - 2
    a₂ = 3 - 2σ
    a₃ = σ - 1
    return a₁ + a₂ * G1 + a₃ * dG1
end

# K(σ) = hbl · w(σ) · σ · (1 + σ · G(σ)). Vanishes at σ=0; matches interior K
# and dK/dσ at σ=1 by construction.
@inline boundary_layer_diffusivity(σ, hbl, w, G1, dG1) = hbl * w * σ * (one(σ) + σ * shape_function(σ, G1, dG1))

# G(1) = Kint / (hbl·w);   dG/dσ|₁ = -dKdz / w + Cˢᵗ·Bo·Kint/u★⁴.
# The stability term only contributes under stabilizing forcing (Bo > 0);
# dG1 is clamped to ≤ 0 so K does not grow upward at the BL base.
@inline function matching_coefficients(hbl, Kint, dKdz, w, Bo, u★, p)
    FT  = typeof(hbl)
    G1  = Kint / max(hbl * w, FT(1e-10))
    f₁  = ifelse(Bo ≥ zero(FT), p.Cˢᵗ * Bo / max(u★^4, FT(1e-10)), zero(FT))
    dG1 = - dKdz / max(w, FT(1e-10)) + f₁ * Kint
    return G1, min(dG1, zero(FT))
end

# Nonlocal transport coefficient for tracers under destabilizing forcing.
# Zero for momentum and under stabilizing forcing. The coefficient
#     cg = C★ · κᵥ · (Bˢ · κᵥ · ε)^(1/3)
# is derived from C★ (Large 1994's "cstar", typically 10) and reproduces
# MITgcm's `cg ≈ 3.45` (kpp_init_fixed.F:126).
@inline function nonlocal_transport(hbl, ws, Bo, p)
    FT = typeof(hbl)
    cg = p.C★ * p.κᵥ * cbrt(p.Bˢ * p.κᵥ * p.ε)
    return ifelse(Bo < zero(FT), cg / max(ws * hbl, FT(1e-10)), zero(FT))
end
