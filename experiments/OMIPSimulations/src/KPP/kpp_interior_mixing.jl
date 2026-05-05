# Interior viscosity / diffusivity at (Center, Center, Face) interfaces.
# Sum of shear-instability + convective-instability contributions.

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    return ∂z_u² + ∂z_v²
end

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    FT = eltype(grid)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return N² / max(S², FT(1e-10))
end

# Smooth shear-instability shape: 1 at Ri ≤ 0, 0 at Ri ≥ Ri∞, cubic in between.
@inline function shear_factor(Ri, Ri∞, FT)
    r = min(max(Ri, zero(FT)) / Ri∞, one(FT))
    f = one(FT) - r * r
    return f * f * f
end

# Smooth convective-instability shape: 1 at N² ≤ N²ᶜᵒⁿ, 0 at N² ≥ 0, cubic in between.
@inline function convective_factor(N², N²ᶜᵒⁿ, FT)
    Ng = max(N², N²ᶜᵒⁿ)
    r  = min((N²ᶜᵒⁿ - Ng) / N²ᶜᵒⁿ, one(FT))
    f  = one(FT) - r * r
    return f * f * f
end

@inline function interior_viscosityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    FT = eltype(grid)
    p  = getclosure(i, j, closure).parameters
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)

    Ri   = N² / max(S², FT(1e-10))
    fRi  = shear_factor(Ri, p.Ri∞, FT)
    fcon = convective_factor(N², p.N²ᶜᵒⁿ, FT)

    return fRi * p.ν₀ˢʰ + fcon * p.νᶜᵒⁿ
end

@inline function interior_diffusivityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    FT = eltype(grid)
    p  = getclosure(i, j, closure).parameters
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)

    Ri   = N² / max(S², FT(1e-10))
    fRi  = shear_factor(Ri, p.Ri∞, FT)
    fcon = convective_factor(N², p.N²ᶜᵒⁿ, FT)

    return fRi * p.κ₀ˢʰ + fcon * p.κᶜᵒⁿ
end
