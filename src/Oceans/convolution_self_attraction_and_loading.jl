using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Operators: Azᶜᶜᶜ

#####
##### Self-attraction and loading by convolution with a closed-form Green's function
#####

"""
    AsymptoticLoadLoveNumbers(; a₁ = -2.7, b₀ = -6.21196, b₁ = 6.1)

The asymptotic fit `k′ₙ ≈ a₁ / n`, `h′ₙ ≈ b₀ + b₁ / n` to the PREM load Love numbers of Wang et al. (2012)
used by Chen, Wang, Arbic and Krasny (2026, arXiv:2602.01416). With it, the Green's function of
self-attraction and loading, `G(x ⋅ y) = Σₙ 3ρ₀ (1 + k′ₙ - h′ₙ) Pₙ(x ⋅ y) / 4πρₑ`, has the closed form

```math
G(c) = \\frac{3ρ₀}{4πρₑ} \\left[\\frac{1 - b₀}{\\sqrt{2 - 2c}} - (a₁ - b₁) \\log\\left(\\sqrt{\\frac{1 - c}{2}} + \\frac{1 - c}{2}\\right)\\right],
```

where `c = x ⋅ y` is the cosine of the angle between two points on the unit sphere. The degree-0 term of the
fit is `1 - b₀`.
"""
Base.@kwdef struct AsymptoticLoadLoveNumbers{FT}
    a₁ :: FT = -2.7
    b₀ :: FT = -6.21196
    b₁ :: FT = 6.1
end

"""
    one_plus_k_minus_h(love_numbers::AsymptoticLoadLoveNumbers, n)

Return `1 + k′ₙ - h′ₙ` of the asymptotic fit (`1 - b₀` at `n = 0`).
"""
one_plus_k_minus_h(L::AsymptoticLoadLoveNumbers, n) = n == 0 ? 1 - L.b₀ : 1 + (L.a₁ - L.b₁) / n - L.b₀

struct ConvolutionLoadingOperator{FT, V, I}
    x :: V                  # Cartesian components of wet cell centers on the unit sphere
    y :: V
    z :: V
    solid_angle :: V        # cell areas divided by the radius squared
    i :: I                  # indices of wet cells
    j :: I
    weighted_source :: V    # pᵇ times solid angle at wet cells, workspace
    target_values :: V      # workspace for the transpose
    radius :: FT
    coefficient :: FT       # 3ρ₀ / 4πρₑ
    singular_coefficient :: FT    # 1 - b₀
    logarithmic_coefficient :: FT # a₁ - b₁
end

Base.summary(op::ConvolutionLoadingOperator) =
    string("convolution with the closed-form load Green's function over ", length(op.x), " wet cells")

"""
    ConvolutionLoadingOperator(grid; love_numbers = AsymptoticLoadLoveNumbers(),
                               reference_density = 1035, earth_density = 5517)

Build the linear operator that maps the kinematic bottom pressure anomaly `pᵇ` (for a single-layer model,
`g η`) to the horizontal gradient of the self-attraction and loading potential `Ψ = - G * pᵇ` at cell centers,
by direct summation of the gradient of the closed-form Green's function over the wet cells of `grid`, as in
Chen, Wang, Arbic and Krasny (2026):

```math
∇Ψ(xᵢ) = - \\sum_{j ≠ i} ∇ₓG(xᵢ ⋅ xⱼ) \\, pᵇ(xⱼ) \\, Ωⱼ ,
```

where `Ωⱼ` is the solid angle of cell `j` and the singular self term is dropped. The ocean is forced by `-∇Ψ`.
The sum costs `O(N²)` for `N` wet cells. Apply it with [`loading_gradient!`](@ref), and its transpose with
[`loading_gradient_transpose!`](@ref).
"""
function ConvolutionLoadingOperator(grid; love_numbers = AsymptoticLoadLoveNumbers(),
                                    reference_density = 1035, earth_density = 5517)
    arch = architecture(grid)
    FT = eltype(grid)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(cpu_grid)
    R = cpu_grid.radius

    wet = [(i, j) for j in 1:Ny for i in 1:Nx if !inactive_node(i, j, Nz, cpu_grid, Center(), Center(), Center())]
    λ = [deg2rad(λnode(i, j, 1, cpu_grid, Center(), Center(), Center())) for (i, j) in wet]
    φ = [deg2rad(φnode(i, j, 1, cpu_grid, Center(), Center(), Center())) for (i, j) in wet]
    Ω = [Azᶜᶜᶜ(i, j, 1, cpu_grid) / R^2 for (i, j) in wet]

    device(a) = on_architecture(arch, a)
    N = length(wet)

    return ConvolutionLoadingOperator(device(FT.(cos.(φ) .* cos.(λ))),
                                      device(FT.(cos.(φ) .* sin.(λ))),
                                      device(FT.(sin.(φ))),
                                      device(FT.(Ω)),
                                      device(Int32[i for (i, j) in wet]),
                                      device(Int32[j for (i, j) in wet]),
                                      device(zeros(FT, N)),
                                      device(zeros(FT, N)),
                                      FT(R),
                                      FT(3 * reference_density / (4π * earth_density)),
                                      FT(1 - love_numbers.b₀),
                                      FT(love_numbers.a₁ - love_numbers.b₁))
end

# dG/dc for the closed-form Green's function, divided by 3ρ₀ / 4πρₑ, with s = 1 - c:
#   d/dc (2s)^(-1/2) = (2s)^(-3/2)
#   d/dc [- ln(√(s/2) + s/2)] = (1 + 2r) / (4u (1 + r)),  u = s/2, r = √u
@inline function green_function_derivative(s, singular_coefficient, logarithmic_coefficient)
    u = s / 2
    r = sqrt(u)
    return singular_coefficient / (2s * sqrt(2s)) + logarithmic_coefficient * (1 + 2r) / (4u * (1 + r))
end

@kernel function _gather_weighted_source!(q, pᵇ, solid_angle, I, J)
    n = @index(Global)
    @inbounds q[n] = pᵇ[I[n], J[n], 1] * solid_angle[n]
end

# ∂Ψ/∂x and ∂Ψ/∂y at the center of wet cell n, along the local eastward and northward directions eλ and eφ.
# ∇ₓ(x ⋅ y) = (y - (x ⋅ y) x) / R, and eλ ⋅ x = eφ ⋅ x = 0.
@kernel function _loading_gradient!(∂xΨ, ∂yΨ, X, Y, Z, q, I, J, N, R, K, α, β)
    n = @index(Global)
    FT = eltype(X)

    @inbounds begin
        x, y, z = X[n], Y[n], Z[n]
        ρ = sqrt(x^2 + y^2)
        cosλ, sinλ = ifelse(ρ > 0, x / ρ, one(FT)), ifelse(ρ > 0, y / ρ, zero(FT))
    end

    # Unnormalized sums of dG/dc (y - c x) pᵇ Ω
    Sx = zero(FT)
    Sy = zero(FT)
    Sz = zero(FT)
    Sc = zero(FT)

    for m in 1:N
        @inbounds begin
            xm, ym, zm, qm = X[m], Y[m], Z[m], q[m]
        end
        c = x * xm + y * ym + z * zm
        s = max(1 - c, eps(FT))
        dG = ifelse(m == n, zero(FT), green_function_derivative(s, α, β) * qm)
        Sx += dG * xm
        Sy += dG * ym
        Sz += dG * zm
    end

    gλ = - sinλ * Sx + cosλ * Sy
    gφ = - z * (cosλ * Sx + sinλ * Sy) + ρ * Sz

    @inbounds ∂xΨ[I[n], J[n], 1] = - K * gλ / R
    @inbounds ∂yΨ[I[n], J[n], 1] = - K * gφ / R
end

"""
    loading_gradient!(∂xΨ, ∂yΨ, operator::ConvolutionLoadingOperator, pᵇ)

Compute the eastward and northward components of `∇Ψ` at cell centers (fields at `Center, Center`) from the
kinematic bottom pressure anomaly `pᵇ`. Land cells are left untouched. The map is linear in `pᵇ`.
"""
function loading_gradient!(∂xΨ, ∂yΨ, op::ConvolutionLoadingOperator, pᵇ)
    arch = architecture(pᵇ)
    N = length(op.x)
    launch!(arch, pᵇ.grid, (N,), _gather_weighted_source!, op.weighted_source, pᵇ, op.solid_angle, op.i, op.j)
    launch!(arch, pᵇ.grid, (N,), _loading_gradient!, ∂xΨ, ∂yΨ, op.x, op.y, op.z, op.weighted_source, op.i, op.j,
            N, op.radius, op.coefficient, op.singular_coefficient, op.logarithmic_coefficient)
    return nothing
end

# Transpose: pᵇ*(m) = - K Ωₘ Σₙ dG/dc(cₙₘ) [(eλₙ ⋅ xₘ) ∂xΨ*(n) + (eφₙ ⋅ xₘ) ∂yΨ*(n)] / R
@kernel function _loading_gradient_transpose!(pᵇ, ∂xΨ, ∂yΨ, X, Y, Z, solid_angle, I, J, N, R, K, α, β)
    m = @index(Global)
    FT = eltype(X)

    @inbounds xm, ym, zm = X[m], Y[m], Z[m]
    S = zero(FT)

    for n in 1:N
        @inbounds begin
            x, y, z = X[n], Y[n], Z[n]
            aλ = ∂xΨ[I[n], J[n], 1]
            aφ = ∂yΨ[I[n], J[n], 1]
        end
        ρ = sqrt(x^2 + y^2)
        cosλ, sinλ = ifelse(ρ > 0, x / ρ, one(FT)), ifelse(ρ > 0, y / ρ, zero(FT))
        c = x * xm + y * ym + z * zm
        s = max(1 - c, eps(FT))
        dG = ifelse(m == n, zero(FT), green_function_derivative(s, α, β))
        eλ_xm = - sinλ * xm + cosλ * ym
        eφ_xm = - z * (cosλ * xm + sinλ * ym) + ρ * zm
        S += dG * (eλ_xm * aλ + eφ_xm * aφ)
    end

    @inbounds pᵇ[I[m], J[m], 1] = - K * solid_angle[m] * S / R
end

"""
    loading_gradient_transpose!(pᵇ, operator::ConvolutionLoadingOperator, ∂xΨ, ∂yΨ)

Apply the transpose (adjoint, in the Euclidean inner product over wet cells) of [`loading_gradient!`](@ref):
`Σ (∂xΨ ∂xΨ* + ∂yΨ ∂yΨ*) = Σ pᵇ pᵇ*`, sums over wet cells. Land cells of `pᵇ` are left untouched.
"""
function loading_gradient_transpose!(pᵇ, op::ConvolutionLoadingOperator, ∂xΨ, ∂yΨ)
    N = length(op.x)
    launch!(architecture(pᵇ), pᵇ.grid, (N,), _loading_gradient_transpose!, pᵇ, ∂xΨ, ∂yΨ, op.x, op.y, op.z,
            op.solid_angle, op.i, op.j, N, op.radius, op.coefficient, op.singular_coefficient, op.logarithmic_coefficient)
    return nothing
end
