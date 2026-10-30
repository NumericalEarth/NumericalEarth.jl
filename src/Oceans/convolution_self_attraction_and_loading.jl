using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Operators: Azᶜᶜᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ
using Downloads: Downloads
using Scratch: get_scratch!

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
fit is `1 - b₀`. The fit approaches tabulated load Love numbers only at high degree: `1 + k′ₙ - h′ₙ` is about
2.3–2.8 times the values of Gegout (2010) for n = 3–20, and twice them at n = 40.
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

const prem_load_love_numbers_url = "https://raw.githubusercontent.com/pyTMD/pyTMD/main/pyTMD/data/wang-prem-lln.txt"

"""
    prem_load_love_numbers(; maximum_degree = 5000)

Return the PREM load Love numbers of Wang et al. (2012) as `(; h, k)`, indexed by degree + 1, up to
`maximum_degree` (the table lists every degree to 5000), downloading the table distributed with pyTMD once.
Degree 1 is in the centre-of-Earth frame.
"""
function prem_load_love_numbers(; maximum_degree = 5000)
    path = joinpath(get_scratch!(@__MODULE__, "load_love_numbers"), "wang-prem-lln.txt")
    isfile(path) || Downloads.download(prem_load_love_numbers_url, path)
    rows = [parse.(Float64, split(line)) for line in eachline(path) if !startswith(strip(line), '#') && !isempty(strip(line))]
    rows = filter(row -> row[1] ≤ maximum_degree, rows)
    [Int(row[1]) for row in rows] == 0:maximum_degree || error("$path does not list every degree up to $maximum_degree")
    return (h = [row[2] for row in rows], k = [row[3] for row in rows])
end

"""
    loading_kernel_correction(love_numbers, asymptotic; points = 20000)

Tabulate `C(u) = Σₙ₌₁ᴺ [(1 + k′ₙ - h′ₙ) - (1 + k′ₙ - h′ₙ)ᵃˢʸᵐᵖᵗᵒᵗⁱᶜ] dPₙ/dc` at `points + 1` equally spaced values
of `u = sin(γ / 2) = √((1 - c) / 2)` from 0 to 1, where `γ` is the angle between two points and `N` is the last
degree of `love_numbers = (; h, k)`. Added to the derivative of the closed-form Green's function of `asymptotic`,
it gives the derivative of the Green's function with the tabulated load Love numbers to degree `N`, and the
asymptotic ones beyond. The degree-0 term does not contribute to the gradient.
"""
function loading_kernel_correction(love_numbers, asymptotic; points = 20000)
    (; h, k) = love_numbers
    N = length(h) - 1
    d = [(1 + k[n+1] - h[n+1]) - one_plus_k_minus_h(asymptotic, n) for n in 1:N]
    table = zeros(points + 1)
    Threads.@threads for q in 0:points
        u = q / points
        c = 1 - 2u^2
        # Pₙ and Pₙ′ = Pₙ₋₂′ + (2n - 1) Pₙ₋₁
        P², P¹ = 1.0, c
        P′², P′¹ = 0.0, 1.0
        S = d[1] * P′¹
        for n in 2:N
            P = ((2n - 1) * c * P¹ - (n - 1) * P²) / n
            P′ = P′² + (2n - 1) * P¹
            S += d[n] * P′
            P², P¹ = P¹, P
            P′², P′¹ = P′¹, P′
        end
        table[q+1] = S
    end
    return table
end

@inline function interpolate_correction(table, u)
    M = length(table) - 1
    x = u * M
    q = min(unsafe_trunc(Int, x), M - 1)
    ξ = x - q
    return @inbounds (1 - ξ) * table[q+1] + ξ * table[q+2]
end

struct ConvolutionLoadingOperator{FT, V, I}
    x :: V                  # Cartesian components of wet cell centers on the unit sphere
    y :: V
    z :: V
    e¹ :: NTuple{3, V}      # unit vectors of the two output components at each wet cell
    e² :: NTuple{3, V}
    solid_angle :: V        # cell areas divided by the radius squared
    i :: I                  # indices of wet cells
    j :: I
    weighted_source :: V    # pᵇ times solid angle at wet cells, workspace
    correction :: V         # tabulated correction to dG/dc for tabulated load Love numbers (zeros for the fit)
    neighbours :: NTuple{4, I}   # compact indices of the wet neighbours at i + 1, i - 1, j + 1, j - 1 (0 if none)
    stencil :: NTuple{6, V}      # weights of ∂p/∂x (self, i + 1, i - 1) and ∂p/∂y (self, j + 1, j - 1)
    self_term :: NTuple{4, V}    # maps (∂p/∂x, ∂p/∂y) to the (e¹, e²) components of the self-cell contribution
    radius :: FT
    coefficient :: FT       # 3ρ₀ / 4πρₑ
    singular_coefficient :: FT    # 1 - b₀
    logarithmic_coefficient :: FT # a₁ - b₁
end

Base.summary(op::ConvolutionLoadingOperator{FT}) where FT =
    string("convolution with the closed-form load Green's function over ", length(op.x), " wet cells in ", FT)

"""
    ConvolutionLoadingOperator(grid; components = :geographic, love_numbers = nothing, self_term = true,
                               reference_density = 1035, earth_density = 5517, FT = eltype(grid))

Build the linear operator that maps the kinematic bottom pressure anomaly `pᵇ` (for a single-layer model,
`g η`) to the horizontal gradient of the self-attraction and loading potential `Ψ = - G * pᵇ` at cell centers,
by direct summation of the gradient of the closed-form Green's function over the wet cells of `grid`, as in
Chen, Wang, Arbic and Krasny (2026):

```math
∇Ψ(xᵢ) = - \\sum_{j ≠ i} ∇ₓG(xᵢ ⋅ xⱼ) \\, pᵇ(xⱼ) \\, Ωⱼ ,
```

where `Ωⱼ` is the solid angle of cell `j`. The ocean is forced by `-∇Ψ`. The singular `i = j` term of the sum is
replaced by its integral over the cell for a linear `pᵇ`, `-K (1 - b₀) R⁻¹ ∫ r̂ r̂ᵀ / r dA ⋅ ∇pᵇ`, with
`∫ r̂ r̂ᵀ / r dA = diag(2Δy asinh(Δx / Δy), 2Δx asinh(Δy / Δx))` for a `Δx × Δy` cell and `∇pᵇ` from centred
differences between wet cells. Without it the gradient would be too small by a fraction that grows with degree,
about 15% at degree 40 on a 0.36ᵒ grid. `self_term = false` drops it, as Chen et al. do.

Their MOM6 code (Zenodo 10.5281/zenodo.18445619) also divides the kernel by 3 when it is used alone; with
`love_numbers = AsymptoticLoadLoveNumbers(), self_term = false, earth_density = 3 * 5517` this operator
reproduces it.

The Green's function is the closed form of [`AsymptoticLoadLoveNumbers`](@ref), which is exact only beyond degree
about 2000 and makes self-attraction and loading twice too strong at degree 40. Unless `love_numbers` is an
`AsymptoticLoadLoveNumbers` (the paper's method), a tabulated correction (see [`loading_kernel_correction`](@ref))
makes the kernel exact for each degree of `love_numbers`: a `NamedTuple` `(; h, k)` indexed by degree + 1, or
`nothing` for the PREM numbers of Wang et al. (2012) to degree 5000 ([`prem_load_love_numbers`](@ref)).

With `components = :geographic` the two components are eastward and northward; with `components = :grid` they are
along the direction of increasing `i` and perpendicular to it, to the left. The sum costs `O(N²)` for `N` wet cells
and is computed in precision `FT`; the angle between two cells is computed from their chord, so `Float32` is
accurate also for neighbours. Apply it with [`loading_gradient!`](@ref), and its transpose with
[`loading_gradient_transpose!`](@ref).
"""
function ConvolutionLoadingOperator(grid; components = :geographic, love_numbers = nothing, self_term = true,
                                    reference_density = 1035, earth_density = 5517, FT = eltype(grid))
    arch = architecture(grid)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(cpu_grid)
    R = cpu_grid.radius

    function position(i, j)
        λ = deg2rad(λnode(i, j, 1, cpu_grid, Center(), Center(), Center()))
        φ = deg2rad(φnode(i, j, 1, cpu_grid, Center(), Center(), Center()))
        return (cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ))
    end

    wet = [(i, j) for j in 1:Ny for i in 1:Nx if !inactive_node(i, j, Nz, cpu_grid, Center(), Center(), Center())]
    X = [position(i, j) for (i, j) in wet]

    normalize(v) = v ./ sqrt(sum(abs2, v))
    tangent(x, v) = normalize(v .- sum(x .* v) .* x)
    cross(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

    if components == :geographic
        e¹ = [tangent(x, (-x[2], x[1], 0.0)) for x in X]           # eastward
    elseif components == :grid                                       # along increasing i (periodic in i)
        e¹ = [tangent(x, position(mod1(i + 1, Nx), j) .- position(mod1(i - 1, Nx), j)) for ((i, j), x) in zip(wet, X)]
    else
        throw(ArgumentError("components must be :geographic or :grid; got $components"))
    end
    e² = [cross(x, e) for (x, e) in zip(X, e¹)]                     # northward, or across the grid line

    device(a) = on_architecture(arch, a)
    component(vs, n) = device(FT[v[n] for v in vs])

    asymptotic = love_numbers isa AsymptoticLoadLoveNumbers ? love_numbers : AsymptoticLoadLoveNumbers()
    K = 3 * reference_density / (4π * earth_density)

    # Self-cell term: five-point gradient of pᵇ between wet cells along the grid, and the cell integral of the
    # singular part of the kernel, projected on e¹ and e²
    compact = Dict(ij => n for (n, ij) in enumerate(wet))
    neighbour(i, j) = 1 ≤ j ≤ Ny ? get(compact, (mod1(i, Nx), j), 0) : 0
    neighbours = ntuple(q -> Int32[neighbour(i + (1, -1, 0, 0)[q], j + (0, 0, 1, -1)[q]) for (i, j) in wet], 4)
    function one_dimensional_stencil(forward, backward, Δ⁺, Δ⁻)
        w⁺ = forward > 0 ? (backward > 0 ? 1 / 2Δ⁺ : 1 / Δ⁺) : 0.0
        w⁻ = backward > 0 ? (forward > 0 ? -1 / 2Δ⁻ : -1 / Δ⁻) : 0.0
        return -(w⁺ + w⁻), w⁺, w⁻
    end
    weights = [(one_dimensional_stencil(neighbours[1][n], neighbours[2][n], Δxᶠᶜᶜ(i + 1, j, 1, cpu_grid), Δxᶠᶜᶜ(i, j, 1, cpu_grid))...,
                one_dimensional_stencil(neighbours[3][n], neighbours[4][n], Δyᶜᶠᶜ(i, j + 1, 1, cpu_grid), Δyᶜᶠᶜ(i, j, 1, cpu_grid))...)
               for (n, (i, j)) in enumerate(wet)]
    t¹ = [tangent(x, position(mod1(i + 1, Nx), j) .- position(mod1(i - 1, Nx), j)) for ((i, j), x) in zip(wet, X)]
    t² = [cross(x, t) for (x, t) in zip(X, t¹)]
    function self_term_coefficients(n, (i, j))
        self_term || return (0.0, 0.0, 0.0, 0.0)
        a, b = Δxᶜᶜᶜ(i, j, 1, cpu_grid), Δyᶜᶜᶜ(i, j, 1, cpu_grid)
        cˣ = - K * (1 - asymptotic.b₀) * 2b * asinh(a / b) / R # ∇Ψ = -∇(G * pᵇ)
        cʸ = - K * (1 - asymptotic.b₀) * 2a * asinh(b / a) / R
        dot(u, v) = sum(u .* v)
        return (dot(e¹[n], t¹[n]) * cˣ, dot(e¹[n], t²[n]) * cʸ, dot(e²[n], t¹[n]) * cˣ, dot(e²[n], t²[n]) * cʸ)
    end
    self_terms = [self_term_coefficients(n, ij) for (n, ij) in enumerate(wet)]
    correction = if love_numbers isa AsymptoticLoadLoveNumbers
        zeros(2)
    else
        loading_kernel_correction(isnothing(love_numbers) ? prem_load_love_numbers() : love_numbers, asymptotic)
    end

    return ConvolutionLoadingOperator(component(X, 1), component(X, 2), component(X, 3),
                                      ntuple(n -> component(e¹, n), 3),
                                      ntuple(n -> component(e², n), 3),
                                      device(FT[Azᶜᶜᶜ(i, j, 1, cpu_grid) / R^2 for (i, j) in wet]),
                                      device(Int32[i for (i, j) in wet]),
                                      device(Int32[j for (i, j) in wet]),
                                      device(zeros(FT, length(wet))),
                                      device(FT.(correction)),
                                      map(device, neighbours),
                                      ntuple(q -> component(weights, q), 6),
                                      ntuple(q -> component(self_terms, q), 4),
                                      FT(R),
                                      FT(K),
                                      FT(1 - asymptotic.b₀),
                                      FT(asymptotic.a₁ - asymptotic.b₁))
end

# dG/dc for the closed-form Green's function, divided by 3ρ₀ / 4πρₑ, with s = 1 - c:
#   d/dc (2s)^(-1/2) = (2s)^(-3/2)
#   d/dc [- ln(√(s/2) + s/2)] = (1 + 2r) / (4u (1 + r)),  u = s/2, r = √u
@inline function green_function_derivative(s, singular_coefficient, logarithmic_coefficient)
    u = s / 2
    r = sqrt(u)
    return singular_coefficient / (2s * sqrt(2s)) + logarithmic_coefficient * (1 + 2r) / (4u * (1 + r))
end

# 1 - x ⋅ y = |x - y|² / 2, which keeps its precision for nearby points
@inline one_minus_cosine(x, y, z, xm, ym, zm) = ((x - xm)^2 + (y - ym)^2 + (z - zm)^2) / 2

@kernel function _gather_weighted_source!(q, pᵇ, solid_angle, I, J)
    n = @index(Global)
    @inbounds q[n] = pᵇ[I[n], J[n], 1] * solid_angle[n]
end

# Components of ∇Ψ along e¹ and e² at wet cell n. ∇ₓ(x ⋅ y) = (y - (x ⋅ y) x) / R, and e¹ ⋅ x = e² ⋅ x = 0.
@kernel function _loading_gradient!(Ψ¹, Ψ², X, Y, Z, e¹, e², q, I, J, N, R, K, α, β, correction)
    n = @index(Global)
    FT = eltype(X)

    @inbounds x, y, z = X[n], Y[n], Z[n]

    # Sums of dG/dc y pᵇ Ω
    Sx = zero(FT)
    Sy = zero(FT)
    Sz = zero(FT)

    for m in 1:N
        @inbounds xm, ym, zm, qm = X[m], Y[m], Z[m], q[m]
        s = max(one_minus_cosine(x, y, z, xm, ym, zm), eps(FT))
        dG = green_function_derivative(s, α, β) + interpolate_correction(correction, sqrt(s / 2))
        dG = ifelse(m == n, zero(FT), dG * qm)
        Sx += dG * xm
        Sy += dG * ym
        Sz += dG * zm
    end

    @inbounds begin
        g¹ = e¹[1][n] * Sx + e¹[2][n] * Sy + e¹[3][n] * Sz
        g² = e²[1][n] * Sx + e²[2][n] * Sy + e²[3][n] * Sz
        Ψ¹[I[n], J[n], 1] = - K * g¹ / R
        Ψ²[I[n], J[n], 1] = - K * g² / R
    end
end

# The self-cell term, from pᵇ = q / Ω at wet cells
@inline function compact_gradient(q, Ω, neighbours, stencil, n)
    p(m) = @inbounds ifelse(m > 0, q[max(m, 1)] / Ω[max(m, 1)], zero(eltype(q)))
    @inbounds begin
        ∂x = stencil[1][n] * p(n) + stencil[2][n] * p(neighbours[1][n]) + stencil[3][n] * p(neighbours[2][n])
        ∂y = stencil[4][n] * p(n) + stencil[5][n] * p(neighbours[3][n]) + stencil[6][n] * p(neighbours[4][n])
    end
    return ∂x, ∂y
end

@kernel function _add_self_term!(Ψ¹, Ψ², q, Ω, I, J, neighbours, stencil, self_term)
    n = @index(Global)
    ∂x, ∂y = compact_gradient(q, Ω, neighbours, stencil, n)
    @inbounds begin
        Ψ¹[I[n], J[n], 1] += self_term[1][n] * ∂x + self_term[2][n] * ∂y
        Ψ²[I[n], J[n], 1] += self_term[3][n] * ∂x + self_term[4][n] * ∂y
    end
end

"""
    loading_gradient!(∂¹Ψ, ∂²Ψ, operator::ConvolutionLoadingOperator, pᵇ)

Compute the two components of `∇Ψ` at cell centers (fields at `Center, Center`) from the kinematic bottom pressure
anomaly `pᵇ`. Land cells are left untouched. The map is linear in `pᵇ`.
"""
function loading_gradient!(∂¹Ψ, ∂²Ψ, op::ConvolutionLoadingOperator, pᵇ)
    arch = architecture(pᵇ)
    N = length(op.x)
    launch!(arch, pᵇ.grid, (N,), _gather_weighted_source!, op.weighted_source, pᵇ, op.solid_angle, op.i, op.j)
    launch!(arch, pᵇ.grid, (N,), _loading_gradient!, ∂¹Ψ, ∂²Ψ, op.x, op.y, op.z, op.e¹, op.e², op.weighted_source,
            op.i, op.j, N, op.radius, op.coefficient, op.singular_coefficient, op.logarithmic_coefficient, op.correction)
    launch!(arch, pᵇ.grid, (N,), _add_self_term!, ∂¹Ψ, ∂²Ψ, op.weighted_source, op.solid_angle, op.i, op.j,
            op.neighbours, op.stencil, op.self_term)
    return nothing
end

# Transpose: pᵇ*(m) = - K Ωₘ Σₙ dG/dc(cₙₘ) [(e¹ₙ ⋅ xₘ) ∂¹Ψ*(n) + (e²ₙ ⋅ xₘ) ∂²Ψ*(n)] / R
@kernel function _loading_gradient_transpose!(pᵇ, Ψ¹, Ψ², X, Y, Z, e¹, e², solid_angle, I, J, N, R, K, α, β, correction)
    m = @index(Global)
    FT = eltype(X)

    @inbounds xm, ym, zm = X[m], Y[m], Z[m]
    S = zero(FT)

    for n in 1:N
        @inbounds begin
            x, y, z = X[n], Y[n], Z[n]
            a¹ = Ψ¹[I[n], J[n], 1]
            a² = Ψ²[I[n], J[n], 1]
            e¹xm = e¹[1][n] * xm + e¹[2][n] * ym + e¹[3][n] * zm
            e²xm = e²[1][n] * xm + e²[2][n] * ym + e²[3][n] * zm
        end
        s = max(one_minus_cosine(x, y, z, xm, ym, zm), eps(FT))
        dG = green_function_derivative(s, α, β) + interpolate_correction(correction, sqrt(s / 2))
        dG = ifelse(m == n, zero(FT), dG)
        S += dG * (e¹xm * a¹ + e²xm * a²)
    end

    @inbounds pᵇ[I[m], J[m], 1] = - K * solid_angle[m] * S / R
end

# Transpose of the self-cell term: each wet cell m collects from itself and from the neighbours whose stencils use it
@kernel function _add_self_term_transpose!(pᵇ, Ψ¹, Ψ², I, J, neighbours, stencil, self_term)
    m = @index(Global)
    FT = eltype(stencil[1])
    S = zero(FT)
    for slot in 0:4
        n = slot == 0 ? Int32(m) : @inbounds neighbours[slot][m]
        if n > 0
            @inbounds begin
                a¹ = Ψ¹[I[n], J[n], 1]
                a² = Ψ²[I[n], J[n], 1]
                gx = self_term[1][n] * a¹ + self_term[3][n] * a² # coefficient of ∂x pᵇ at n
                gy = self_term[2][n] * a¹ + self_term[4][n] * a²
                wx = ifelse(n == m, stencil[1][n], ifelse(neighbours[1][n] == m, stencil[2][n], ifelse(neighbours[2][n] == m, stencil[3][n], zero(FT))))
                wy = ifelse(n == m, stencil[4][n], ifelse(neighbours[3][n] == m, stencil[5][n], ifelse(neighbours[4][n] == m, stencil[6][n], zero(FT))))
            end
            S += gx * wx + gy * wy
        end
    end
    @inbounds pᵇ[I[m], J[m], 1] += S
end

"""
    loading_gradient_transpose!(pᵇ, operator::ConvolutionLoadingOperator, ∂¹Ψ, ∂²Ψ)

Apply the transpose (adjoint, in the Euclidean inner product over wet cells) of [`loading_gradient!`](@ref):
`Σ (∂¹Ψ ∂¹Ψ* + ∂²Ψ ∂²Ψ*) = Σ pᵇ pᵇ*`, sums over wet cells. Land cells of `pᵇ` are left untouched.
"""
function loading_gradient_transpose!(pᵇ, op::ConvolutionLoadingOperator, ∂¹Ψ, ∂²Ψ)
    N = length(op.x)
    launch!(architecture(pᵇ), pᵇ.grid, (N,), _loading_gradient_transpose!, pᵇ, ∂¹Ψ, ∂²Ψ, op.x, op.y, op.z, op.e¹, op.e²,
            op.solid_angle, op.i, op.j, N, op.radius, op.coefficient, op.singular_coefficient, op.logarithmic_coefficient,
            op.correction)
    launch!(architecture(pᵇ), pᵇ.grid, (N,), _add_self_term_transpose!, pᵇ, ∂¹Ψ, ∂²Ψ, op.i, op.j, op.neighbours,
            op.stencil, op.self_term)
    return nothing
end

#####
##### In a model: the gradient at cell centres, averaged to faces
#####

"""
    ConvolutionSelfAttractionAndLoading(grid; kwargs...)

Self-attraction and loading computed as in Chen, Wang, Arbic and Krasny (2026): each time the model state is
updated, the gradient of the self-attraction and loading potential is computed at cell centers by convolving the
bottom pressure anomaly (as for [`SphericalHarmonicSelfAttractionAndLoading`](@ref)) with the gradient of the
closed-form Green's function (see [`ConvolutionLoadingOperator`](@ref), to which `kwargs` are passed), and then
averaged to the velocity points, where it forces `u` and `v`. The direct sum costs `O(N²)` per update for `N` wet
columns; `FT = Float32` is several times faster on most GPUs.
"""
struct ConvolutionSelfAttractionAndLoading{G, P, O} <: AbstractSelfAttractionAndLoading
    gradient :: G # components of ∇Ψ along the grid directions at cell centers
    bottom_pressure_anomaly :: P
    reference_bottom_pressure :: P
    reference_set :: Base.RefValue{Bool}
    operator :: O
end

function ConvolutionSelfAttractionAndLoading(grid; kwargs...)
    operator = ConvolutionLoadingOperator(grid; components = :grid, kwargs...)
    gradient = (Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid))
    pᵇ = Field{Center, Center, Nothing}(grid)
    reference = Field{Center, Center, Nothing}(grid)
    return ConvolutionSelfAttractionAndLoading(gradient, pᵇ, reference, Ref(false), operator)
end

Base.summary(sal::ConvolutionSelfAttractionAndLoading) = string("ConvolutionSelfAttractionAndLoading with ", summary(sal.operator))
Base.show(io::IO, sal::ConvolutionSelfAttractionAndLoading) = print(io, summary(sal))

function compute_self_attraction_and_loading!(model, sal::ConvolutionSelfAttractionAndLoading)
    compute_bottom_pressure_anomaly!(sal, model)
    loading_gradient!(sal.gradient..., sal.operator, sal.bottom_pressure_anomaly)
    fill_halo_regions!(sal.gradient) # only the periodic halos in i are used
    return nothing
end

struct LoadingGradientForcing{D, F}
    direction :: D
    gradient :: F
end

Adapt.@adapt_structure LoadingGradientForcing

@inline (f::LoadingGradientForcing{<:XDirection})(i, j, k, grid, clock, fields) = @inbounds - (f.gradient[i-1, j, 1] + f.gradient[i, j, 1]) / 2
@inline (f::LoadingGradientForcing{<:YDirection})(i, j, k, grid, clock, fields) = @inbounds - (f.gradient[i, j-1, 1] + f.gradient[i, j, 1]) / 2

function add_self_attraction_and_loading_forcing(forcing, sal::ConvolutionSelfAttractionAndLoading)
    Fu = LoadingGradientForcing(XDirection(), sal.gradient[1])
    Fv = LoadingGradientForcing(YDirection(), sal.gradient[2])
    u_forcing = :u ∈ keys(forcing) ? (forcing.u, Fu) : Fu
    v_forcing = :v ∈ keys(forcing) ? (forcing.v, Fv) : Fv
    return merge(forcing, (u = u_forcing, v = v_forcing))
end

add_self_attraction_and_loading_forcing(forcing, sal) = forcing

function add_self_attraction_and_loading_callback!(simulation, sal::ConvolutionSelfAttractionAndLoading)
    simulation.callbacks[:self_attraction_and_loading] =
        Callback(compute_self_attraction_and_loading!; parameters = sal, callsite = UpdateStateCallsite())
    return nothing
end
