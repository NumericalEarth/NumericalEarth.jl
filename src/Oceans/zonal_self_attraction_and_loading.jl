using Oceananigans.Solvers: FFTW

#####
##### Self-attraction and loading as a zonally invariant operator on a latitude-longitude grid
#####

"""
    ZonalLoadingOperator(Kr, Ki, mask)

A self-attraction and loading operator on a grid that is uniform in longitude:

```math
η_{SAL} = \\mathcal{F}^{-1}\\left[(K_r + i K_i) ⋅ \\mathcal{F}(m η)\\right],
```

where ``\\mathcal{F}`` is the discrete Fourier transform along each latitude row, ``m`` is the ocean mask, and
the product is a batched matrix product over the zonal wavenumber (`Kr[j, j′, m+1]`). Every operator that commutes
with rotations about the polar axis by a whole number of cells has this form, including spherical-harmonic SAL with
any Love numbers and truncation and convolution with any Green's function ``G(x ⋅ y)``. Applying it costs two FFTs and
`O(Nm Ny²)` multiply-adds, and it is differentiable by Enzyme under Reactant (FFTs and one `dot_general`).
Build kernels with [`spherical_harmonic_zonal_kernel`](@ref), [`convolution_zonal_kernel`](@ref) or
[`hybrid_zonal_kernel`](@ref), and apply with [`zonal_self_attraction_and_loading`](@ref).
"""
struct ZonalLoadingOperator{KR, KI, M, P}
    Kr :: KR
    Ki :: KI
    mask :: M
    pad :: P
    Nx :: Int
end

function ZonalLoadingOperator(Kr, Ki, mask)
    Ny, _, Nm = size(Kr)
    Nx = size(mask, 1)
    Nk = Nx ÷ 2 + 1
    pad = Nm < Nk ? zeros(Complex{Float64}, Nk - Nm, Ny) : nothing
    return ZonalLoadingOperator(Kr, Ki, Float64.(mask), pad, Nx)
end

Base.summary(op::ZonalLoadingOperator) = string("ZonalLoadingOperator with ", size(op.Kr, 3), " zonal wavenumbers on ",
                                                op.Nx, " × ", size(op.Kr, 1), " cells")

# Y[m, j, c] = Σⱼ′ K[j, j′, m] X[j′, c, m]
function batched_contract(K::Array, X::Array)
    Ny, _, Nm = size(K)
    Y = zeros(promote_type(eltype(K), eltype(X)), Nm, Ny, size(X, 2))
    Threads.@threads for m in 1:Nm
        Y[m, :, :] .= K[:, :, m] * X[:, :, m]
    end
    return Y
end

"""
    zonal_self_attraction_and_loading(op::ZonalLoadingOperator, η)

Return the SAL elevation (`Nx × Ny`) of the elevation (or load) `η` (`Nx × Ny`).
"""
function zonal_self_attraction_and_loading(op::ZonalLoadingOperator, η)
    Nm = size(op.Kr, 3)
    q = η .* op.mask
    Ny = size(q, 2)
    q̂ = FFTW.rfft(q, 1)[1:Nm, :]
    FT = eltype(op.Kr)
    X = FT.(permutedims(cat(reshape(real.(q̂), Nm, Ny, 1), reshape(imag.(q̂), Nm, Ny, 1); dims = 3), (2, 3, 1)))
    Y = batched_contract(op.Kr, X)
    if isnothing(op.Ki)
        Yr, Yi = Y[:, :, 1], Y[:, :, 2]
    else
        Z = batched_contract(op.Ki, X)
        Yr = Y[:, :, 1] .- Z[:, :, 2]
        Yi = Y[:, :, 2] .+ Z[:, :, 1]
    end
    ψ̂ = complex.(Float64.(Yr), Float64.(Yi))
    isnothing(op.pad) || (ψ̂ = vcat(ψ̂, op.pad))
    return FFTW.irfft(ψ̂, op.Nx, 1)
end

#####
##### Kernels. Rows: latitude faces φᶠ (degrees, length Ny + 1); columns: Nx cells starting at the prime meridian.
#####

sal_weight(ρ₀, ρₑ, k, h, n) = 3ρ₀ / ρₑ * (1 + k - h) / (2n + 1)

# Gauss-Legendre nodes and weights on [-1, 1] by Newton iteration on Pₙ
function gauss_legendre(n)
    x = [cos(π * (i - 1/4) / (n + 1/2)) for i in 1:n]
    w = zeros(n)
    for i in 1:n, _ in 1:100
        P₁, P₀ = x[i], 1.0
        for k in 2:n
            P₁, P₀ = ((2k - 1) * x[i] * P₁ - (k - 1) * P₀) / k, P₁
        end
        dP = n * (x[i] * P₁ - P₀) / (x[i]^2 - 1)
        x[i] -= P₁ / dP
        w[i] = 2 / ((1 - x[i]^2) * dP^2)
    end
    return x, w
end

# Orthonormal associated Legendre functions, ∫ P̄ₗₘ² dμ = 1/2π; P[ℓ+1, m+1]
function orthonormal_legendre!(P, L, M, x)
    fill!(P, 0)
    y = sqrt(max(1 - x^2, 0.0))
    Pmm = 1 / sqrt(4π)
    for m in 0:M
        m > 0 && (Pmm *= sqrt((2m + 1) / 2m) * y)
        P[m+1, m+1] = Pmm
        Pl2, Pl1 = 0.0, Pmm
        for ℓ in m+1:L
            Pl = sqrt((4ℓ^2 - 1) / (ℓ^2 - m^2)) * (x * Pl1 - sqrt(((ℓ - 1)^2 - m^2) / (4(ℓ - 1)^2 - 1)) * Pl2)
            P[ℓ+1, m+1] = Pl
            Pl2, Pl1 = Pl1, Pl
        end
    end
    return P
end

"""
    spherical_harmonic_zonal_kernel(Nx, φᶠ, γ; L = length(γ) - 1)

Kernel of SAL truncated at degree `L` with weights `γ[n+1] = 3ρ₀(1 + k′ₙ − h′ₙ) / ρₑ(2n + 1)`, applied to the
spherical-harmonic projection of the piecewise-constant field on the grid (cell-integrated Legendre functions and a
sinc factor in longitude), evaluated at cell centers.
"""
function spherical_harmonic_zonal_kernel(Nx, φᶠ, γ; L = length(γ) - 1, quadrature = 8)
    μᶠ = sind.(collect(Float64, φᶠ))
    Ny = length(μᶠ) - 1
    M = min(L, Nx ÷ 2)
    Δλ = 2π / Nx
    x, w = gauss_legendre(quadrature)
    Pc = zeros(L + 1, M + 1, Ny)
    Wb = zeros(L + 1, M + 1, Ny)
    Threads.@threads for j in 1:Ny
        P = zeros(L + 1, M + 1)
        orthonormal_legendre!(P, L, M, sind((φᶠ[j] + φᶠ[j+1]) / 2))
        Pc[:, :, j] .= P
        for q in eachindex(x)
            orthonormal_legendre!(P, L, M, (μᶠ[j] + μᶠ[j+1]) / 2 + (μᶠ[j+1] - μᶠ[j]) / 2 * x[q])
            Wb[:, :, j] .+= (μᶠ[j+1] - μᶠ[j]) / 2 * w[q] .* P
        end
    end
    K = zeros(Ny, Ny, M + 1)
    Threads.@threads for m in 0:M
        s = m == 0 ? 1.0 : sin(m * Δλ / 2) / (m * Δλ / 2)
        K[:, :, m+1] .= 2π * s .* (Pc[m+1:L+1, m+1, :]' * (γ[m+1:L+1] .* Wb[m+1:L+1, m+1, :]))
    end
    return K
end

# Green's function of Chen et al. (2026, eq. 16), with η_SAL(x) = ∫ G(x ⋅ y) η(y) dΩ(y)
@inline function closed_form_green_function(c, K, L::AsymptoticLoadLoveNumbers)
    s = max(1 - c, 1e-300)
    return K * ((1 - L.b₀) / sqrt(2s) - (L.a₁ - L.b₁) * log(sqrt(s / 2) + s / 2))
end

"""
    green_function_correction(love_numbers, asymptotic = AsymptoticLoadLoveNumbers();
                              reference_density = 1035, earth_density = 5517, points = 20000)

Tabulate `ΔG(u) = 3ρ₀ / 4πρₑ Σₙ [(1 + k′ₙ − h′ₙ) − (1 + k′ₙ − h′ₙ)ᵃˢʸᵐᵖᵗᵒᵗⁱᶜ] Pₙ(1 − 2u²)` at `points + 1` equally
spaced `u = sin(γ / 2)` in [0, 1], over the degrees of `love_numbers = (; h, k)` (indexed by degree + 1). Added to the
closed-form Green's function it gives the Green's function with the tabulated load Love numbers.
"""
function green_function_correction(love_numbers, asymptotic = AsymptoticLoadLoveNumbers();
                                   reference_density = 1035, earth_density = 5517, points = 20000)
    (; h, k) = love_numbers
    N = length(h) - 1
    d = [(1 + k[n+1] - h[n+1]) - one_plus_k_minus_h(asymptotic, n) for n in 0:N]
    K = 3reference_density / (4π * earth_density)
    table = zeros(points + 1)
    Threads.@threads for q in 0:points
        c = 1 - 2 * (q / points)^2
        P², P¹ = 1.0, c
        S = d[1] + d[2] * c
        for n in 2:N
            P = ((2n - 1) * c * P¹ - (n - 1) * P²) / n
            S += d[n+1] * P
            P², P¹ = P¹, P
        end
        table[q+1] = K * S
    end
    return table
end

loading_green_function(c, K, L, ::Nothing) = closed_form_green_function(c, K, L)
loading_green_function(c, K, L, correction) = closed_form_green_function(c, K, L) + interpolate_correction(correction, sqrt(max(1 - c, 0) / 2))

@inline unit_sphere_point(λ, φ) = (cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ))
@inline dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ∫ G over source cell (js, offset d) for the target at the center of cell (0, jt): midpoint rule far away,
# subcell rule within `near` cell sizes, and the 1/r singularity of the self cell integrated analytically.
function cell_integrated_green_function(Nx, φᶠ, jt, js, d, K, L, correction; subcells = 8, near = 3)
    Δλ = 2π / Nx
    φ₋, φ₊ = deg2rad(φᶠ[js]), deg2rad(φᶠ[js+1])
    φt = deg2rad((φᶠ[jt] + φᶠ[jt+1]) / 2)
    φs = (φ₋ + φ₊) / 2
    xt = unit_sphere_point(0.0, φt)
    c = dot3(xt, unit_sphere_point(d * Δλ, φs))
    Δφ = φ₊ - φ₋
    h = max(Δφ, cos(φs) * Δλ, cos(φt) * Δλ)
    sqrt(max(2 - 2c, 0.0)) > near * h && return loading_green_function(c, K, L, correction) * Δλ * (sin(φ₊) - sin(φ₋))
    self = d == 0 && js == jt
    A = K * (1 - L.b₀)
    n = subcells
    S = 0.0
    for p in 1:n, q in 1:n
        λ = (d + (p - 0.5) / n - 0.5) * Δλ
        φa, φb = φ₋ + (q - 1) / n * Δφ, φ₋ + q / n * Δφ
        φ = (φa + φb) / 2
        Gv = loading_green_function(dot3(xt, unit_sphere_point(λ, φ)), K, L, correction)
        self && (Gv -= A / sqrt((cos(φt) * λ)^2 + (φ - φt)^2))
        S += Gv * Δλ / n * (sin(φb) - sin(φa))
    end
    if self
        a, b = cos(φt) * Δλ, Δφ
        S += A * (2a * asinh(b / a) + 2b * asinh(a / b))
    end
    return S
end

"""
    convolution_zonal_kernel(Nx, φᶠ; love_numbers = AsymptoticLoadLoveNumbers(), correction = nothing,
                             reference_density = 1035, earth_density = 5517)

Kernel of the convolution of η with the closed-form Green's function of Chen et al. (2026), plus `correction`
from [`green_function_correction`](@ref) if given, integrated over each source cell, giving the potential at cell
centers. Evaluates the full sum over all cells.
"""
function convolution_zonal_kernel(Nx, φᶠ; love_numbers = AsymptoticLoadLoveNumbers(), correction = nothing,
                                  reference_density = 1035, earth_density = 5517, subcells = 8)
    Ny = length(φᶠ) - 1
    Nm = Nx ÷ 2 + 1
    K = 3reference_density / (4π * earth_density)
    kernel = zeros(Ny, Ny, Nm)
    plan = FFTW.plan_rfft(zeros(Nx))
    Threads.@threads for jt in 1:Ny
        k = zeros(Nx)
        for js in 1:Ny
            for d in 0:Nx÷2
                k[d+1] = cell_integrated_green_function(Nx, φᶠ, jt, js, d, K, love_numbers, correction; subcells)
            end
            for d in Nx÷2+1:Nx-1
                k[d+1] = k[Nx-d+1]
            end
            @views kernel[jt, js, :] .= real.((plan * k)[1:Nm])
        end
    end
    return kernel
end

"""
    hybrid_zonal_kernel(Nx, φᶠ, γ; N₀ = length(γ) - 1, love_numbers = AsymptoticLoadLoveNumbers(), kwargs...)

The convolution kernel plus the spherical-harmonic correction `Σ_{n ≤ N₀} (γₙ − γₙ^asymptotic) ηₙ`, so that degrees
up to `N₀` use the exact weights `γ`. Same cost per application as the convolution.
"""
function hybrid_zonal_kernel(Nx, φᶠ, γ; N₀ = length(γ) - 1, love_numbers = AsymptoticLoadLoveNumbers(),
                             reference_density = 1035, earth_density = 5517,
                             convolution = convolution_zonal_kernel(Nx, φᶠ; love_numbers, reference_density, earth_density))
    γa = [3reference_density / earth_density * one_plus_k_minus_h(love_numbers, n) / (2n + 1) for n in 0:N₀]
    correction = spherical_harmonic_zonal_kernel(Nx, φᶠ, γ[1:N₀+1] .- γa; L = N₀)
    kernel = copy(convolution)
    kernel[:, :, 1:size(correction, 3)] .+= correction
    return kernel
end
