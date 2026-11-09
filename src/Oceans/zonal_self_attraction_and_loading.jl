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

#####
##### Self-attraction and loading on a curvilinear grid through a coarser auxiliary latitude-longitude grid
#####

# Operator constants carry no derivative, even inside `Duplicated` arguments (Reactant method in the extension)
no_derivative(x) = x

convert_eltype(FT, X) = eltype(X) == FT ? X : FT.(X)
kernel_eltype(K) = eltype(K)

"""
    RemappedZonalLoadingOperator(λᶠᶠ, φᶠᶠ, λᶜᶜ, φᶜᶜ, area, mask; degrees, kernel, subsamples = 2, FT = Float64)

Self-attraction and loading on a curvilinear grid: a conservative remap onto an auxiliary latitude-longitude grid with
spacing `degrees`, a [`ZonalLoadingOperator`](@ref) there, and a bilinear interpolation back to the cell centers.
SAL is smooth, so the auxiliary grid can be much coarser than the model grid.

`λᶠᶠ, φᶠᶠ` are the cell corners in degrees (`(Nx + 1) × (Ny + 1)`), `λᶜᶜ, φᶜᶜ` the centers, `area` the cell areas in
m², and `mask` the ocean cells. `kernel(Nx, φᶠ)` returns the kernel on the auxiliary grid (for example
`(Nx, φᶠ) -> convolution_zonal_kernel(Nx, φᶠ; correction)`).

Both remaps are fixed-width gathers. The reverse of a gather is a scatter-add, which is very slow when one index is
repeated many times, so the padding of the forward table uses distinct indices (with zero weight).
"""
struct RemappedZonalLoadingOperator{Z, IF, WF, IB, WB}
    zonal :: Z
    forward_index :: IF
    forward_weight :: WF
    backward_index :: IB
    backward_weight :: WB
    auxiliary_size :: Tuple{Int, Int}
    model_size :: Tuple{Int, Int}
end

function RemappedZonalLoadingOperator(λᶠᶠ, φᶠᶠ, λᶜᶜ, φᶜᶜ, area, mask; degrees, kernel, subsamples = 2, FT = Float64,
                                      radius = 6.371e6)
    φ₀ = floor(minimum(φᶠᶠ) / degrees) * degrees
    Ny′ = round(Int, (90 - φ₀) / degrees)
    Nx′ = round(Int, 360 / degrees)
    φᶠ = collect(range(φ₀, 90, length = Ny′ + 1))
    μᶠ = sind.(φᶠ)
    Δλ = 360 / Nx′
    Nx, Ny = size(area)
    N′ = Nx′ * Ny′
    s = subsamples

    rows = [Dict{Int32, Float64}() for _ in 1:N′]
    corner(i, j) = collect(unit_sphere_point(deg2rad(λᶠᶠ[i, j]), deg2rad(φᶠᶠ[i, j])))
    for j in 1:Ny, i in 1:Nx
        mask[i, j] || continue
        t = Int32(i + (j - 1) * Nx)
        c₀₀, c₁₀, c₀₁, c₁₁ = corner(i, j), corner(i + 1, j), corner(i, j + 1), corner(i + 1, j + 1)
        for q in 1:s, p in 1:s
            ξ, ζ = (p - 1/2) / s, (q - 1/2) / s
            v = @. (1 - ξ) * (1 - ζ) * c₀₀ + ξ * (1 - ζ) * c₁₀ + (1 - ξ) * ζ * c₀₁ + ξ * ζ * c₁₁
            λ, φ = atand(v[2], v[1]), asind(clamp(v[3] / sqrt(sum(abs2, v)), -1, 1))
            i′ = mod(floor(Int, mod(λ, 360) / Δλ), Nx′) + 1
            j′ = clamp(searchsortedlast(φᶠ, φ), 1, Ny′)
            ℓ = i′ + (j′ - 1) * Nx′
            w = area[i, j] / s^2 / (radius^2 * deg2rad(Δλ) * (μᶠ[j′+1] - μᶠ[j′]))
            rows[ℓ][t] = get(rows[ℓ], t, 0.0) + w
        end
    end

    Kf = max(1, maximum(length, rows))
    forward_index = Matrix{Int32}(undef, N′, Kf)
    forward_weight = zeros(N′, Kf)
    for ℓ in eachindex(rows)
        entries = sort!(collect(rows[ℓ]); by = first)
        for k in 1:Kf
            if k ≤ length(entries)
                forward_index[ℓ, k], forward_weight[ℓ, k] = entries[k]
            else
                forward_index[ℓ, k] = Int32(mod(ℓ * Kf + k, Nx * Ny) + 1)
            end
        end
    end

    φᶜ = (φᶠ[1:end-1] .+ φᶠ[2:end]) ./ 2
    backward_index = Matrix{Int32}(undef, Nx * Ny, 4)
    backward_weight = zeros(Nx * Ny, 4)
    for t in 1:Nx*Ny
        x = mod(λᶜᶜ[t], 360) / Δλ - 1/2
        i₀ = floor(Int, x)
        ξ = x - i₀
        ia, ib = mod(i₀, Nx′) + 1, mod(i₀ + 1, Nx′) + 1
        φ = φᶜᶜ[t]
        ja = clamp(searchsortedlast(φᶜ, φ), 1, Ny′)
        jb = min(ja + 1, Ny′)
        ζ = jb == ja ? 0.0 : clamp((φ - φᶜ[ja]) / (φᶜ[jb] - φᶜ[ja]), 0, 1)
        cell(i, j) = Int32(i + (j - 1) * Nx′)
        backward_index[t, :] .= (cell(ia, ja), cell(ib, ja), cell(ia, jb), cell(ib, jb))
        backward_weight[t, :] .= ((1 - ξ) * (1 - ζ), ξ * (1 - ζ), (1 - ξ) * ζ, ξ * ζ)
    end

    zonal = ZonalLoadingOperator(FT.(kernel(Nx′, φᶠ)), nothing, ones(Nx′, Ny′))
    return RemappedZonalLoadingOperator(zonal, forward_index, forward_weight, backward_index, backward_weight,
                                        (Nx′, Ny′), (Nx, Ny))
end

Base.summary(op::RemappedZonalLoadingOperator) = string("RemappedZonalLoadingOperator from ", op.model_size[1], " × ", op.model_size[2],
                                                        " cells through a ", op.auxiliary_size[1], " × ", op.auxiliary_size[2], " grid")

"""
    zonal_self_attraction_and_loading(op::RemappedZonalLoadingOperator, η)

Return the SAL elevation at the cell centers of the model grid from the elevation `η` (`Nx × Ny`).
"""
function zonal_self_attraction_and_loading(op::RemappedZonalLoadingOperator, η)
    η′ = reshape(sum(no_derivative(op.forward_weight) .* reshape(η, :)[op.forward_index]; dims = 2), op.auxiliary_size)
    z = op.zonal
    Nm = size(z.Kr, 3)
    Ny′ = size(η′, 2)
    q̂ = FFTW.rfft(η′, 1)[1:Nm, :]
    FT = kernel_eltype(z.Kr)
    X = convert_eltype(FT, permutedims(cat(reshape(real.(q̂), Nm, Ny′, 1), reshape(imag.(q̂), Nm, Ny′, 1); dims = 3), (2, 3, 1)))
    Y = batched_contract(no_derivative(z.Kr), X)
    ψ̂ = complex.(convert_eltype(Float64, Y[:, :, 1]), convert_eltype(Float64, Y[:, :, 2]))
    isnothing(z.pad) || (ψ̂ = vcat(ψ̂, z.pad))
    ψ = reshape(FFTW.irfft(ψ̂, z.Nx, 1), :)
    return reshape(sum(no_derivative(op.backward_weight) .* ψ[op.backward_index]; dims = 2), op.model_size)
end
