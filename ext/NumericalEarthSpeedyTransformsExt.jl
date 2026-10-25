module NumericalEarthSpeedyTransformsExt

using NumericalEarth
using NumericalEarth.Oceans: Oceans
using Oceananigans
using Oceananigans.Architectures: architecture, on_architecture, CPU
using Oceananigans.Fields: interior
using Oceananigans.Grids: λnode, φnode
using Oceananigans.ImmersedBoundaries: underlying_grid
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using ConservativeRegridding: ConservativeRegridding, Regridder, regrid!
using SpeedyTransforms: SpeedyTransforms, SpectralTransform, transform!
using SpeedyTransforms.RingGrids: RingGrids, FullGaussianGrid
using SpeedyTransforms.LowerTriangularArrays: eachharmonic
using Downloads: Downloads
using Scratch: get_scratch!

#####
##### Load Love numbers
#####

const load_love_numbers_url =
    "https://raw.githubusercontent.com/MITgcm-contrib/llc_hires/master/athena/llc_1080/input_sal/Load_Love2_CM.dat"

"""
    default_load_love_numbers()

Return the path to the load Love numbers of Gegout (2010) in the center-of-mass frame, to degree 1024,
used by the LLC4320v2 simulations (`Load_Love2_CM.dat` in MITgcm-contrib/llc_hires), downloading them once.
"""
function default_load_love_numbers()
    path = joinpath(get_scratch!(NumericalEarth, "load_love_numbers"), "Load_Love2_CM.dat")
    isfile(path) || Downloads.download(load_love_numbers_url, path)
    return path
end

"""
    read_load_love_numbers(path)

Read a table of load Love numbers with columns `degree h l k` (Fortran `D` exponents allowed, `#` comments)
and return `(; h, l, k)` vectors indexed by degree + 1.
"""
function read_load_love_numbers(path)
    rows = [parse.(Float64, split(replace(line, 'D' => 'e'))) for line in eachline(path)
            if !startswith(strip(line), '#') && !isempty(strip(line))]
    degrees = [Int(row[1]) for row in rows]
    degrees == 0:length(rows)-1 || error("$path does not list consecutive degrees from 0")
    return (h = [row[2] for row in rows], l = [row[3] for row in rows], k = [row[4] for row in rows])
end

load_love_numbers(love_numbers::NamedTuple) = love_numbers
load_love_numbers(path::AbstractString) = read_load_love_numbers(path)
load_love_numbers(::Nothing) = read_load_love_numbers(default_load_love_numbers())

#####
##### The operator
#####

struct SpectralLoadingOperator{R, V, F, S, C, W, D, A}
    regridder :: R           # conservative map from the ocean grid to the Gaussian grid
    gaussian_values :: V     # pᵇ on the Gaussian grid, on the ocean's architecture
    gaussian_field :: F      # pᵇ on the Gaussian grid, on the CPU
    transform :: S           # spectral transform on the CPU
    coefficients :: C        # spherical harmonic coefficients on the CPU
    weights :: W             # γₗ for each coefficient
    device_coefficients :: D # γₗ pᵇₗₘ on the ocean's architecture
    recursion :: A           # Legendre recursion coefficients on the ocean's architecture
    degree :: Int            # maximum degree
end

function Base.summary(op::SpectralLoadingOperator)
    grid = op.gaussian_field.grid
    return string("degree-", op.degree, " spherical harmonics on a ", RingGrids.get_nlat(grid), " × ",
                  RingGrids.get_nlon_max(grid), " Gaussian grid")
end

# ConservativeRegridding needs a grid with a regular vertical coordinate; only the horizontal cells matter.
horizontal_grid(grid) = horizontal_grid(underlying_grid(grid))
horizontal_grid(grid::Oceananigans.Grids.AbstractUnderlyingGrid) = on_architecture(CPU(), grid)

fold_topology(::Tripolar{N, F, S, TY}) where {N, F, S, TY} = TY
fold_pivot(::Tripolar{N, F, S, TY, P}) where {N, F, S, TY, P} = P

function horizontal_grid(grid::TripolarGrid)
    mapping = grid.conformal_mapping
    Nx, Ny, _ = size(grid)
    # ConservativeRegridding splits the fold row into quarters
    Nx % 4 == 0 || throw(ArgumentError("SphericalHarmonicSelfAttractionAndLoading needs a TripolarGrid with Nx divisible by 4; got Nx = $Nx"))
    return TripolarGrid(CPU(), eltype(grid);
                        size = (Nx, Ny, 1),
                        halo = (grid.Hx, grid.Hy, 1),
                        z = (-1, 0),
                        radius = grid.radius,
                        southernmost_latitude = mapping.southernmost_latitude,
                        north_poles_latitude = mapping.north_poles_latitude,
                        first_pole_longitude = mapping.first_pole_longitude,
                        fold_topology = fold_topology(mapping),
                        pivot = fold_pivot(mapping))
end

"""
    spherical_harmonic_loading_operator(grid; truncation = 40, nlat_half = 24, love_numbers = nothing,
                                        reference_density = 1035, earth_density = 5517)

Build the operator for [`SphericalHarmonicSelfAttractionAndLoading`](@ref) on `grid`: spherical harmonics to degree
`truncation` on a `FullGaussianGrid` with `2 nlat_half` latitudes (the defaults match LLC4320v2's degree 40 on 48
latitudes). `love_numbers` may be a path to a table, a `NamedTuple` `(; h, k)` indexed by degree + 1, or `nothing`
for [`default_load_love_numbers`](@ref).
"""
function Oceans.spherical_harmonic_loading_operator(grid::Oceananigans.Grids.AbstractGrid;
                                                    truncation = 40,
                                                    nlat_half = 24,
                                                    love_numbers = nothing,
                                                    reference_density = 1035,
                                                    earth_density = 5517)

    2nlat_half > truncation || throw(ArgumentError("nlat_half = $nlat_half is too small for degree $truncation"))

    gaussian_grid = FullGaussianGrid(nlat_half)
    ocean_grid = horizontal_grid(grid)
    manifold = ConservativeRegridding.GOCore.best_manifold(ocean_grid)
    regridder = Regridder(manifold, gaussian_grid, ocean_grid)
    regridder = on_architecture(architecture(grid), regridder)

    gaussian_field = RingGrids.Field(Float64, gaussian_grid)
    gaussian_values = on_architecture(architecture(grid), zeros(Float64, length(gaussian_field.data)))
    transform = SpectralTransform(gaussian_grid; truncation = truncation + 1, NF = Float64)
    coefficients = SpeedyTransforms.transform(gaussian_field, transform)

    (; h, k) = load_love_numbers(love_numbers)
    length(h) > truncation || throw(ArgumentError("load Love numbers stop at degree $(length(h) - 1) < $truncation"))
    γ(ℓ) = 3 * reference_density / earth_density * (1 + k[ℓ+1] - h[ℓ+1]) / (2ℓ + 1)
    weights = [γ(ℓ - 1) for (ℓ, m) in eachharmonic(coefficients)]

    FT = eltype(grid)
    device_coefficients = on_architecture(architecture(grid), zeros(Complex{FT}, length(coefficients.data)))
    recursion = on_architecture(architecture(grid), legendre_recursion_coefficients(FT, truncation))

    return SpectralLoadingOperator(regridder, gaussian_values, gaussian_field, transform, coefficients,
                                   weights, device_coefficients, recursion, truncation)
end

# Coefficients of the recursion for orthonormal associated Legendre functions P̄ₗₘ(x), stored like the spherical
# harmonic coefficients (by order m, then by degree ℓ ≥ m):
#   P̄ₘₘ = - √((2m + 1) / 2m) √(1 - x²) P̄ₘ₋₁ₘ₋₁       (column 1 holds the factor, at ℓ = m)
#   P̄ₗₘ = aₗₘ (x P̄ₗ₋₁ₘ - bₗₘ P̄ₗ₋₂ₘ)                   (columns 1 and 2, at ℓ > m)
function legendre_recursion_coefficients(FT, L)
    coefficients = zeros(FT, (L + 1) * (L + 2) ÷ 2, 2)
    n = 1
    for m in 0:L
        coefficients[n, 1] = m == 0 ? 1 / sqrt(4 * big(π)) : - sqrt(big(2m + 1) / 2m)
        for ℓ in m+1:L
            coefficients[n + ℓ - m, 1] = sqrt(big(4ℓ^2 - 1) / (ℓ^2 - m^2))
            coefficients[n + ℓ - m, 2] = sqrt(big((ℓ - 1)^2 - m^2) / (4 * (ℓ - 1)^2 - 1))
        end
        n += L + 1 - m
    end
    return coefficients
end

function Oceans.apply_loading_operator!(Ψ, op::SpectralLoadingOperator, pᵇ)
    grid = pᵇ.grid
    Nx, Ny, _ = size(grid)

    # Conservative average onto the Gaussian grid; cells the ocean grid does not reach carry no load
    src = op.regridder.src_temp
    reshape(src, Nx, Ny) .= view(interior(pᵇ), :, :, 1)
    regrid!(op.gaussian_values, op.regridder, src)
    op.gaussian_values .= ifelse.(isfinite.(op.gaussian_values), op.gaussian_values, 0)

    # The Gaussian grid is small, so the transform runs on the CPU
    copyto!(op.gaussian_field.data, op.gaussian_values)
    transform!(op.coefficients, op.gaussian_field, op.transform)
    op.coefficients.data .*= op.weights
    copyto!(op.device_coefficients, op.coefficients.data)

    launch!(architecture(grid), grid, :xy, _synthesize_loading_potential!, Ψ, grid, op.device_coefficients, op.recursion, op.degree)

    return nothing
end

# Ψ = - Σₗₘ γₗ pᵇₗₘ Yₗₘ evaluated at cell centers, with orthonormal associated Legendre functions computed by
# recursion. Coefficients are stored as SpeedyTransforms stores them: by order m, then by degree ℓ ≥ m.
@kernel function _synthesize_loading_potential!(Ψ, grid, coefficients, recursion, L)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)

    λ = deg2rad(λnode(i, j, 1, grid, Center(), Center(), Center()))
    φ = deg2rad(φnode(i, j, 1, grid, Center(), Center(), Center()))
    x = sin(φ)
    y = cos(φ)

    ψ = zero(FT)
    Pᵐᵐ = one(FT)
    n = 1

    for m in 0:L
        Pᵐᵐ *= @inbounds recursion[n, 1] * ifelse(m == 0, one(FT), y)
        Pˡ⁻² = zero(FT)
        Pˡ⁻¹ = Pᵐᵐ
        S = @inbounds coefficients[n] * Pᵐᵐ

        for q in n+1:n+L-m
            Pˡ = @inbounds recursion[q, 1] * (x * Pˡ⁻¹ - recursion[q, 2] * Pˡ⁻²)
            S += @inbounds coefficients[q] * Pˡ
            Pˡ⁻², Pˡ⁻¹ = Pˡ⁻¹, Pˡ
        end

        ψ += ifelse(m == 0, 1, 2) * real(S * cis(m * λ))
        n += L + 1 - m
    end

    @inbounds Ψ[i, j, 1] = - ψ
end

end # module
