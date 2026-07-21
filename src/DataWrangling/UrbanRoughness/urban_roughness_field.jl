#####
##### Apply the urban morphometric closure over a grid: (built fraction λp, building
##### height H) → momentum roughness length z0 and zero-plane displacement d0.
#####

using Oceananigans: Center, launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field
using KernelAbstractions: @kernel, @index

@kernel function _compute_urban_roughness!(z0m, d0, λp, H, grid, method, estimator, p)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    @inbounds λij = convert(FT, λp[i, j, 1])
    @inbounds Hij = convert(FT, H[i, j, 1])
    z0, dd = urban_roughness_point(λij, Hij, method, estimator, p)
    @inbounds z0m[i, j, 1] = z0
    @inbounds d0[i, j, 1] = dd
end

method_code(method::Symbol) =
    method === :macdonald ? MACDONALD :
    method === :kanda     ? KANDA :
    method === :lookup    ? LOOKUP :
    throw(ArgumentError("urban roughness `method` must be :macdonald, :kanda, or :lookup, got :$method"))

estimator_code(estimator::Symbol) =
    estimator === :isotropic ? ISOTROPIC :
    estimator === :cuboid    ? CUBOID :
    throw(ArgumentError("urban roughness `frontal_area` must be :isotropic or :cuboid, got :$estimator"))

"""
$(TYPEDSIGNATURES)

Fill `z0m` and `d0` (metres) in place from a built-up-fraction field `λp` and a mean
building-height field `H` on `grid`, applying the urban morphometric closure over
every cell. `method` selects `:kanda` (default), `:macdonald`, or `:lookup`;
`frontal_area` selects the `:isotropic` (`λf ≈ λp`) or `:cuboid` frontal-area
estimator; remaining keywords override [`UrbanRoughnessParameters`](@ref).
"""
function compute_urban_roughness!(z0m, d0, λp, H, grid;
                                  method = :kanda,
                                  frontal_area = :isotropic,
                                  parameters = UrbanRoughnessParameters(eltype(grid)),
                                  kw...)
    arch = architecture(grid)
    p = isempty(kw) ? parameters : UrbanRoughnessParameters(parameters; kw...)
    launch!(arch, grid, :xy, _compute_urban_roughness!,
            z0m, d0, λp, H, grid, method_code(method), estimator_code(frontal_area), p)
    return z0m, d0
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0m` and zero-plane displacement `d0` (as `Field`s on the
grid of `H`) for the urban tile, from a mean building-height field `H` and a built-up
plan-area fraction field `λp`. Convenience wrapper around
[`compute_urban_roughness!`](@ref); accepts the same `method` / `frontal_area` /
parameter keywords. Where `λp → 0` the result reduces to a bare-soil roughness.
"""
function urban_roughness(H, λp; kw...)
    grid = H.grid
    z0m = Field{Center, Center, Nothing}(grid)
    d0  = Field{Center, Center, Nothing}(grid)
    compute_urban_roughness!(z0m, d0, λp, H, grid; kw...)
    return z0m, d0
end
