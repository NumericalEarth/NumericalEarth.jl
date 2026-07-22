#####
##### Apply an urban morphometric closure over a grid: (built fraction λp, building
##### height H) → momentum roughness length z0 and zero-plane displacement d0.
#####

using Oceananigans: Center, launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field
using KernelAbstractions: @kernel, @index

@kernel function _compute_urban_roughness!(z0m, d0, λp, H, grid, closure)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    @inbounds λij = convert(FT, λp[i, j, 1])
    @inbounds Hij = convert(FT, H[i, j, 1])
    z0ij, d0ij = roughness_lengths(closure, λij, Hij)
    @inbounds z0m[i, j, 1] = z0ij
    @inbounds d0[i, j, 1] = d0ij
end

"""
$(TYPEDSIGNATURES)

Fill `z0m` and `d0` (meters) in place from a built-up-fraction field `λp` and a mean
building-height field `H` on `grid`, applying `closure` (an [`AbstractUrbanRoughness`](@ref),
default [`KandaRoughness`](@ref)) over every cell.
"""
function compute_urban_roughness!(z0m, d0, λp, H, grid;
                                  closure = KandaRoughness(eltype(grid)))
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_urban_roughness!, z0m, d0, λp, H, grid, closure)
    return z0m, d0
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0m` and zero-plane displacement `d0` (as `Field`s on the grid
of `H`) for the urban tile, from a mean building-height field `H` and a built-up plan-area
fraction field `λp`. Convenience wrapper around [`compute_urban_roughness!`](@ref); pass a
`closure` (default [`KandaRoughness`](@ref)) to select the morphometry. Where `λp → 0` the
result reduces to a bare-soil roughness.
"""
function urban_roughness(H, λp; closure = KandaRoughness(eltype(H.grid)))
    grid = H.grid
    z0m = Field{Center, Center, Nothing}(grid)
    d0  = Field{Center, Center, Nothing}(grid)
    compute_urban_roughness!(z0m, d0, λp, H, grid; closure)
    return z0m, d0
end
