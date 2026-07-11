#####
##### Depth-layer combination: reduce 3-D texture + bulk-density fields to a single
##### effective set of van Genuchten parameters per horizontal column, for the
##### single-layer `VariablySaturatedHydrology` slab.
#####
##### The PTF is applied per depth layer, then each parameter is upscaled over the
##### part of the soil column inside `slab_depth` with its physically correct law:
#####   * ν, θʳ, n  — thickness-weighted arithmetic mean (storage adds in volume)
#####   * Kₛ        — thickness-weighted harmonic mean (layers in series; a clay
#####                 horizon throttles vertical drainage)
#####   * α         — thickness-weighted geometric mean (α spans orders of magnitude)
#####

"""
    layer_weights(z_interfaces, slab_depth)

Per-layer thicknesses (m), deepest-first to match the dataset vertical axis,
clipped to the soil column `[-slab_depth, 0]`. `z_interfaces` are the layer faces
increasing upward (e.g. `[-1.0, -0.6, -0.3, 0.0]`); layer `k` spans
`[z_interfaces[k], z_interfaces[k+1]]`. Layers outside the column get zero weight,
so a thin `slab_depth` degenerates to using only the near-surface layer(s).

```jldoctest
using NumericalEarth

layer_weights([-1.0, -0.6, -0.3, 0.0], 0.3)

# output
3-element Vector{Float64}:
 0.0
 0.0
 0.3
```
"""
function layer_weights(z_interfaces, slab_depth)
    slab_depth isa Number ||
        throw(ArgumentError("layer_weights requires a scalar slab_depth; got $(typeof(slab_depth))"))
    D  = float(slab_depth)
    FT = typeof(D)
    Nz = length(z_interfaces) - 1
    return FT[max(zero(FT),
                  min(FT(z_interfaces[k+1]), zero(FT)) - max(FT(z_interfaces[k]), -D))
              for k in 1:Nz]
end

@kernel function _soil_hydraulic_properties!(porosity, residual, α, n, K_saturated,
                                             sand, silt, clay, bulk_density,
                                             w, W, Nz, ptf)
    i, j = @index(Global, NTuple)
    FT = eltype(porosity)

    Σν = zero(FT); Σθʳ = zero(FT); Σn = zero(FT)
    Σln_α = zero(FT); Σw_over_K = zero(FT)

    @inbounds for k in 1:Nz
        wk = w[k]
        p  = soil_hydraulic_parameters(ptf, sand[i, j, k], silt[i, j, k],
                                       clay[i, j, k], bulk_density[i, j, k])
        Σν        += wk * p.ν
        Σθʳ       += wk * p.θʳ
        Σn        += wk * p.n
        Σln_α     += wk * log(p.α)
        Σw_over_K += wk / p.K_saturated
    end

    @inbounds begin
        porosity[i, j, 1]    = Σν / W
        residual[i, j, 1]    = Σθʳ / W
        n[i, j, 1]           = Σn / W
        α[i, j, 1]           = exp(Σln_α / W)     # geometric
        K_saturated[i, j, 1] = W / Σw_over_K      # harmonic
    end
end

"""
    soil_hydraulic_properties(sand, silt, clay, bulk_density;
                              slab_depth, z_interfaces, ptf = ContinuousPedotransfer())

Reduce the 3-D texture (`sand`, `silt`, `clay`, kg/kg) and `bulk_density` (kg/m³)
`Field`s to a NamedTuple of 2-D effective van Genuchten properties

    (; porosity, residual_liquid_fraction, α, n, K_saturated)

for [`VariablySaturatedHydrology`](@ref). The pedotransfer function `ptf` is applied
per depth layer, then each parameter is upscaled over `slab_depth` using its
per-parameter law (arithmetic `ν`/`θʳ`/`n`, harmonic `K_saturated`, geometric `α`;
see [`layer_weights`](@ref)).

Each output is a `Field{Center, Center, Nothing}` on the inputs' grid — a 2-D field
the slab reads at `[i, j]`. `slab_depth` must be a scalar; `z_interfaces` are the
dataset layer faces (e.g. `DataWrangling.z_interfaces(OpenLandMapSoilDB())`).
"""
function soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                   slab_depth, z_interfaces, ptf = ContinuousPedotransfer())
    grid = sand.grid
    arch = architecture(grid)
    FT   = eltype(sand)
    Nz   = size(sand, 3)

    length(z_interfaces) == Nz + 1 ||
        throw(ArgumentError("z_interfaces must have length size(sand, 3) + 1 = $(Nz + 1); " *
                            "got $(length(z_interfaces))"))

    weights = layer_weights(z_interfaces, slab_depth)
    W = sum(weights)
    W > 0 ||
        throw(ArgumentError("slab_depth = $slab_depth does not overlap the soil column " *
                            "spanned by z_interfaces = $z_interfaces"))

    w = on_architecture(arch, convert.(FT, weights))

    porosity    = Field{Center, Center, Nothing}(grid)
    residual    = Field{Center, Center, Nothing}(grid)
    α           = Field{Center, Center, Nothing}(grid)
    n           = Field{Center, Center, Nothing}(grid)
    K_saturated = Field{Center, Center, Nothing}(grid)

    launch!(arch, grid, :xy, _soil_hydraulic_properties!,
            porosity, residual, α, n, K_saturated,
            sand, silt, clay, bulk_density,
            w, convert(FT, W), Nz, ptf)

    return (porosity = porosity,
            residual_liquid_fraction = residual,
            α = α,
            n = n,
            K_saturated = K_saturated)
end
