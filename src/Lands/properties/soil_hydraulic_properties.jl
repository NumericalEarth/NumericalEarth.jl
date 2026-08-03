#####
##### Depth-layer combination: reduce 3-D texture + bulk-density fields to a single
##### effective set of van Genuchten parameters per horizontal column, for the
##### single-layer `VariablySaturatedHydrology` slab.
#####
##### The PTF is applied per depth layer, then each parameter is upscaled over the
##### part of the soil column inside `slab_depth` with its physically correct law:
#####   * ν, θʳ     — thickness-weighted arithmetic mean (storage adds in volume)
#####   * Kₛ        — thickness-weighted harmonic mean (layers in series; a clay
#####                 horizon throttles vertical drainage)
#####   * α         — thickness-weighted geometric mean (α spans orders of magnitude)
#####   * n         — geometric mean of λ = n - 1, the pore-size index the regression
#####                 is fitted in. Mixing contrasting layers flattens the column's
#####                 retention curve, so the effective n must fall toward the
#####                 smaller value; averaging n itself instead biases it high.
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
    issorted(z_interfaces) ||
        throw(ArgumentError("z_interfaces must increase upward, deepest face first; " *
                            "got $z_interfaces"))
    D  = float(slab_depth)
    FT = typeof(D)
    Nz = length(z_interfaces) - 1
    return FT[max(zero(FT),
                  min(FT(z_interfaces[k+1]), zero(FT)) - max(FT(z_interfaces[k]), -D))
              for k in 1:Nz]
end

"""
    layer_depths(z_interfaces)

Depth below the surface of each layer's midpoint (m, positive down), deepest-first
to match [`layer_weights`](@ref). The pedotransfer function reads topsoil or subsoil
off these.

```jldoctest
using NumericalEarth

layer_depths([-1.0, -0.5, 0.0])

# output
2-element Vector{Float64}:
 0.75
 0.25
```
"""
layer_depths(z_interfaces) =
    [-(float(z_interfaces[k]) + float(z_interfaces[k+1])) / 2
     for k in 1:length(z_interfaces)-1]

@kernel function _soil_hydraulic_properties!(porosity, residual, α, n, K_saturated,
                                            sand, silt, clay, bulk_density,
                                            w, depths, W, Nz, ptf)
    i, j = @index(Global, NTuple)
    FT = eltype(porosity)

    Σν = zero(FT); Σθʳ = zero(FT)
    Σln_α = zero(FT); Σln_λ = zero(FT); Σw_over_K = zero(FT)

    @inbounds for k in 1:Nz
        wk = w[k]
        p  = soil_hydraulic_parameters(ptf, sand[i, j, k], silt[i, j, k],
                                       clay[i, j, k], bulk_density[i, j, k], depths[k])
        # A layer outside the slab must contribute nothing at all: `0 * NaN` is NaN,
        # so without this mask one missing-data layer below `slab_depth` would carry
        # into every parameter of a column it has no business touching.
        inside = wk > 0
        Σν        += ifelse(inside, wk * p.porosity, zero(FT))
        Σθʳ       += ifelse(inside, wk * p.residual_liquid_fraction, zero(FT))
        Σln_α     += ifelse(inside, wk * log(p.inverse_air_entry_head), zero(FT))
        Σln_λ     += ifelse(inside, wk * log(p.pore_size_uniformity - 1), zero(FT))
        Σw_over_K += ifelse(inside, wk / p.K_saturated, zero(FT))
    end

    @inbounds begin
        porosity[i, j, 1]    = Σν / W
        residual[i, j, 1]    = Σθʳ / W
        α[i, j, 1]           = exp(Σln_α / W)         # geometric
        n[i, j, 1]           = 1 + exp(Σln_λ / W)     # geometric in λ = n - 1
        K_saturated[i, j, 1] = W / Σw_over_K          # harmonic
    end
end

"""
    soil_hydraulic_properties(sand, silt, clay, bulk_density;
                              slab_depth, z_interfaces, ptf = ContinuousPedotransfer())

Reduce the 3-D texture (`sand`, `silt`, `clay`, kg/kg) and `bulk_density` (kg/m³)
`Field`s to a NamedTuple of 2-D effective van Genuchten properties

    (; porosity, residual_liquid_fraction, inverse_air_entry_head, pore_size_uniformity, K_saturated)

whose keys match the keyword arguments of [`VariablySaturatedHydrology`](@ref),
[`VanGenuchtenRetention`](@ref), and [`VanGenuchtenConductivity`](@ref). The
pedotransfer function `ptf` is applied per depth layer — reading topsoil or subsoil
off each layer's depth — then each parameter is upscaled over `slab_depth` using its
own law: arithmetic `ν`/`θʳ`, harmonic `K_saturated`, geometric `α`, and geometric in
`n - 1` for `pore_size_uniformity` (see [`layer_weights`](@ref)).

The parameters describe the soil inside `[-slab_depth, 0]` and nothing below it, so
the slab's storage capacity and pressure head refer to the volume it actually holds.
Soil below the slab belongs to the deep-flux closure, not to these properties.

Each output is a `Field{Center, Center, Nothing}` on the inputs' grid — a 2-D field
the slab reads at `[i, j]`. `slab_depth` must be a scalar; `z_interfaces` are the
dataset layer faces (e.g. `DataWrangling.z_interfaces(OpenLandMapSoilDB())`).

`ptf` is rebuilt at the inputs' float type before the kernel launches, since devices
reject float types they do not support (Metal has no `Float64`).
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

    w      = on_architecture(arch, convert.(FT, weights))
    depths = on_architecture(arch, convert.(FT, layer_depths(z_interfaces)))

    porosity    = Field{Center, Center, Nothing}(grid)
    residual    = Field{Center, Center, Nothing}(grid)
    α           = Field{Center, Center, Nothing}(grid)
    n           = Field{Center, Center, Nothing}(grid)
    K_saturated = Field{Center, Center, Nothing}(grid)

    launch!(arch, grid, :xy, _soil_hydraulic_properties!,
            porosity, residual, α, n, K_saturated,
            sand, silt, clay, bulk_density,
            w, depths, convert(FT, W), Nz, on_float_type(FT, ptf))

    return (porosity = porosity,
            residual_liquid_fraction = residual,
            inverse_air_entry_head = α,
            pore_size_uniformity = n,
            K_saturated = K_saturated)
end
