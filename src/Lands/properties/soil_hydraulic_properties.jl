#####
##### Depth-layer combination: reduce 3-D texture + bulk-density fields to a single
##### effective set of van Genuchten parameters per horizontal column, for the
##### single-layer `VariablySaturatedHydrology` slab.
#####
##### The PTF is applied per depth layer, then the layers are combined. The slab carries
##### one storage variable `M` at one pressure head, so `M(ψ)/W` *is* the
##### thickness-weighted mean retention curve θ̄(ψ) = Σ wₖθₖ(ψ)/W. That fixes the
##### reduction with no free choices:
#####   * ν, θʳ  — thickness-weighted arithmetic mean, exactly θ̄ at ψ = 0 and ψ → ∞.
#####   * α, n   — chosen so the effective curve passes through θ̄ at two intermediate
#####               heads as well, `matching_heads`. Four constraints, four parameters.
#####   * K₀     — thickness-weighted harmonic mean; layers in series add resistance,
#####               so a clay horizon throttles vertical drainage. Exact at ψ = 0.
#####   * ηᴷ     — thickness-weighted arithmetic mean. Weighting by resistance instead
#####               is the correct local slope of log K at fixed 𝒮, but the weights
#####               themselves move as the layers dry, and it scores worse over the
#####               range the slab operates in.
#####
##### Against a sand-over-clay column, the reduced retention curve tracks θ̄ to 0.002 over
##### four decades of suction, and the reduced K stays within a factor of 3.7 of the column's
##### series resistance while remaining exact at saturation.
#####

"""
$(TYPEDSIGNATURES)

Per-layer thicknesses (m), deepest-first to match the dataset vertical axis, clipped to
the soil column `[-slab_depth, 0]`. `z_interfaces` are the layer faces increasing upward
(e.g. `[-1.0, -0.6, -0.3, 0.0]`); layer `k` spans `[z_interfaces[k], z_interfaces[k+1]]`.
Layers outside the column get zero weight.

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
$(TYPEDSIGNATURES)

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

@inline retention_residual(n, 𝒮¹, 𝒮², logψ) =
    logexpm1(-log(𝒮¹) / van_genuchten_m(n)) -
    logexpm1(-log(𝒮²) / van_genuchten_m(n)) - n * logψ

"""
$(TYPEDSIGNATURES)

The `(α, n)` whose van Genuchten curve passes through water contents `θ¹` and `θ²` at
suction heads `ψ¹` and `ψ²`, given `θʳ` and `ν`.

Eliminating `α` between the two constraints leaves one equation in `n`,

    log[(𝒮¹^(-1/m) - 1) / (𝒮²^(-1/m) - 1)] = n log(ψ¹/ψ²),   m = 1 - 1/n,

whose residual (left side less right side) rises monotonically from `-∞` as `n → 1` to `+∞`
as `n → ∞`. There is therefore exactly one root, and one sign change across the search
bracket `1.01 ≤ n ≤ 12` establishes that it lies inside. `α` then follows from either
constraint.

That bracket holds every root the shipped pedotransfer functions reach: their per-layer `n`
spans 1.05 to 2.25, and combining contrasting layers flattens the mean curve only as far as
1.07. A root outside it comes back as `NaN` rather than as the bracket end, as does a pair of
water contents that determines no root at all.
"""
@inline function matched_retention_parameters(θ¹, θ², θʳ, ν, ψ¹, ψ²)
    FT = typeof(θ¹)
    Δ  = ν - θʳ
    ϵ  = convert(FT, 1//1_000_000)
    # Only the wet end needs a margin: `logexpm1` keeps the dry branch representable down to
    # `θ = θʳ`, where the residual turns non-finite and the sign test below reads it as
    # missing data. The zero floor also keeps `log` off the negative axis for `θ < θʳ`.
    𝒮¹ = clamp((θ¹ - θʳ) / Δ, zero(FT), one(FT) - ϵ)
    𝒮² = clamp((θ² - θʳ) / Δ, zero(FT), one(FT) - ϵ)
    logψ = log(ψ¹ / ψ²)

    lo = convert(FT, 101//100)
    hi = convert(FT, 12)
    # One sign change across the bracket is the whole test, the residual being monotone. A
    # non-finite residual fails it, since `NaN < 0` is false.
    bracketed = (retention_residual(lo, 𝒮¹, 𝒮², logψ) < 0) &
                (retention_residual(hi, 𝒮¹, 𝒮², logψ) > 0)

    for _ in 1:40                                    # accuracy plateaus at 34
        n  = (lo + hi) / 2
        up = retention_residual(n, 𝒮¹, 𝒮², logψ) > 0
        hi = ifelse(up, n, hi)
        lo = ifelse(up, lo, n)
    end

    n = (lo + hi) / 2
    α = exp(logexpm1(-log(𝒮¹) / van_genuchten_m(n)) / n) / ψ¹

    return ifelse(bracketed, α, convert(FT, NaN)),
           ifelse(bracketed, n, convert(FT, NaN))
end

@kernel function _soil_hydraulic_properties!(porosity, residual, α, n, K₀, ηᴷ,
                                            sand, silt, clay, bulk_density,
                                            w, depths, W, Nz, ψ¹, ψ², fallback, ptf)
    i, j = @index(Global, NTuple)
    FT = eltype(porosity)

    Σν = zero(FT); Σθʳ = zero(FT); Σθ¹ = zero(FT); Σθ² = zero(FT)
    Σηᴷ = zero(FT); ΣR = zero(FT)

    @inbounds for k in 1:Nz
        wk = w[k]
        texture = (sand[i, j, k], silt[i, j, k], clay[i, j, k], bulk_density[i, j, k])
        # Rock, water and out-of-coverage cells arrive as NaN. `fallback` is what stands in
        # for them, itself NaN unless the caller supplied a `fallback_texture`.
        gap = isnan(texture[1]) | isnan(texture[2]) | isnan(texture[3]) | isnan(texture[4])
        p   = soil_hydraulic_parameters(ptf, ifelse(gap, fallback, texture)..., depths[k])
        νk  = p.porosity
        θʳk = p.residual_liquid_fraction
        Δk  = νk - θʳk
        nk  = p.pore_size_uniformity
        θ¹  = θʳk + Δk * van_genuchten_saturation(p.inverse_air_entry_head * ψ¹, nk)
        θ²  = θʳk + Δk * van_genuchten_saturation(p.inverse_air_entry_head * ψ², nk)
        # `0 * NaN` is NaN, so zero weight alone would not keep a missing-data layer below
        # `slab_depth` out of the column.
        inside = wk > 0
        Σν  += ifelse(inside, wk * νk, zero(FT))
        Σθʳ += ifelse(inside, wk * θʳk, zero(FT))
        Σθ¹ += ifelse(inside, wk * θ¹, zero(FT))
        Σθ² += ifelse(inside, wk * θ², zero(FT))
        Σηᴷ += ifelse(inside, wk * p.pore_connectivity_exponent, zero(FT))
        # Layers in series add resistance R = w/K, which is what makes the mean harmonic.
        ΣR  += ifelse(inside, wk / p.matching_point_conductivity, zero(FT))
    end

    νᶜ  = Σν / W
    θʳᶜ = Σθʳ / W
    αᶜ, nᶜ = matched_retention_parameters(Σθ¹ / W, Σθ² / W, θʳᶜ, νᶜ, ψ¹, ψ²)

    @inbounds begin
        porosity[i, j, 1] = νᶜ
        residual[i, j, 1] = θʳᶜ
        α[i, j, 1]        = αᶜ
        n[i, j, 1]        = nᶜ
        K₀[i, j, 1]       = W / ΣR
        ηᴷ[i, j, 1]       = Σηᴷ / W
    end
end

"""
$(TYPEDSIGNATURES)

Reduce the 3-D texture (`sand`, `silt`, `clay`, kg/kg) and `bulk_density` (kg/m³)
`Field`s to a NamedTuple of 2-D effective van Genuchten properties

    (; porosity, residual_liquid_fraction, inverse_air_entry_head,
       pore_size_uniformity, matching_point_conductivity, pore_connectivity_exponent)

whose keys match the keyword arguments of [`VariablySaturatedHydrology`](@ref),
[`VanGenuchtenRetention`](@ref), and [`VanGenuchtenConductivity`](@ref). The
pedotransfer function `ptf` is applied per depth layer — reading topsoil or subsoil
off each layer's depth — and the layers are then combined over `slab_depth`.

Because the slab carries one storage variable at one pressure head, the object to
reproduce is the thickness-weighted mean retention curve. `ν` and `θʳ` are arithmetic
means, which is exact at `ψ = 0` and `ψ → ∞`, and `α` and `n` are then solved for so the
effective curve also passes through the mean curve at `matching_heads` (m). `K₀` is a
harmonic mean, since layers in series add resistance, and `ηᴷ` is an arithmetic one.
The layer faces come from the inputs' own grid; see [`layer_weights`](@ref).

The default heads are field capacity and the permanent wilting point, which bracket the
range the slab operates in. 1 m rather than the textbook 3.3 m follows
[Balsamo et al. (2009)](@cite balsamo2009), who measured `-0.10` bar as the better field
capacity for van Genuchten parameters of this family; averaged over contrasting columns
the choice is worth little, hence a keyword rather than a constant.

The parameters describe the soil inside `[-slab_depth, 0]` and nothing below it. Soil
below the slab belongs to the deep-flux closure.

Rock, water and out-of-coverage cells arrive as `NaN`. `fallback_texture` is the texture
(a NamedTuple of `sand`, `silt`, `clay` and `bulk_density`) substituted for them instead,
e.g. `(sand = 0.4, silt = 0.4, clay = 0.2, bulk_density = 1400)` for a nominal loam.
Parameters a pedotransfer function holds constant never read the data and so come back as
that constant either way — `θʳ` for [`WeynantsPedotransfer`](@ref), `θʳ` and `ηᴷ` for
[`HYPRESPedotransfer`](@ref) — so a mask built from these fields has to be built on a predicted one.

`matching_point_conductivity` inherits whatever `ptf` means by it, which for
[`WeynantsPedotransfer`](@ref) is a matrix matching point rather than the value an
infiltration cap wants (see [`saturated_conductivity`](@ref)).

Each output is a `Field{Center, Center, Nothing}` on the inputs' grid, read by the slab at
`[i, j]`. `slab_depth` must be a scalar.
"""
function soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                   slab_depth,
                                   ptf = WeynantsPedotransfer(),
                                   matching_heads = (1, 150),
                                   fallback_texture = nothing)
    grid = sand.grid
    arch = architecture(grid)
    FT   = eltype(sand)
    Nz   = size(sand, 3)

    # Grid equality catches a field read over a different region, which `size` alone does not;
    # `size` catches a windowed view, whose grid is its parent's.
    for (name, field) in pairs((; silt, clay, bulk_density))
        (field.grid == grid && size(field) == size(sand)) ||
            throw(ArgumentError("$name is $(size(field)) on $(summary(field.grid)) but sand " *
                                "is $(size(sand)) on $(summary(grid)); the four inputs must " *
                                "share one grid"))
    end

    z_interfaces = znodes(grid, Face())

    ψ¹, ψ² = matching_heads
    0 < ψ¹ < ψ² ||
        throw(ArgumentError("matching_heads must be two increasing positive suction " *
                            "heads (m); got $matching_heads"))

    weights = layer_weights(z_interfaces, slab_depth)
    W = sum(weights)
    W > 0 ||
        throw(ArgumentError("slab_depth = $slab_depth does not overlap the soil column " *
                            "spanned by the grid's z interfaces $(collect(z_interfaces))"))

    w      = on_architecture(arch, convert.(FT, weights))
    depths = on_architecture(arch, convert.(FT, layer_depths(z_interfaces)))

    fallback = if isnothing(fallback_texture)
        ntuple(_ -> convert(FT, NaN), 4)
    else
        map(name -> convert(FT, getproperty(fallback_texture, name)),
            (:sand, :silt, :clay, :bulk_density))
    end

    porosity = Field{Center, Center, Nothing}(grid)
    residual = Field{Center, Center, Nothing}(grid)
    α        = Field{Center, Center, Nothing}(grid)
    n        = Field{Center, Center, Nothing}(grid)
    K₀       = Field{Center, Center, Nothing}(grid)
    ηᴷ       = Field{Center, Center, Nothing}(grid)

    launch!(arch, grid, :xy, _soil_hydraulic_properties!,
            porosity, residual, α, n, K₀, ηᴷ,
            sand, silt, clay, bulk_density,
            w, depths, convert(FT, W), Nz, convert(FT, ψ¹), convert(FT, ψ²),
            fallback, convert_eltype(FT, ptf))

    return (porosity = porosity,
            residual_liquid_fraction = residual,
            inverse_air_entry_head = α,
            pore_size_uniformity = n,
            matching_point_conductivity = K₀,
            pore_connectivity_exponent = ηᴷ)
end
