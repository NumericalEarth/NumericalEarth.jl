#####
##### Depth-layer combination: 3-D texture and bulk density → one effective set of van
##### Genuchten parameters per column, for the single-layer `VariablySaturatedHydrology` slab.
#####

"""
$(TYPEDSIGNATURES)

Per-layer thicknesses (m) inside the soil column `[-slab_depth, 0]`, deepest-first to match
the dataset vertical axis. `z_interfaces` are the layer faces increasing upward
(e.g. `[-1.0, -0.6, -0.3, 0.0]`); layer `k` spans `[z_interfaces[k], z_interfaces[k+1]]`,
and layers outside the column get zero weight.

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
    D  = float(slab_depth)
    FT = typeof(D)
    return FT[max(0, min(FT(z_interfaces[k+1]), 0) - max(FT(z_interfaces[k]), -D))
              for k in 1:length(z_interfaces)-1]
end

"""
$(TYPEDSIGNATURES)

Depth below the surface of each layer's midpoint (m, positive down), deepest-first
to match [`layer_weights`](@ref).

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

@inline retention_residual(𝓃, 𝒮¹, 𝒮², logψ) =
    logexpm1(-log(𝒮¹) / van_genuchten_m(𝓃)) -
    logexpm1(-log(𝒮²) / van_genuchten_m(𝓃)) - 𝓃 * logψ

"""
$(TYPEDSIGNATURES)

The `(αᵃᵉ, 𝓃)` whose van Genuchten curve passes through water contents `θ¹` and `θ²` at
suction heads `ψ¹` and `ψ²`, given `θʳ` and `ν`. Eliminating `αᵃᵉ` leaves one equation in `𝓃`,

    log[(𝒮¹^(-1/𝓂) - 1) / (𝒮²^(-1/𝓂) - 1)] = 𝓃 log(ψ¹/ψ²),   𝓂 = 1 - 1/𝓃,

whose residual increases monotonically in `𝓃`; it is bisected over `1.01 ≤ 𝓃 ≤ 12`, and
`αᵃᵉ` follows from the first constraint. A root outside that bracket, or a pair of water
contents that determines none, returns `NaN`.
"""
@inline function matched_retention_parameters(θ¹, θ², θʳ, ν, ψ¹, ψ²)
    FT = typeof(θ¹)
    Δ  = ν - θʳ
    ϵ  = convert(FT, 1//1_000_000)
    𝒮¹ = clamp((θ¹ - θʳ) / Δ, 0, 1 - ϵ)
    𝒮² = clamp((θ² - θʳ) / Δ, 0, 1 - ϵ)
    logψ = log(ψ¹ / ψ²)

    lo = convert(FT, 101//100)
    hi = convert(FT, 12)
    # a non-finite residual fails the sign test, since `NaN < 0` is false
    bracketed = (retention_residual(lo, 𝒮¹, 𝒮², logψ) < 0) &
                (retention_residual(hi, 𝒮¹, 𝒮², logψ) > 0)

    for _ in 1:40
        𝓃  = (lo + hi) / 2
        up = retention_residual(𝓃, 𝒮¹, 𝒮², logψ) > 0
        hi = ifelse(up, 𝓃, hi)
        lo = ifelse(up, lo, 𝓃)
    end

    𝓃 = (lo + hi) / 2
    αᵃᵉ = exp(logexpm1(-log(𝒮¹) / van_genuchten_m(𝓃)) / 𝓃) / ψ¹

    return ifelse(bracketed, αᵃᵉ, convert(FT, NaN)),
           ifelse(bracketed, 𝓃, convert(FT, NaN))
end

@kernel function _soil_hydraulic_properties!(ν, θʳ, αᵃᵉ, 𝓃, K₀, ηᴷ,
                                            sand, silt, clay, bulk_density,
                                            Δz, depths, ΣΔz, Nz, ψ¹, ψ², ptf)
    i, j = @index(Global, NTuple)
    FT = eltype(ν)

    Σν = zero(FT); Σθʳ = zero(FT); Σθ¹ = zero(FT); Σθ² = zero(FT)
    Σηᴷ = zero(FT); ΣR = zero(FT)

    @inbounds for k in 1:Nz
        Δzk = Δz[k]
        p   = soil_hydraulic_parameters(ptf, sand[i, j, k], silt[i, j, k],
                                        clay[i, j, k], bulk_density[i, j, k], depths[k])
        νk  = p.porosity
        θʳk = p.residual_liquid_fraction
        nk  = p.pore_size_uniformity
        θ¹  = θʳk + (νk - θʳk) * van_genuchten_saturation(p.inverse_air_entry_head * ψ¹, nk)
        θ²  = θʳk + (νk - θʳk) * van_genuchten_saturation(p.inverse_air_entry_head * ψ², nk)
        # `0 * NaN` is NaN, so zero weight alone would not drop an out-of-column layer
        inside = Δzk > 0
        Σν  += ifelse(inside, Δzk * νk, zero(FT))
        Σθʳ += ifelse(inside, Δzk * θʳk, zero(FT))
        Σθ¹ += ifelse(inside, Δzk * θ¹, zero(FT))
        Σθ² += ifelse(inside, Δzk * θ², zero(FT))
        Σηᴷ += ifelse(inside, Δzk * p.pore_connectivity_exponent, zero(FT))
        # layers in series add resistance Δz / K, hence a harmonic mean for K₀
        ΣR  += ifelse(inside, Δzk / p.matching_point_conductivity, zero(FT))
    end

    @inbounds begin
        ν[i, j, 1]  = Σν / ΣΔz
        θʳ[i, j, 1] = Σθʳ / ΣΔz
        αᵃᵉ[i, j, 1], 𝓃[i, j, 1] = matched_retention_parameters(Σθ¹ / ΣΔz, Σθ² / ΣΔz,
                                                              Σθʳ / ΣΔz, Σν / ΣΔz, ψ¹, ψ²)
        K₀[i, j, 1] = ΣΔz / ΣR
        ηᴷ[i, j, 1] = Σηᴷ / ΣΔz
    end
end

"""
$(TYPEDSIGNATURES)

Reduce the 3-D texture (`sand`, `silt`, `clay`, kg/kg) and `bulk_density` (kg/m³) `Field`s
to a NamedTuple of 2-D `Field{Center, Center, Nothing}` van Genuchten properties

    (; porosity, residual_liquid_fraction, inverse_air_entry_head,
       pore_size_uniformity, matching_point_conductivity, pore_connectivity_exponent)

whose keys match the keyword arguments of [`VariablySaturatedHydrology`](@ref),
[`VanGenuchtenRetention`](@ref) and [`VanGenuchtenConductivity`](@ref).

The pedotransfer function `ptf` is applied to each depth layer of the inputs' grid, at that
layer's depth, and the layers inside `[-slab_depth, 0]` are combined so that the slab
reproduces the thickness-weighted mean retention curve of the column: `ν`, `θʳ` and `ηᴷ`
are thickness-weighted arithmetic means, `K₀` is the harmonic mean of layers in series, and
`αᵃᵉ` and `𝓃` are solved so the effective curve passes through the mean curve at the two
suction heads `matching_heads` (m). The defaults are field capacity and the permanent
wilting point, with field capacity at 1 m after [Balsamo et al. (2009)](@cite balsamo2009).

A cell that is `NaN` in any input is `NaN` in every predicted output; fill the texture
first, e.g. with `Field(metadatum; inpainting = NearestNeighborInpainting(n))`.
"""
function soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                   slab_depth,
                                   ptf = WeynantsPedotransfer(),
                                   matching_heads = (1, 150))
    grid = sand.grid
    arch = architecture(grid)
    FT   = eltype(sand)

    all(f -> f.grid == grid && size(f) == size(sand), (silt, clay, bulk_density)) ||
        throw(ArgumentError("sand, silt, clay and bulk_density must share one grid"))

    ψ¹, ψ² = matching_heads
    0 < ψ¹ < ψ² ||
        throw(ArgumentError("matching_heads must be two increasing positive suction heads, found $matching_heads"))

    z_interfaces = on_architecture(CPU(), znodes(grid, Face()))
    Δz  = layer_weights(z_interfaces, slab_depth)
    ΣΔz = sum(Δz)
    ΣΔz > 0 ||
        throw(ArgumentError("slab_depth = $slab_depth does not overlap the soil column $(extrema(z_interfaces))"))

    ν, θʳ, αᵃᵉ, 𝓃, K₀, ηᴷ = ntuple(_ -> Field{Center, Center, Nothing}(grid), 6)

    launch!(arch, grid, :xy, _soil_hydraulic_properties!,
            ν, θʳ, αᵃᵉ, 𝓃, K₀, ηᴷ,
            sand, silt, clay, bulk_density,
            on_architecture(arch, convert.(FT, Δz)),
            on_architecture(arch, convert.(FT, layer_depths(z_interfaces))),
            convert(FT, ΣΔz), size(sand, 3), convert(FT, ψ¹), convert(FT, ψ²),
            convert_eltype(FT, ptf))

    return (porosity = ν,
            residual_liquid_fraction = θʳ,
            inverse_air_entry_head = αᵃᵉ,
            pore_size_uniformity = 𝓃,
            matching_point_conductivity = K₀,
            pore_connectivity_exponent = ηᴷ)
end
