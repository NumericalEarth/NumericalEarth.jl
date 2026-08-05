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
#####   * Kₛ     — thickness-weighted harmonic mean; layers in series add resistance,
#####               so a clay horizon throttles vertical drainage. Exact at ψ = 0.
#####   * ℓ      — thickness-weighted arithmetic mean. Weighting by resistance instead
#####               is the correct local slope of log K at fixed 𝒮, but the weights
#####               themselves move as the layers dry, and it scores worse over the
#####               range the slab operates in.
#####
##### Upward evaporative flux through a layered column is not recoverable from any single
##### parameter set — the limiting layer moves as the column dries.
#####

"""
    layer_weights(z_interfaces, slab_depth)

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

"""
    matched_retention_parameters(θ¹, θ², θʳ, ν, ψ¹, ψ²)

The `(α, n)` whose van Genuchten curve passes through water contents `θ¹` and `θ²` at
suction heads `ψ¹` and `ψ²`, given `θʳ` and `ν`.

Eliminating `α` between the two constraints leaves one equation in `n`,

    log[(𝒮¹^(-1/m) - 1) / (𝒮²^(-1/m) - 1)] = n log(ψ¹/ψ²),   m = 1 - 1/n,

whose left side less its right side increases monotonically from `-∞` at `n → 1` to
`+∞`, so bisection cannot miss it. `α` then follows from either constraint.

Both constraints have to carry information, which fails if the soil has already drained
past roughly `𝒮 = 0.02` at the wetter head: the two water contents are then both within
rounding of `θʳ`, and `n` and `α` stay bounded but stop being meaningful. That needs
`n ≳ 3` together with `α ≳ 3 m⁻¹`, well outside what either shipped pedotransfer function
produces, but a much coarser one would want wetter `matching_heads`.
"""
@inline function matched_retention_parameters(θ¹, θ², θʳ, ν, ψ¹, ψ²)
    FT = typeof(θ¹)
    Δ  = ν - θʳ
    ϵ  = convert(FT, 1//1_000_000)
    𝒮¹ = clamp((θ¹ - θʳ) / Δ, ϵ, one(FT) - ϵ)
    𝒮² = clamp((θ² - θʳ) / Δ, ϵ, one(FT) - ϵ)
    logψ = log(ψ¹ / ψ²)

    # Bisect on the midpoint alone: the bracket ends overflow `𝒮^(-1/m)` as m → 0, so an
    # f(lo)·f(mid) test would compare against a non-finite value.
    lo = convert(FT, 101//100)
    hi = convert(FT, 12)
    for _ in 1:40
        n   = (lo + hi) / 2
        m⁻¹ = 1 / van_genuchten_m(n)
        f   = log((𝒮¹^(-m⁻¹) - one(FT)) / (𝒮²^(-m⁻¹) - one(FT))) - n * logψ
        hi  = ifelse(f > 0, n, hi)
        lo  = ifelse(f > 0, lo, n)
    end

    n   = (lo + hi) / 2
    m⁻¹ = 1 / van_genuchten_m(n)
    α   = (𝒮¹^(-m⁻¹) - one(FT))^(1/n) / ψ¹

    # A NaN input makes every `f` NaN, and `NaN > 0` is false, so the loop above walks `lo`
    # to the bracket and returns n = 12 — a plausible number in place of missing data.
    missing_data = isnan(𝒮¹) | isnan(𝒮²)
    return ifelse(missing_data, convert(FT, NaN), α),
           ifelse(missing_data, convert(FT, NaN), n)
end

@kernel function _soil_hydraulic_properties!(porosity, residual, α, n, K₀, ℓ,
                                            sand, silt, clay, bulk_density,
                                            w, depths, W, Nz, ψ¹, ψ², ptf)
    i, j = @index(Global, NTuple)
    FT = eltype(porosity)

    Σν = zero(FT); Σθʳ = zero(FT); Σθ¹ = zero(FT); Σθ² = zero(FT)
    Σℓ = zero(FT); Σw_over_K = zero(FT)

    @inbounds for k in 1:Nz
        wk = w[k]
        p  = soil_hydraulic_parameters(ptf, sand[i, j, k], silt[i, j, k],
                                       clay[i, j, k], bulk_density[i, j, k], depths[k])
        νk  = p.porosity
        θʳk = p.residual_liquid_fraction
        Δk  = νk - θʳk
        nk  = p.pore_size_uniformity
        θ¹  = θʳk + Δk * van_genuchten_saturation(p.inverse_air_entry_head * ψ¹, nk)
        θ²  = θʳk + Δk * van_genuchten_saturation(p.inverse_air_entry_head * ψ², nk)
        # `0 * NaN` is NaN, so zero weight alone would not keep a missing-data layer below
        # `slab_depth` out of the column.
        inside = wk > 0
        Σν        += ifelse(inside, wk * νk, zero(FT))
        Σθʳ       += ifelse(inside, wk * θʳk, zero(FT))
        Σθ¹       += ifelse(inside, wk * θ¹, zero(FT))
        Σθ²       += ifelse(inside, wk * θ², zero(FT))
        Σℓ        += ifelse(inside, wk * p.pore_connectivity_exponent, zero(FT))
        Σw_over_K += ifelse(inside, wk / p.matching_point_conductivity, zero(FT))
    end

    νᶜ  = Σν / W
    θʳᶜ = Σθʳ / W
    αᶜ, nᶜ = matched_retention_parameters(Σθ¹ / W, Σθ² / W, θʳᶜ, νᶜ, ψ¹, ψ²)

    @inbounds begin
        porosity[i, j, 1]    = νᶜ
        residual[i, j, 1]    = θʳᶜ
        α[i, j, 1]           = αᶜ
        n[i, j, 1]           = nᶜ
        K₀[i, j, 1]          = W / Σw_over_K                  # harmonic
        ℓ[i, j, 1]           = Σℓ / W
    end
end

"""
    soil_hydraulic_properties(sand, silt, clay, bulk_density;
                              slab_depth, z_interfaces,
                              ptf = WeynantsPedotransfer(),
                              matching_heads = (1, 150))

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
harmonic mean, since layers in series add resistance, and `ℓ` is an arithmetic one.
See [`layer_weights`](@ref).

The default heads are field capacity and the permanent wilting point, which bracket the
range the slab operates in. 1 m rather than the textbook 3.3 m follows
[Balsamo et al. (2009)](@cite balsamo2009), who measured `-0.10` bar as the better field
capacity for van Genuchten parameters of this family; averaged over contrasting columns
the choice is worth little, hence a keyword rather than a constant.

The parameters describe the soil inside `[-slab_depth, 0]` and nothing below it. Soil
below the slab belongs to the deep-flux closure.

Rock, water and out-of-coverage cells arrive as `NaN` and stay `NaN` in every output the
data feeds. Parameters a pedotransfer function holds constant never read the data and so
come back as that constant — `θʳ` for [`WeynantsPedotransfer`](@ref), `θʳ` and `ℓ` for
[`HYPRESPedotransfer`](@ref) — so mask on a predicted field.

`matching_point_conductivity` inherits whatever `ptf` means by it, which for
[`WeynantsPedotransfer`](@ref) is a matrix matching point rather than the value an
infiltration cap wants (see [`saturated_conductivity`](@ref)).

Each output is a `Field{Center, Center, Nothing}` on the inputs' grid, read by the slab at
`[i, j]`. `slab_depth` must be a scalar; `z_interfaces` are the dataset layer faces (e.g.
`DataWrangling.z_interfaces(OpenLandMapSoilDB())`).
"""
function soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                   slab_depth, z_interfaces,
                                   ptf = WeynantsPedotransfer(),
                                   matching_heads = (1, 150))
    grid = sand.grid
    arch = architecture(grid)
    FT   = eltype(sand)
    Nz   = size(sand, 3)

    length(z_interfaces) == Nz + 1 ||
        throw(ArgumentError("z_interfaces must have length size(sand, 3) + 1 = $(Nz + 1); " *
                            "got $(length(z_interfaces))"))

    ψ¹, ψ² = matching_heads
    0 < ψ¹ < ψ² ||
        throw(ArgumentError("matching_heads must be two increasing positive suction " *
                            "heads (m); got $matching_heads"))

    weights = layer_weights(z_interfaces, slab_depth)
    W = sum(weights)
    W > 0 ||
        throw(ArgumentError("slab_depth = $slab_depth does not overlap the soil column " *
                            "spanned by z_interfaces = $z_interfaces"))

    w      = on_architecture(arch, convert.(FT, weights))
    depths = on_architecture(arch, convert.(FT, layer_depths(z_interfaces)))

    porosity = Field{Center, Center, Nothing}(grid)
    residual = Field{Center, Center, Nothing}(grid)
    α        = Field{Center, Center, Nothing}(grid)
    n        = Field{Center, Center, Nothing}(grid)
    K₀       = Field{Center, Center, Nothing}(grid)
    ℓ        = Field{Center, Center, Nothing}(grid)

    launch!(arch, grid, :xy, _soil_hydraulic_properties!,
            porosity, residual, α, n, K₀, ℓ,
            sand, silt, clay, bulk_density,
            w, depths, convert(FT, W), Nz, convert(FT, ψ¹), convert(FT, ψ²),
            convert_eltype(FT, ptf))

    return (porosity = porosity,
            residual_liquid_fraction = residual,
            inverse_air_entry_head = α,
            pore_size_uniformity = n,
            matching_point_conductivity = K₀,
            pore_connectivity_exponent = ℓ)
end
