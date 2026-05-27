#####
##### `BucketHydrology` — single-bucket land water budget.
#####
##### One prognostic variable `water_storage` (Mˡᵃ, kg m⁻²) evolving under
##### `P − E` (the flux accumulators `fluxes.precipitation`,
##### `fluxes.evaporation`, kg m⁻² s⁻¹). The atmosphere-facing
##### `surface_saturation` is the continuous diagnostic `saturation`
##### `𝒮 = Mˡᵃ/Mˡᵃ⁺ ∈ [0, 1]`; the interface humidity models derive their
##### availability from it.
#####

"""
    BucketHydrology(FT = Float64;
                    maximum_water_storage = 150,
                    root_depth = 1)

Single-bucket land hydrology. The land water mass per area `Mˡᵃ`
(`water_storage`) evolves under `P − E`, clamped to `[0, Mˡᵃ⁺]` where `Mˡᵃ⁺`
is the bucket capacity (`maximum_water_storage`, the soil-science "field
capacity"; 150 kg m⁻² ≈ 15 cm of equivalent liquid water, scaled by
`root_depth`).

The atmosphere-facing `surface_saturation` is the continuous saturation
`𝒮 = clamp(Mˡᵃ/Mˡᵃ⁺, 0, 1)`. The interface surface-humidity models derive their
own availability from it: [`BulkHumidity`](@ref) is the binary limit (saturated
where `𝒮 > 0`), [`FractionalHumidity`](@ref) scales saturation by `β(𝒮)`, and
[`SkinHumidity`](@ref) solves a vapor-flux balance. The availability
parameterization belongs on the interface, not here.

`maximum_water_storage` and `root_depth` may be scalars or per-cell `Field`s.
"""
struct BucketHydrology{Mmax, R} <: AbstractHydrology
    maximum_water_storage :: Mmax
    root_depth            :: R
end

function BucketHydrology(FT::Type = Float64;
                         maximum_water_storage = 150.0,
                         root_depth            = 1)
    maximum_water_storage = normalize_property(FT, maximum_water_storage)
    root_depth            = normalize_property(FT, root_depth)
    return BucketHydrology(maximum_water_storage, root_depth)
end

# The container always allocates `water_storage` (prognostic) and
# `saturation` (diagnostic, recomputed in `update_diagnostics!`).
flux_variables(::BucketHydrology) = (:precipitation, :evaporation)

@inline function _bucket_root_zone_capacity(maximum_water_storage, root_depth, i, j, k=1)
    M_max = property_value(maximum_water_storage, i, j, k)
    zʳ    = property_value(root_depth, i, j, k)
    return max(M_max * max(zʳ, 0), 0)
end

@kernel function _bucket_hydrology_step!(M, P, E, Δt, maximum_water_storage, root_depth)
    i, j = @index(Global, NTuple)
    @inbounds begin
        M_max = _bucket_root_zone_capacity(maximum_water_storage, root_depth, i, j, 1)
        # Saturation cap; excess water above M_max is shed (runoff diagnostic
        # to be reintroduced when downstream coupling needs it).
        M[i, j, 1] = clamp(M[i, j, 1] + (P[i, j, 1] - E[i, j, 1]) * Δt, 0, M_max)
    end
end

function step!(b::BucketHydrology, land, Δt)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _bucket_hydrology_step!,
            land.water_storage,
            land.fluxes.precipitation, land.fluxes.evaporation,
            Δt, b.maximum_water_storage, b.root_depth)
    return nothing
end

# Continuous surface saturation 𝒮 = M / M_max ∈ [0, 1] (`𝒮` follows Breeze's
# saturation symbol). The interface humidity models (`BulkHumidity`,
# `FractionalHumidity`, …) derive their own availability from it; `BulkHumidity`'s
# wet/dry test (`𝒮 > 0`) is the binary special case.
@kernel function _bucket_hydrology_saturation!(saturation, M, maximum_water_storage, root_depth)
    i, j = @index(Global, NTuple)
    @inbounds begin
        M_max = _bucket_root_zone_capacity(maximum_water_storage, root_depth, i, j, 1)
        𝒮 = ifelse(M_max > 0, clamp(M[i, j, 1] / M_max, 0, 1), zero(eltype(M)))
        saturation[i, j, 1] = 𝒮
    end
end

function update_diagnostics!(b::BucketHydrology, land)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _bucket_hydrology_saturation!,
            land.saturation, land.water_storage,
            b.maximum_water_storage, b.root_depth)
    return nothing
end

saturation(::BucketHydrology, land) = land.saturation

Base.summary(b::BucketHydrology) =
    string("BucketHydrology(maximum_water_storage=", prettysummary(b.maximum_water_storage),
           ", root_depth=", prettysummary(b.root_depth), ")")
