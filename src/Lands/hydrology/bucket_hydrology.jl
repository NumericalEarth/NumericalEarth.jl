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
    BucketHydrology(FT = Oceananigans.defaults.FloatType;
                    maximum_water_storage = 150)

Single-bucket land hydrology. The land water mass per area `Mˡᵃ`
(`water_storage`) evolves under `P − E`, clamped to `[0, Mˡᵃ⁺]` where `Mˡᵃ⁺`
is the bucket capacity (`maximum_water_storage`, the soil-science "field
capacity"; 150 kg m⁻² ≈ 15 cm of equivalent liquid water).

The atmosphere-facing `surface_saturation` is the continuous saturation
`𝒮 = clamp(Mˡᵃ/Mˡᵃ⁺, 0, 1)`. The interface surface-humidity models derive their
own availability from it: [`BulkHumidity`](@ref) is the binary limit (saturated
where `𝒮 > 0`), [`FractionalHumidity`](@ref) scales saturation by `β(𝒮)`, and
[`SkinHumidity`](@ref) solves a vapor-flux balance. The availability
parameterization belongs on the interface, not here.

`maximum_water_storage` may be a scalar or a per-cell `Field`.
"""
struct BucketHydrology{Mmax} <: AbstractHydrology
    maximum_water_storage :: Mmax
end

function BucketHydrology(FT::Type = Oceananigans.defaults.FloatType;
                         maximum_water_storage = 150.0)
    maximum_water_storage = normalize_property(FT, maximum_water_storage)
    return BucketHydrology(maximum_water_storage)
end

# The container always allocates `water_storage` (prognostic) and
# `saturation` (diagnostic, recomputed in `update_diagnostics!`).
flux_variables(::BucketHydrology) = (:precipitation, :evaporation)

@inline function _bucket_capacity(maximum_water_storage, i, j, k=1)
    M_max = property_value(maximum_water_storage, i, j, k)
    return max(M_max, 0)
end

@kernel function _bucket_hydrology_step!(M, P, E, Δt, maximum_water_storage)
    i, j = @index(Global, NTuple)
    @inbounds begin
        M_max = _bucket_capacity(maximum_water_storage, i, j, 1)
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
            Δt, b.maximum_water_storage)
    return nothing
end

# Continuous surface saturation 𝒮 = M / M_max ∈ [0, 1] (`𝒮` follows Breeze's
# saturation symbol). The interface humidity models (`BulkHumidity`,
# `FractionalHumidity`, …) derive their own availability from it; `BulkHumidity`'s
# wet/dry test (`𝒮 > 0`) is the binary special case.
@kernel function _bucket_hydrology_saturation!(saturation, M, maximum_water_storage)
    i, j = @index(Global, NTuple)
    @inbounds begin
        M_max = _bucket_capacity(maximum_water_storage, i, j, 1)
        𝒮 = ifelse(M_max > 0, clamp(M[i, j, 1] / M_max, 0, 1), zero(eltype(M)))
        saturation[i, j, 1] = 𝒮
    end
end

function update_diagnostics!(b::BucketHydrology, land)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _bucket_hydrology_saturation!,
            land.saturation, land.water_storage,
            b.maximum_water_storage)
    return nothing
end

saturation(::BucketHydrology, land) = land.saturation

Base.summary(b::BucketHydrology) =
    string("BucketHydrology(maximum_water_storage=", prettysummary(b.maximum_water_storage), ")")
