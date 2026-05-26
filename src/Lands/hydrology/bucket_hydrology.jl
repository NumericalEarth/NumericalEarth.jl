#####
##### `BucketHydrology` — Manabe-style single-bucket soil moisture.
#####
##### One prognostic variable `state.water_storage` (Mˡᵃ, kg m⁻²) and
##### the flux accumulators `fluxes.precipitation`, `fluxes.evaporation`
##### (kg m⁻² s⁻¹). The atmosphere-facing moisture availability `β` is
##### a diagnostic of `water_storage` provided through
##### `wetness(::BucketHydrology, state)`.
#####

"""
    BucketHydrology(FT = Float64;
                    maximum_water_storage = 150,
                    critical_wetness_ratio = 0.75,
                    root_depth = 1,
                    leaf_area_index = 0,
                    lai_stress = 0)

Manabe-style single-bucket hydrology. The land water mass per area `Mˡᵃ`
evolves under `P − E`, clamped to `[0, Mˡᵃ⁺]` where `Mˡᵃ⁺` is the bucket
capacity (`maximum_water_storage`, the soil-science "field capacity"; 150 kg m⁻²
corresponds to roughly 15 cm of equivalent liquid water).

The moisture availability factor `β` plateaus at 1 above the critical wetness
threshold `εʷ · Mˡᵃ⁺`, where `εʷ = critical_wetness_ratio`. Below the
threshold, `β` rises linearly with `Mˡᵃ`.

Both `maximum_water_storage` and `critical_wetness_ratio` may be scalars or
per-cell `Field`s. `root_depth` and `leaf_area_index` support per-cell fields
and introduce simple literature-inspired modifiers:

- effective capacity scales with `maximum_water_storage * root_depth`
- canopy stress scales moisture availability by `exp(-lai_stress * LAI)`

The defaults (`root_depth = 1`, `leaf_area_index = 0`, `lai_stress = 0`) recover
the classic Manabe bucket.
"""
struct BucketHydrology{Mmax, Ew, R, L, S} <: AbstractHydrology
    maximum_water_storage  :: Mmax
    critical_wetness_ratio :: Ew
    root_depth             :: R
    leaf_area_index        :: L
    lai_stress             :: S
end

function BucketHydrology(FT::Type = Float64;
                         maximum_water_storage  = 150.0,
                         critical_wetness_ratio = 0.75,
                         root_depth             = 1,
                         leaf_area_index        = 0,
                         lai_stress             = 0.0)
    maximum_water_storage  = normalize_property(FT, maximum_water_storage)
    critical_wetness_ratio = normalize_property(FT, critical_wetness_ratio)
    root_depth             = normalize_property(FT, root_depth)
    leaf_area_index        = normalize_property(FT, leaf_area_index)
    lai_stress             = normalize_property(FT, lai_stress)
    return BucketHydrology(maximum_water_storage, critical_wetness_ratio,
                           root_depth, leaf_area_index, lai_stress)
end

# `:water_storage` is the prognostic; `:moisture_availability` is a diagnostic
# recomputed in `update_diagnostics!`. Both share the `state` namedtuple
# so the container allocates a field for each.
prognostic_variables(::BucketHydrology) = (:water_storage, :moisture_availability)
flux_variables(::BucketHydrology)       = (:precipitation, :evaporation)

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

function step!(b::BucketHydrology, state, fluxes, surface, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _bucket_hydrology_step!,
            state.water_storage,
            fluxes.precipitation, fluxes.evaporation,
            Δt, b.maximum_water_storage, b.root_depth)
    return nothing
end

@kernel function _bucket_hydrology_moisture_availability!(moisture_availability, M,
                                                          maximum_water_storage,
                                                          critical_wetness_ratio,
                                                          root_depth,
                                                          leaf_area_index,
                                                          lai_stress)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT    = eltype(M)
        M_max = _bucket_root_zone_capacity(maximum_water_storage, root_depth, i, j, 1)
        εʷ    = property_value(critical_wetness_ratio, i, j, 1)
        Mnow  = max(M[i, j, 1], 0)

        # M_max ≤ 0 means no storage capacity (e.g. inactive tile) ⇒ no moisture.
        # εʷ ≤ 0 is the degenerate limit (everything above zero M behaves like
        # a saturated surface).
        β = ifelse(M_max <= 0, 0,
            ifelse(εʷ <= 0, one(FT),
                   clamp(Mnow / (εʷ * M_max), 0, one(FT))))

        LAI = property_value(leaf_area_index, i, j, 1)
        kᴸ  = property_value(lai_stress, i, j, 1)
        β *= exp(-kᴸ * LAI)
        β  = clamp(β, 0, one(FT))

        moisture_availability[i, j, 1] = β
    end
end

function update_diagnostics!(b::BucketHydrology, state, fluxes, surface, grid)
    arch = architecture(grid)

    launch!(arch, grid, :xy, _bucket_hydrology_moisture_availability!,
            state.moisture_availability, state.water_storage,
            b.maximum_water_storage, b.critical_wetness_ratio,
            b.root_depth, b.leaf_area_index, b.lai_stress)

    return nothing
end

wetness(::BucketHydrology, state) = state.moisture_availability

Base.summary(b::BucketHydrology) =
    string("BucketHydrology(maximum_water_storage=", prettysummary(b.maximum_water_storage),
           ", critical_wetness_ratio=", prettysummary(b.critical_wetness_ratio),
           ", root_depth=", prettysummary(b.root_depth),
           ", leaf_area_index=", prettysummary(b.leaf_area_index),
           ", lai_stress=", prettysummary(b.lai_stress), ")")
