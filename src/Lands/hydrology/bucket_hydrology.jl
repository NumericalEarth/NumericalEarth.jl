#####
##### `BucketHydrology` — Manabe (1969) single-bucket soil moisture.
#####
##### One prognostic variable `state.W` (land water storage, kg m⁻²),
##### one diagnostic `diagnostics.moisture_availability` (β), and three
##### flux accumulators `fluxes.precipitation`, `fluxes.evaporation`,
##### and `fluxes.runoff` (all in kg m⁻² s⁻¹). `wetness` returns the
##### cached `moisture_availability` field, which the atmosphere--land
##### specific humidity closure reads.
#####

"""
    BucketHydrology(FT = Float64;
                   field_capacity = 150,
                   critical_wetness = 0.75,
                   root_depth = 1,
                   leaf_area_index = 0,
                   lai_stress = 0)

Manabe (1969) bucket. `field_capacity` is the maximum areal liquid-water
storage in `kg m⁻²` (≈ 15 cm of liquid water when `field_capacity = 150`).
`critical_wetness` is the threshold wetness ratio above which
`moisture_availability` saturates to one. Both may be scalars or per-cell
`Field`s. `root_depth` and `leaf_area_index` support per-cell fields and
introduce simple literature-inspired modifiers:

- effective field capacity scales with `field_capacity * root_depth`
- canopy stress scales moisture availability by `exp(-lai_stress * LAI)`

The defaults (`root_depth = 1`, `leaf_area_index = 0`, `lai_stress = 0`) recover
the classic Manabe bucket.
"""
struct BucketHydrology{C, W, R, L, S} <: AbstractHydrology
    field_capacity   :: C
    critical_wetness :: W
    root_depth       :: R
    leaf_area_index  :: L
    lai_stress       :: S
end

function BucketHydrology(FT::Type = Float64;
                        field_capacity = 150.0,
                        critical_wetness = 0.75,
                        root_depth = 1,
                        leaf_area_index = 0,
                        lai_stress = 0.0)
    field_capacity   = normalize_property(FT, field_capacity)
    critical_wetness = normalize_property(FT, critical_wetness)
    root_depth       = normalize_property(FT, root_depth)
    leaf_area_index  = normalize_property(FT, leaf_area_index)
    lai_stress       = normalize_property(FT, lai_stress)
    return BucketHydrology(field_capacity, critical_wetness, root_depth, leaf_area_index, lai_stress)
end

# `:W` is the only true prognostic; `:moisture_availability` is a diagnostic
# of `W` recomputed in `update_diagnostics!`. Both share the `state` namedtuple
# so the container allocates a field for each.
prognostic_variables(::BucketHydrology) = (:W, :moisture_availability)
flux_variables(::BucketHydrology)       = (:precipitation, :evaporation, :runoff)

@inline function _bucket_root_zone_capacity(field_capacity, root_depth, i, j, k=1)
    W_cap = property_value(field_capacity, i, j, k)
    zʳ   = property_value(root_depth, i, j, k)
    return max(W_cap * max(zʳ, 0), 0)
end

@kernel function _bucket_hydrology_step!(W, runoff, P, E, Δt, field_capacity, root_depth)
    i, j = @index(Global, NTuple)
    @inbounds begin
        W_cap = _bucket_root_zone_capacity(field_capacity, root_depth, i, j, 1)
        Wnew = W[i, j, 1] + (P[i, j, 1] - E[i, j, 1]) * Δt
        # Saturation cap; excess liquid water leaves as runoff.
        R = max(Wnew - W_cap, 0) / Δt
        Wnew = clamp(Wnew, 0, W_cap)

        runoff[i, j, 1] = R
        W[i, j, 1] = Wnew
    end
end

function step!(b::BucketHydrology, state, fluxes, surface, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _bucket_hydrology_step!,
            state.W, fluxes.runoff,
            fluxes.precipitation, fluxes.evaporation,
            Δt, b.field_capacity, b.root_depth)
    return nothing
end

@kernel function _bucket_hydrology_moisture_availability!(moisture_availability, W,
                                                          field_capacity,
                                                          critical_wetness,
                                                          root_depth,
                                                          leaf_area_index,
                                                          lai_stress)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(W)
        W_cap = _bucket_root_zone_capacity(field_capacity, root_depth, i, j, 1)
        w_crit = property_value(critical_wetness, i, j, 1)
        Wnow = max(W[i, j, 1], 0)

        # W_cap ≤ 0 means no storage capacity (e.g. inactive tile) ⇒ no moisture.
        # w_crit ≤ 0 is the Manabe degenerate limit (everything above zero W
        # behaves like a saturated surface).
        β = ifelse(W_cap <= 0, 0,
            ifelse(w_crit <= 0, one(FT),
                   clamp(Wnow / (w_crit * W_cap), 0, one(FT))))

        LAI = property_value(leaf_area_index, i, j, 1)
        kᴸ = property_value(lai_stress, i, j, 1)
        β *= exp(-kᴸ * LAI)
        β = clamp(β, 0, one(FT))

        moisture_availability[i, j, 1] = β
    end
end

function update_diagnostics!(b::BucketHydrology, state, fluxes, surface, grid)
    arch = architecture(grid)

    launch!(arch, grid, :xy, _bucket_hydrology_moisture_availability!,
            state.moisture_availability, state.W,
            b.field_capacity, b.critical_wetness,
            b.root_depth, b.leaf_area_index, b.lai_stress)

    return nothing
end

wetness(::BucketHydrology, state) = state.moisture_availability

Base.summary(b::BucketHydrology) =
    string("BucketHydrology(field_capacity=", prettysummary(b.field_capacity),
           ", critical_wetness=", prettysummary(b.critical_wetness),
           ", root_depth=", prettysummary(b.root_depth),
           ", leaf_area_index=", prettysummary(b.leaf_area_index),
           ", lai_stress=", prettysummary(b.lai_stress), ")")
