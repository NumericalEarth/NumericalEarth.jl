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
    BucketHydrology(FT = Float64; field_capacity = 150, critical_wetness = 0.75)

Manabe (1969) bucket. `field_capacity` is the maximum areal liquid-water
storage in `kg m⁻²` (≈ 15 cm of liquid water when `field_capacity = 150`).
`critical_wetness` is the threshold wetness ratio above which
`moisture_availability` saturates to one. Both may be scalars or per-cell
`Field`s.
"""
struct BucketHydrology{C, W} <: AbstractHydrology
    field_capacity   :: C
    critical_wetness :: W
end

function BucketHydrology(FT::Type = Float64; field_capacity = 150.0, critical_wetness = 0.75)
    field_capacity   = normalize_property(FT, field_capacity)
    critical_wetness = normalize_property(FT, critical_wetness)
    return BucketHydrology(field_capacity, critical_wetness)
end

# `:W` is the only true prognostic; `:moisture_availability` is a diagnostic
# of `W` recomputed in `update_diagnostics!`. Both share the `state` namedtuple
# so the container allocates a field for each.
prognostic_variables(::BucketHydrology) = (:W, :moisture_availability)
flux_variables(::BucketHydrology)       = (:precipitation, :evaporation, :runoff)

@kernel function _bucket_hydrology_step!(W, runoff, P, E, Δt, field_capacity)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(W)
        W_cap = max(property_value(field_capacity, i, j, 1), zero(FT))
        Wnew = W[i, j, 1] + (P[i, j, 1] - E[i, j, 1]) * Δt
        # Saturation cap; excess liquid water leaves as runoff.
        R = max(Wnew - W_cap, zero(FT)) / Δt
        Wnew = clamp(Wnew, zero(FT), W_cap)

        runoff[i, j, 1] = R
        W[i, j, 1] = Wnew
    end
end

function step!(b::BucketHydrology, state, fluxes, surface, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _bucket_hydrology_step!,
            state.W, fluxes.runoff,
            fluxes.precipitation, fluxes.evaporation,
            Δt, b.field_capacity)
    return nothing
end

@kernel function _bucket_hydrology_moisture_availability!(moisture_availability, W, field_capacity, critical_wetness)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(W)
        W_cap = max(property_value(field_capacity, i, j, 1), zero(FT))
        w_crit = property_value(critical_wetness, i, j, 1)
        Wnow = max(W[i, j, 1], zero(FT))

        # W_cap ≤ 0 means no storage capacity (e.g. inactive tile) ⇒ no moisture.
        # w_crit ≤ 0 is the Manabe degenerate limit (everything above zero W
        # behaves like a saturated surface).
        β = ifelse(W_cap <= zero(FT),  zero(FT),
            ifelse(w_crit <= zero(FT), one(FT),
                   clamp(Wnow / (w_crit * W_cap), zero(FT), one(FT))))

        moisture_availability[i, j, 1] = β
    end
end

function update_diagnostics!(b::BucketHydrology, state, fluxes, surface, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _bucket_hydrology_moisture_availability!,
            state.moisture_availability, state.W,
            b.field_capacity, b.critical_wetness)
    return nothing
end

wetness(::BucketHydrology, state) = state.moisture_availability

Base.summary(b::BucketHydrology) =
    string("BucketHydrology(field_capacity=", prettysummary(b.field_capacity),
           ", critical_wetness=", prettysummary(b.critical_wetness), ")")
