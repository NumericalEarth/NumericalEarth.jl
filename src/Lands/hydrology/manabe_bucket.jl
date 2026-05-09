#####
##### `ManabeBucket` — Manabe (1969) single-bucket soil moisture.
#####
##### One prognostic variable `state.W`, two flux accumulators
##### `fluxes.precipitation` and `fluxes.evaporation` (kg m⁻² s⁻¹).
##### `wetness` returns `β = min(W / (W_crit · W_max), 1)` lazily as a
##### `KernelFunctionOperation`-style construct; here we materialise
##### into a Field cached on the closure for the atmosphere to read.
#####

"""
    ManabeBucket(FT = Float64; field_capacity = 150, critical_wetness = 0.75)

Manabe (1969) bucket. `field_capacity` is `W_max` in `kg m⁻²` (≈ 15 cm
of liquid water). `critical_wetness` is `W_crit / W_max` — the
fractional wetness above which `β` saturates to one.
"""
struct ManabeBucket{FT, B} <: AbstractHydrology
    field_capacity   :: FT
    critical_wetness :: FT
    β                :: B   # cached wetness Field; allocated by initial_state
end

function ManabeBucket(FT::Type = Float64; field_capacity = 150.0, critical_wetness = 0.75)
    return ManabeBucket{FT, Nothing}(convert(FT, field_capacity),
                                     convert(FT, critical_wetness),
                                     nothing)
end

prognostic_variables(::ManabeBucket) = (:W, :β)
flux_variables(::ManabeBucket)       = (:precipitation, :evaporation)

initial_state(::ManabeBucket{FT}, ::Symbol, grid) where FT = CenterField(grid)

@kernel function _manabe_bucket_step!(W, β, P, E, Δt, W_max, W_crit)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(W)
        Wnew = W[i, j, 1] + (P[i, j, 1] - E[i, j, 1]) * Δt
        # Saturation cap; overflow would be runoff (not tracked here).
        Wnew = clamp(Wnew, zero(FT), W_max)
        W[i, j, 1] = Wnew
        β[i, j, 1] = clamp(Wnew / (W_crit * W_max), zero(FT), one(FT))
    end
end

function step!(b::ManabeBucket, state, fluxes, surface, parameters, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _manabe_bucket_step!,
            state.W, state.β,
            fluxes.precipitation, fluxes.evaporation,
            Δt, b.field_capacity, b.critical_wetness)
    return nothing
end

wetness(::ManabeBucket, state, parameters) = state.β

Base.summary(b::ManabeBucket{FT}) where FT =
    string("ManabeBucket{$FT}(W_max=", b.field_capacity, ", W_crit/W_max=", b.critical_wetness, ")")
