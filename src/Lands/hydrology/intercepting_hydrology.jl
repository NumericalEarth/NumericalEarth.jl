#####
##### `InterceptingHydrology` — a canopy water store `Mᶜ` between the rain and a soil
##### hydrology: intercepted rain fills it, the interface's wet-canopy evaporation drains
##### it, over-capacity drip and the uncaught rain reach the soil as throughfall.
#####

"""
$(TYPEDSIGNATURES)

Canopy water store `Mᶜ` (kg m⁻²) wrapping a [`VariablySaturatedHydrology`](@ref)
`soil`. Rain is caught by the Beer–Lambert canopy fraction `1 − e^{−KΩ·LAI}` up to the
capacity `c·LAI`; the wet canopy evaporates at the rate `Jʷᵉᵗ` the interface computes
(a [`CanopyGroundSurface`](@ref)), and the uncaught rain plus the
over-capacity drip reach the soil as throughfall.

`leaf_area_index` and `cover_fraction` default to the land surface's canopy LAI and
vegetated fraction when the land is constructed. A `FieldTimeSeries` LAI or cover
fraction runs on CPU only. The store is kept per unit vegetated area. Changes in cover
conserve `cover_fraction * Mᶜ`, with water exceeding the new canopy capacity reaching
the soil as throughfall. Interception and evaporation use the cover at the start of the step.

* `capacity_per_leaf_area` — `c` (kg m⁻² per unit LAI).
* `extinction`, `clumping` — Beer–Lambert `K`, `Ω` of the caught fraction.
* `cover_fraction` — vegetated fraction of the cell (a `Number`, `Field`, or `FieldTimeSeries`).
"""
struct InterceptingHydrology{S, L, C, FT} <: AbstractHydrology
    soil                     :: S
    leaf_area_index          :: L
    cover_fraction           :: C
    capacity_per_leaf_area   :: FT
    extinction               :: FT
    clumping                 :: FT
end

function InterceptingHydrology(FT::Type = Oceananigans.defaults.FloatType;
                               soil,
                               leaf_area_index = nothing,
                               capacity_per_leaf_area = 0.1,
                               extinction = 0.5,
                               clumping = 1,
                               cover_fraction = nothing)
    return InterceptingHydrology(soil,
                                 convert_if_number(FT, leaf_area_index),
                                 convert_if_number(FT, cover_fraction),
                                 convert(FT, capacity_per_leaf_area),
                                 convert(FT, extinction),
                                 convert(FT, clumping))
end

Adapt.adapt_structure(to, h::InterceptingHydrology) =
    InterceptingHydrology(Adapt.adapt(to, h.soil),
                          Adapt.adapt(to, h.leaf_area_index),
                          Adapt.adapt(to, h.cover_fraction),
                          h.capacity_per_leaf_area,
                          h.extinction,
                          h.clumping)

prognostic_variables(h::InterceptingHydrology) =
    merge_unique(prognostic_variables(h.soil), (:canopy_water_storage,))

flux_variables(h::InterceptingHydrology) =
    merge_unique(flux_variables(h.soil), (:liquid_precipitation_flux, :canopy_evaporation))

diagnostic_variables(h::InterceptingHydrology) =
    merge_unique(diagnostic_variables(h.soil), (:throughfall, :canopy_water_storage_tendency, :previous_cover_fraction))

initial_flux(h::InterceptingHydrology, name::Symbol, grid) = initial_flux(h.soil, name, grid)
initial_diagnostic(h::InterceptingHydrology, name::Symbol, grid) = initial_diagnostic(h.soil, name, grid)

#####
##### Interception step, ahead of the soil step that reads its throughfall.
#####

# Mᶜ and Mᶜⁿ⁺¹ are per unit vegetated area; uncapped_storage, drip, and throughfall are per
# unit cell area.
# rain = (fⁿ⁺¹ Mᶜⁿ⁺¹ − fⁿ Mᶜ)/Δt + fⁿ Jʷᵉᵗ + throughfall exactly: evaporation beyond what the
# store held is taken from the throughfall.
@inline function canopy_store_update(Mᶜ, rain, Jʷᵉᵗ, Mᶜ⁺, fⁱⁿᵗ, fⁿ, fⁿ⁺¹, Δt)
    Pⁱⁿᵗ             = fⁱⁿᵗ * rain
    uncapped_storage = fⁿ * (Mᶜ + Δt * (Pⁱⁿᵗ - Jʷᵉᵗ))
    drip             = max(uncapped_storage - fⁿ⁺¹ * Mᶜ⁺, 0) / Δt
    shortfall        = max(-uncapped_storage, 0) / Δt
    Mᶜⁿ⁺¹            = clamp(uncapped_storage, 0, fⁿ⁺¹ * Mᶜ⁺) / ifelse(fⁿ⁺¹ > 0, fⁿ⁺¹, one(fⁿ⁺¹))
    return Mᶜⁿ⁺¹, drip, rain - fⁿ * Pⁱⁿᵗ + drip - shortfall
end

function prepare_time_step!(h::InterceptingHydrology, land)
    time = land.clock.time
    update_field_time_series!(h.cover_fraction, Time(time))
    launch!(architecture(land.grid), land.grid, :xy, _cache_cover_fraction!,
            land.diagnostics.previous_cover_fraction, h.cover_fraction, land.grid, time)
    return nothing
end

@kernel function _cache_cover_fraction!(previous_cover_fraction, cover_fraction, grid, time)
    i, j = @index(Global, NTuple)
    f = stateindex(cover_fraction, i, j, 1, grid, Time(time), (Center, Center, Center))
    @inbounds previous_cover_fraction[i, j, 1] = clamp(f, 0, 1)
end

@kernel function _interception_step!(canopy_water_storage, liquid_precipitation_flux, canopy_evaporation,
                                     throughfall, canopy_water_storage_tendency, previous_cover_fraction,
                                     h, Δt, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Mᶜⁿ  = canopy_water_storage[i, j, 1]
        rain = liquid_precipitation_flux[i, j, 1]
        Jʷᵉᵗ = canopy_evaporation[i, j, 1]
        fⁿ   = previous_cover_fraction[i, j, 1]
    end
    FT   = typeof(Mᶜⁿ)
    LAI  = convert(FT, stateindex(h.leaf_area_index, i, j, 1, grid, Time(time), (Center, Center, Center)))
    fⁿ⁺¹ = clamp(convert(FT, stateindex(h.cover_fraction, i, j, 1, grid, Time(time), (Center, Center, Center))), 0, 1)
    Mᶜ⁺  = h.capacity_per_leaf_area * LAI
    fⁱⁿᵗ = 1 - canopy_transmittance(h.extinction, h.clumping, LAI)

    Mᶜⁿ⁺¹, _, Pˡ = canopy_store_update(Mᶜⁿ, rain, Jʷᵉᵗ, Mᶜ⁺, fⁱⁿᵗ, fⁿ, fⁿ⁺¹, Δt)

    @inbounds begin
        canopy_water_storage[i, j, 1]          = Mᶜⁿ⁺¹
        throughfall[i, j, 1]                   = Pˡ
        canopy_water_storage_tendency[i, j, 1] = (Mᶜⁿ⁺¹ - Mᶜⁿ) / Δt
    end
end

function time_step!(h::InterceptingHydrology, land, Δt, time)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _interception_step!,
            land.prognostic.canopy_water_storage,
            land.fluxes.liquid_precipitation_flux,
            land.fluxes.canopy_evaporation,
            land.diagnostics.throughfall,
            land.diagnostics.canopy_water_storage_tendency,
            land.diagnostics.previous_cover_fraction,
            h, Δt, land.grid, time)
    time_step!(h.soil, land, Δt, time, land.diagnostics.throughfall)
    return nothing
end

update_diagnostics!(h::InterceptingHydrology, land) = update_diagnostics!(h.soil, land)

saturation(h::InterceptingHydrology, land) = saturation(h.soil, land)

# The store capacity `Mᶜ⁺ = c·LAI` the interface normalizes the wet fraction by.
canopy_water_capacity(h::InterceptingHydrology, land) =
    KernelFunctionOperation{Center, Center, Nothing}(canopy_water_capacity, land.grid, h, land.clock)

@inline canopy_water_capacity(i, j, k, grid, h, clock) =
    h.capacity_per_leaf_area * stateindex(h.leaf_area_index, i, j, 1, grid, Time(clock.time), (Center, Center, Center))

Base.summary(h::InterceptingHydrology) =
    string("InterceptingHydrology(soil=", summary(h.soil),
           ", c=", prettysummary(h.capacity_per_leaf_area), ")")
