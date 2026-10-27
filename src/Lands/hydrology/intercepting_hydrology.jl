#####
##### `InterceptingHydrology` — a canopy water store `Mᶜᵃⁿ` between the rain and a soil
##### hydrology: intercepted rain fills it, the interface's wet-canopy evaporation drains
##### it, over-capacity drip and the uncaught rain reach the soil as throughfall.
#####

"""
$(TYPEDSIGNATURES)

Canopy water store `Mᶜᵃⁿ` (kg m⁻²) wrapping a [`VariablySaturatedHydrology`](@ref)
`soil`. Rain is caught by the Beer–Lambert canopy fraction `1 − e^{−KΩ·LAI}` up to the
capacity `c·LAI`; the wet canopy evaporates at the rate `Jʷᵉᵗ` the interface computes
(a [`CanopyGroundSurface`](@ref)), and the uncaught rain plus the
over-capacity drip reach the soil as throughfall.

`leaf_area_index` supplies the canopy LAI; use the same value or field for the canopy interface.
A `FieldTimeSeries` LAI runs on CPU only.

* `capacity_per_leaf_area` — `c` (kg m⁻² per unit LAI).
* `extinction`, `clumping` — Beer–Lambert `K`, `Ω` of the caught fraction.
"""
struct InterceptingHydrology{S, L, FT} <: AbstractHydrology
    soil                     :: S
    leaf_area_index          :: L
    capacity_per_leaf_area   :: FT
    extinction               :: FT
    clumping                 :: FT
end

function InterceptingHydrology(FT::Type = Oceananigans.defaults.FloatType;
                               soil,
                               leaf_area_index,
                               capacity_per_leaf_area = 0.1,
                               extinction = 0.5,
                               clumping = 1)
    return InterceptingHydrology(soil,
                                 convert_if_number(FT, leaf_area_index),
                                 convert(FT, capacity_per_leaf_area),
                                 convert(FT, extinction),
                                 convert(FT, clumping))
end

Adapt.adapt_structure(to, h::InterceptingHydrology) =
    InterceptingHydrology(Adapt.adapt(to, h.soil),
                          Adapt.adapt(to, h.leaf_area_index),
                          h.capacity_per_leaf_area,
                          h.extinction,
                          h.clumping)

prognostic_variables(h::InterceptingHydrology) =
    merge_unique(prognostic_variables(h.soil), (:canopy_water_storage,))

flux_variables(h::InterceptingHydrology) =
    merge_unique(flux_variables(h.soil), (:liquid_precipitation_flux, :canopy_evaporation))

diagnostic_variables(h::InterceptingHydrology) =
    merge_unique(diagnostic_variables(h.soil), (:throughfall, :canopy_water_storage_tendency))

initial_flux(h::InterceptingHydrology, name::Symbol, grid) = initial_flux(h.soil, name, grid)
initial_diagnostic(h::InterceptingHydrology, name::Symbol, grid) = initial_diagnostic(h.soil, name, grid)

#####
##### Interception step, ahead of the soil step that reads its throughfall.
#####

# rain = (Mᶜᵃⁿⁿ⁺¹ − Mᶜᵃⁿ)/Δt + Jʷᵉᵗ + throughfall exactly: evaporation beyond what the
# store held is taken from the throughfall.
@inline function canopy_store_update(Mᶜᵃⁿ, rain, Jʷᵉᵗ, Mᶜᵃⁿ⁺, fⁱⁿᵗ, Δt)
    Pⁱⁿᵗ             = fⁱⁿᵗ * rain
    uncapped_storage = Mᶜᵃⁿ + Δt * (Pⁱⁿᵗ - Jʷᵉᵗ)
    drip             = max(uncapped_storage - Mᶜᵃⁿ⁺, 0) / Δt
    shortfall        = max(-uncapped_storage, 0) / Δt
    Mᶜᵃⁿⁿ⁺¹          = clamp(uncapped_storage, 0, Mᶜᵃⁿ⁺)
    return Mᶜᵃⁿⁿ⁺¹, drip, rain - Pⁱⁿᵗ + drip - shortfall
end

@kernel function _interception_step!(canopy_water_storage, liquid_precipitation_flux, canopy_evaporation,
                                     throughfall, canopy_water_storage_tendency,
                                     h, Δt, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Mᶜᵃⁿⁿ = canopy_water_storage[i, j, 1]
        rain  = liquid_precipitation_flux[i, j, 1]
        Jʷᵉᵗ  = canopy_evaporation[i, j, 1]
    end
    FT    = typeof(Mᶜᵃⁿⁿ)
    LAI   = convert(FT, stateindex(h.leaf_area_index, i, j, 1, grid, Time(time), (Center, Center, Center)))
    Mᶜᵃⁿ⁺ = h.capacity_per_leaf_area * LAI
    fⁱⁿᵗ  = 1 - canopy_transmittance(h.extinction, h.clumping, LAI)

    Mᶜᵃⁿⁿ⁺¹, _, Pˡ = canopy_store_update(Mᶜᵃⁿⁿ, rain, Jʷᵉᵗ, Mᶜᵃⁿ⁺, fⁱⁿᵗ, Δt)

    @inbounds begin
        canopy_water_storage[i, j, 1]          = Mᶜᵃⁿⁿ⁺¹
        throughfall[i, j, 1]                   = Pˡ
        canopy_water_storage_tendency[i, j, 1] = (Mᶜᵃⁿⁿ⁺¹ - Mᶜᵃⁿⁿ) / Δt
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
            h, Δt, land.grid, time)
    time_step!(h.soil, land, Δt, time, land.diagnostics.throughfall)
    return nothing
end

update_diagnostics!(h::InterceptingHydrology, land) = update_diagnostics!(h.soil, land)

saturation(h::InterceptingHydrology, land) = saturation(h.soil, land)

# The store capacity `Mᶜᵃⁿ⁺ = c·LAI` the interface normalizes the wet fraction by.
canopy_water_capacity(h::InterceptingHydrology, land) =
    KernelFunctionOperation{Center, Center, Nothing}(canopy_water_capacity, land.grid, h, land.clock)

@inline canopy_water_capacity(i, j, k, grid, h, clock) =
    h.capacity_per_leaf_area * stateindex(h.leaf_area_index, i, j, 1, grid, Time(clock.time), (Center, Center, Center))

Base.summary(h::InterceptingHydrology) =
    string("InterceptingHydrology(soil=", summary(h.soil),
           ", c=", prettysummary(h.capacity_per_leaf_area), ")")
