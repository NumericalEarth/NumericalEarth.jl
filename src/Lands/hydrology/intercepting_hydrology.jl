#####
##### `InterceptingHydrology` — a canopy water store `Wᶜ` between the rain and a soil
##### hydrology: intercepted rain fills it, the interface's wet-canopy evaporation drains
##### it, over-capacity drip and the uncaught rain reach the soil as throughfall.
#####

"""
    InterceptingHydrology(FT = Oceananigans.defaults.FloatType;
                          soil,
                          leaf_area_index,
                          capacity_per_leaf_area = 0.1,
                          extinction = 0.5,
                          clumping = 1,
                          cover_fraction = 1)

Canopy water store `Wᶜ` (kg m⁻²) wrapping a [`VariablySaturatedHydrology`](@ref)
`soil`. Rain is caught by the Beer–Lambert canopy fraction `1 − e^{−KΩ·LAI}` up to the
capacity `c·LAI`; the wet canopy evaporates at the rate `Eʷᵉᵗ` the interface computes
(a [`CanopyAirSpace`](@ref) with `CanopyInterception`), and the uncaught rain plus the
over-capacity drip reach the soil as throughfall.

`leaf_area_index` should be the same LAI passed to the interface canopy; a
`FieldTimeSeries` LAI runs on CPU only. The store is kept per unit vegetated area; on a
tiled cell `cover_fraction` is the vegetated fraction of a [`TiledLandInterface`](@ref),
which scales the interception and drip the soil sees.

* `capacity_per_leaf_area` — `c` (kg m⁻² per unit LAI).
* `extinction`, `clumping` — Beer–Lambert `K`, `Ω` of the caught fraction.
* `cover_fraction` — vegetated fraction of the cell (a `Number` or `Field`).
"""
struct InterceptingHydrology{S, L, C, FT} <: AbstractHydrology
    soil                     :: S
    leaf_area_index          :: L
    cover_fraction           :: C
    capacity_per_leaf_area   :: FT
    extinction               :: FT
    clumping                 :: FT
end

@inline canopy_lai_property(x::Number, FT) = convert(FT, x)
@inline canopy_lai_property(x, FT) = x

function InterceptingHydrology(FT::Type = Oceananigans.defaults.FloatType;
                               soil,
                               leaf_area_index,
                               capacity_per_leaf_area = 0.1,
                               extinction = 0.5,
                               clumping = 1,
                               cover_fraction = 1)
    return InterceptingHydrology(soil,
                                 canopy_lai_property(leaf_area_index, FT),
                                 canopy_lai_property(cover_fraction, FT),
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
    merge_unique(diagnostic_variables(h.soil),
                 (:throughfall, :canopy_water_storage_tendency, :canopy_water_capacity))

initial_flux(h::InterceptingHydrology, name::Symbol, grid) = initial_flux(h.soil, name, grid)
initial_diagnostic(h::InterceptingHydrology, name::Symbol, grid) = initial_diagnostic(h.soil, name, grid)

#####
##### Interception step, ahead of the soil step that reads its throughfall.
#####

# Canopy store update over one step, per unit vegetated area: returns the new store, the
# drip, and the throughfall reaching the soil of a cell with vegetated fraction `f`,
# with rain = (Wᶜⁿ⁺¹ − Wᶜ)/Δt · f + Eʷᵉᵗ f + throughfall exactly.
@inline function canopy_store_update(Wᶜ, rain, Eʷᵉᵗ, Wᶜᵐᵃˣ, fⁱⁿᵗ, f, Δt)
    Pⁱⁿᵗ      = fⁱⁿᵗ * rain
    Wᶜᵗ       = Wᶜ + Δt * (Pⁱⁿᵗ - Eʷᵉᵗ)
    drip_mass = max(Wᶜᵗ - Wᶜᵐᵃˣ, zero(Wᶜ))
    Wᶜⁿ⁺¹     = max(Wᶜᵗ - drip_mass, zero(Wᶜ))
    drip      = drip_mass / Δt
    return Wᶜⁿ⁺¹, drip, rain - f * (Pⁱⁿᵗ - drip)
end

@kernel function _interception_step!(Wc, Pl, Cev, throughfall, dWcdt, h, Δt, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Wcⁿ  = Wc[i, j, 1]
        rain = Pl[i, j, 1]
        Eʷᵉᵗ = Cev[i, j, 1]
    end
    FT    = typeof(Wcⁿ)
    LAI   = convert(FT, stateindex(h.leaf_area_index, i, j, 1, grid, Time(time), (Center, Center, Center)))
    f     = convert(FT, stateindex(h.cover_fraction, i, j, 1, grid, Time(time), (Center, Center, Center)))
    Wcᵐᵃˣ = h.capacity_per_leaf_area * LAI
    fⁱⁿᵗ  = 1 - canopy_transmittance(h.extinction, h.clumping, LAI)

    Wcⁿ⁺¹, _, Pˡ = canopy_store_update(Wcⁿ, rain, Eʷᵉᵗ, Wcᵐᵃˣ, fⁱⁿᵗ, f, Δt)

    @inbounds begin
        Wc[i, j, 1]          = Wcⁿ⁺¹
        throughfall[i, j, 1] = Pˡ
        dWcdt[i, j, 1]       = (Wcⁿ⁺¹ - Wcⁿ) / Δt
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

# The store capacity `Wᶜᵐᵃˣ = c·LAI` the interface normalizes the wet fraction by.
function update_diagnostics!(h::InterceptingHydrology, land)
    update_diagnostics!(h.soil, land)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _canopy_capacity!,
            land.diagnostics.canopy_water_capacity, h, land.grid, land.clock.time)
    return nothing
end

@kernel function _canopy_capacity!(capacity, h, grid, time)
    i, j = @index(Global, NTuple)
    FT  = eltype(capacity)
    LAI = convert(FT, stateindex(h.leaf_area_index, i, j, 1, grid, Time(time), (Center, Center, Center)))
    @inbounds capacity[i, j, 1] = h.capacity_per_leaf_area * LAI
end

saturation(h::InterceptingHydrology, land) = saturation(h.soil, land)

EarthSystemModels.surface_retention_curve(h::InterceptingHydrology) =
    EarthSystemModels.surface_retention_curve(h.soil)

Base.summary(h::InterceptingHydrology) =
    string("InterceptingHydrology(soil=", summary(h.soil),
           ", c=", prettysummary(h.capacity_per_leaf_area), ")")
