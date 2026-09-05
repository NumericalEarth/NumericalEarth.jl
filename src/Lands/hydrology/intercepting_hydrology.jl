#####
##### `InterceptingHydrology` — a canopy interception store wrapping a soil hydrology.
#####

"""
    InterceptingHydrology(FT = Oceananigans.defaults.FloatType;
                          soil,
                          leaf_area_index,
                          vegetation_fraction = 1,
                          capacity_per_leaf_area = 0.1,
                          extinction = 0.5,
                          clumping = 1,
                          drainage_smoothing_width = 0)

Canopy interception store `Wᶜ` wrapping a soil hydrology `soil`. It splits rain into
interception and throughfall, applies the canopy vapor flux, and sheds water over capacity.

`leaf_area_index` is one-sided leaf area per vegetated ground area. A `Number` or
static `Field` runs on CPU and GPU; a `FieldTimeSeries` currently runs on CPU only.

* `capacity_per_leaf_area` — `c`, canopy water capacity per unit LAI (kg m⁻² ≈ 0.1 mm/LAI).
* `vegetation_fraction` — vegetated fraction of the grid cell.
* `extinction`, `clumping` — Beer–Lambert `K`, `Ω` setting the caught fraction `fⁱⁿᵗ`.
* `drainage_smoothing_width` — `w` (kg m⁻²), the over-capacity interval in which
  drip ramps smoothly from zero to the sharp cap. It permits a temporary overshoot of at
  most `4w/27`; `0` (default) is the exact cap.
"""
struct InterceptingHydrology{S, L, V, FT} <: AbstractHydrology
    soil                     :: S
    leaf_area_index          :: L
    vegetation_fraction      :: V
    capacity_per_leaf_area   :: FT
    extinction               :: FT
    clumping                 :: FT
    drainage_smoothing_width :: FT
end

@inline canopy_surface_property(x::Number, FT) = convert(FT, x)
@inline canopy_surface_property(x, FT) = x

function InterceptingHydrology(FT::Type = Oceananigans.defaults.FloatType;
                               soil,
                               leaf_area_index,
                               vegetation_fraction = 1,
                               capacity_per_leaf_area = 0.1,
                               extinction = 0.5,
                               clumping = 1,
                               drainage_smoothing_width = 0)
    return InterceptingHydrology(soil,
                                 canopy_surface_property(leaf_area_index, FT),
                                 canopy_surface_property(vegetation_fraction, FT),
                                 convert(FT, capacity_per_leaf_area),
                                 convert(FT, extinction),
                                 convert(FT, clumping),
                                 convert(FT, drainage_smoothing_width))
end

Adapt.adapt_structure(to, h::InterceptingHydrology) =
    InterceptingHydrology(Adapt.adapt(to, h.soil),
                          Adapt.adapt(to, h.leaf_area_index),
                          Adapt.adapt(to, h.vegetation_fraction),
                          h.capacity_per_leaf_area,
                          h.extinction,
                          h.clumping,
                          h.drainage_smoothing_width)

prognostic_variables(h::InterceptingHydrology) =
    merge_unique(prognostic_variables(h.soil), (:canopy_water_storage,))

flux_variables(h::InterceptingHydrology) =
    merge_unique(flux_variables(h.soil), (:liquid_precipitation_flux, :canopy_evaporation))

diagnostic_variables(h::InterceptingHydrology) =
    merge_unique(diagnostic_variables(h.soil),
                 (:throughfall, :canopy_water_storage_tendency, :wet_canopy_evaporation,
                  :canopy_water_capacity))

# Delegate the initial-field builders to the wrapped soil so any soil-specific field
# shapes are preserved; interception's own fields fall through to the defaults.
initial_flux(h::InterceptingHydrology, name::Symbol, grid) = initial_flux(h.soil, name, grid)
initial_diagnostic(h::InterceptingHydrology, name::Symbol, grid) = initial_diagnostic(h.soil, name, grid)

#####
##### Interception step — runs before the delegated soil hydrology step.
#####

# Smooth (C¹) positive part that is zero for `x ≤ 0` and equals `x` for `x ≥ w`.
@inline function smooth_positive_part(value, width)
    hard_positive_part = max(value, zero(value))
    has_smoothing = width > zero(width)
    effective_width = ifelse(has_smoothing, width, one(width))
    smoothing_fraction = clamp(hard_positive_part / effective_width, zero(value), one(value))
    smoothed_value = hard_positive_part * smoothing_fraction * (2 - smoothing_fraction)
    return ifelse(has_smoothing, smoothed_value, hard_positive_part)
end

# Returns the new store, drip, throughfall, and realized canopy vapor flux.
@inline function canopy_store_update(Wᶜ, rain, Eᶜ, Wᶜᵐᵃˣ, fⁱⁿᵗ, w, Δt)
    Pⁱⁿᵗ = fⁱⁿᵗ * rain
    realized_canopy_flux = min(Eᶜ, Wᶜ / Δt + Pⁱⁿᵗ)
    Wᶜᵗ = Wᶜ + Δt * (Pⁱⁿᵗ - realized_canopy_flux)
    drip_mass = smooth_positive_part(Wᶜᵗ - Wᶜᵐᵃˣ, w)
    Wᶜⁿ⁺¹ = Wᶜᵗ - drip_mass
    drip = drip_mass / Δt
    throughfall = rain - Pⁱⁿᵗ + drip
    return Wᶜⁿ⁺¹, drip, throughfall, realized_canopy_flux
end

@inline function local_vegetation_fraction(h, i, j, grid, time)
    FT = eltype(grid)
    fraction = stateindex(h.vegetation_fraction, i, j, 1, grid, Time(time),
                          (Center, Center, Center))
    return clamp(convert(FT, fraction), zero(FT), one(FT))
end

@kernel function _interception_step!(Wc, Pl, canopy_vapor_flux, throughfall,
                                     realized_canopy_vapor_flux, dWcdt, h, Δt, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Wcⁿ   = Wc[i, j, 1]
        rain  = Pl[i, j, 1]       # raw rain, positive down
        Eᶜ = canopy_vapor_flux[i, j, 1]
    end
    FT    = eltype(grid)
    # `Time(time)` so a time-varying (`FieldTimeSeries`) LAI interpolates to the clock;
    # a `Number`/`Field` LAI ignores the time argument.
    LAI   = convert(FT, stateindex(h.leaf_area_index, i, j, 1, grid, Time(time), (Center, Center, Center)))
    vegetation_fraction = local_vegetation_fraction(h, i, j, grid, time)
    Wcᵐᵃˣ = vegetation_fraction * h.capacity_per_leaf_area * LAI
    fⁱⁿᵗ = vegetation_fraction * (1 - canopy_transmittance(h.extinction, h.clumping, LAI))

    Wcⁿ⁺¹, _, Pˡ, realized_canopy_flux = canopy_store_update(Wcⁿ, rain, Eᶜ, Wcᵐᵃˣ, fⁱⁿᵗ,
                                                              h.drainage_smoothing_width, Δt)

    @inbounds begin
        Wc[i, j, 1]            = Wcⁿ⁺¹
        Pl[i, j, 1]            = Pˡ
        throughfall[i, j, 1]   = Pˡ
        realized_canopy_vapor_flux[i, j, 1] = realized_canopy_flux
        dWcdt[i, j, 1]         = (Wcⁿ⁺¹ - Wcⁿ) / Δt
    end
end

function time_step!(h::InterceptingHydrology, land, Δt, time)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _interception_step!,
            land.prognostic.canopy_water_storage,
            land.fluxes.liquid_precipitation_flux,
            land.fluxes.canopy_evaporation,
            land.diagnostics.throughfall,
            land.diagnostics.wet_canopy_evaporation,
            land.diagnostics.canopy_water_storage_tendency,
            h, Δt, land.grid, time)
    time_step!(h.soil, land, Δt, time)
    return nothing
end

# Publish the current canopy water capacity.
function update_diagnostics!(h::InterceptingHydrology, land)
    update_diagnostics!(h.soil, land)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _canopy_capacity!,
            land.diagnostics.canopy_water_capacity, h, land.grid, land.clock.time)
    return nothing
end

@kernel function _canopy_capacity!(capacity, h, grid, time)
    i, j = @index(Global, NTuple)
    FT  = eltype(grid)
    LAI = convert(FT, stateindex(h.leaf_area_index, i, j, 1, grid, Time(time), (Center, Center, Center)))
    vegetation_fraction = local_vegetation_fraction(h, i, j, grid, time)
    @inbounds capacity[i, j, 1] = vegetation_fraction * h.capacity_per_leaf_area * LAI
end

saturation(h::InterceptingHydrology, land) = saturation(h.soil, land)

EarthSystemModels.surface_retention_curve(h::InterceptingHydrology) =
    EarthSystemModels.surface_retention_curve(h.soil)

Base.summary(h::InterceptingHydrology) =
    string("InterceptingHydrology(soil=", summary(h.soil),
           ", c=", prettysummary(h.capacity_per_leaf_area), ")")
