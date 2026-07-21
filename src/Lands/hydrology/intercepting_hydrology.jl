#####
##### `InterceptingHydrology` — a canopy interception store wrapping a soil hydrology.
#####
##### A wet canopy is a small leaky bucket between the rain and the soil. This closure
##### adds one prognostic, the canopy water store `Wᶜ` (kg m⁻² ≡ mm), and *wraps* an
##### underlying soil hydrology (a [`VariablySaturatedHydrology`](@ref)) — the same
##### wrap-and-delegate pattern the interface side uses in `CompositeSurfaceHumidity` /
##### `CanopyAirSpace`. Each step, before the soil hydrology runs, the interception step:
#####
#####   f_int  = 1 − exp(−K·LAI·Ω)                        # canopy-caught fraction (Beer–Lambert)
#####   P_int  = f_int · P                                # intercepted rain
#####   Wᶜ    ← clamp(Wᶜ + Δt(P_int − E_wet), 0, Wᶜᵐᵃˣ)   # store update, capacity Wᶜᵐᵃˣ = c·LAI
#####   D      = max(Wᶜ + Δt(P_int − E_wet) − Wᶜᵐᵃˣ, 0)/Δt  # canopy drip (over-capacity shed)
#####   Pˡ    ← (P − P_int) + D                           # throughfall → soil
#####
##### `E_wet` (wet-canopy evaporation, the potential-rate leaf latent share) is computed
##### on the interface side by a `CanopyAirSpace` carrying a `CanopyInterception` and
##### delivered here through the `canopy_evaporation` flux accumulator; the soil vapor
##### sink `land.fluxes.vapor_flux` already excludes it (`Jᵛ − E_wet`). The interception
##### step overwrites `liquid_precipitation_flux` with the throughfall the soil then
##### reads, so ordering within the delegated `time_step!` is interception → soil.
#####
##### Physics: Rutter (1971) running canopy balance, Deardorff (1978) wet fraction,
##### BATS/CLM `Wᶜᵐᵃˣ = c·LAI` capacity (`c ≈ 0.1 kg m⁻²`), Beer–Lambert cover.
#####

"""
    InterceptingHydrology(FT = Oceananigans.defaults.FloatType;
                          soil,
                          leaf_area_index,
                          capacity_per_leaf_area = 0.1,
                          extinction = 0.5,
                          clumping = 1,
                          drainage_smoothing_width = 0)

Canopy interception store `Wᶜ` wrapping a soil hydrology `soil` (typically a
[`VariablySaturatedHydrology`](@ref)). Splits incoming rain into interception and
throughfall, drains the wet canopy at the potential rate `E_wet` supplied by the
interface, sheds drip over capacity, and routes throughfall to `soil`.

`leaf_area_index` should be the *same* LAI object passed to the interface
[`CanopyConductanceHumidity`](@ref). A `Number` or static `Field` LAI runs on CPU
and GPU; a `FieldTimeSeries` (time-varying LAI) currently runs on **CPU only** (the
series is indexed in-kernel, which does not adapt to GPU — a follow-up).

* `capacity_per_leaf_area` — `c`, canopy water capacity per unit LAI (kg m⁻² ≈ 0.1 mm/LAI).
* `extinction`, `clumping` — Beer–Lambert `K`, `Ω` setting the caught fraction `f_int`.
* `drainage_smoothing_width` — `w` (kg m⁻²), softens the over-capacity drip so the store
  update is C¹ for the adjoint (Enzyme/Reactant). `0` (default) is the sharp cap, exact.
"""
struct InterceptingHydrology{S, L, FT} <: AbstractHydrology
    soil                     :: S
    leaf_area_index          :: L
    capacity_per_leaf_area   :: FT
    extinction               :: FT
    clumping                 :: FT
    drainage_smoothing_width :: FT
end

# Keep a scalar LAI as `FT`; a `Field` (static map) or `FieldTimeSeries` (time-varying)
# passes through untouched, mirroring the interface `CanopyConductanceHumidity`.
@inline canopy_lai_property(x::Number, FT) = convert(FT, x)
@inline canopy_lai_property(x, FT) = x

function InterceptingHydrology(FT::Type = Oceananigans.defaults.FloatType;
                               soil,
                               leaf_area_index,
                               capacity_per_leaf_area = 0.1,
                               extinction = 0.5,
                               clumping = 1,
                               drainage_smoothing_width = 0)
    return InterceptingHydrology(soil,
                                 canopy_lai_property(leaf_area_index, FT),
                                 convert(FT, capacity_per_leaf_area),
                                 convert(FT, extinction),
                                 convert(FT, clumping),
                                 convert(FT, drainage_smoothing_width))
end

Adapt.adapt_structure(to, h::InterceptingHydrology) =
    InterceptingHydrology(Adapt.adapt(to, h.soil),
                          Adapt.adapt(to, h.leaf_area_index),
                          h.capacity_per_leaf_area,
                          h.extinction,
                          h.clumping,
                          h.drainage_smoothing_width)

# Declarations — merge the wrapped soil's with the interception's. The store is the
# extra prognostic; `canopy_evaporation` (E_wet) is the extra flux the coupler writes;
# `liquid_precipitation_flux` is guaranteed present (the interception step overwrites it
# with throughfall). Throughfall and the store tendency are published for diagnostics.
prognostic_variables(::InterceptingHydrology) = (:canopy_water_storage,)

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

# Smooth (C¹) positive part `≈ max(x, 0)`, a numerically stable softplus of width `w`.
# `w = 0` recovers the sharp `max(x, 0)` exactly (no NaN in the unused soft branch).
@inline function smooth_positive_part(x, w)
    hard = max(x, zero(x))
    positive = w > zero(w)
    ws   = ifelse(positive, w, one(w))
    soft = hard + ws * log1p(exp(-abs(x) / ws))
    return ifelse(positive, soft, hard)
end

# Pure, allocation-free canopy-store update, split out so the kernel and the
# differentiability tests share it. Returns the new store, drip, throughfall, and the
# *realized* wet-canopy evaporation. Conserves canopy water exactly for any `w`:
# `rain = (Wᶜⁿ⁺¹ − Wᶜ)/Δt + E_wet_realized + throughfall`. The demand cap
# `E_wet_realized = min(E_wet, Wᶜ/Δt + P_int)` keeps `Wᶜ ≥ 0` (the store cannot evaporate
# more water than it holds); the smoothed over-capacity shed bounds it above by `Wᶜᵐᵃˣ`.
@inline function canopy_store_update(Wᶜ, rain, E_wet, Wᶜᵐᵃˣ, f_int, w, Δt)
    P_int       = f_int * rain
    E_wet_r     = min(E_wet, Wᶜ / Δt + P_int)
    Wtent       = Wᶜ + Δt * (P_int - E_wet_r)
    drip_mass   = smooth_positive_part(Wtent - Wᶜᵐᵃˣ, w)
    Wᶜⁿ⁺¹       = Wtent - drip_mass
    drip        = drip_mass / Δt
    throughfall = rain - P_int + drip
    return Wᶜⁿ⁺¹, drip, throughfall, E_wet_r
end

@kernel function _interception_step!(Wc, Pl, Cev, throughfall, Ewet_realized, dWcdt, h, Δt, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Wcⁿ   = Wc[i, j, 1]
        rain  = Pl[i, j, 1]       # raw rain, positive down
        E_wet = Cev[i, j, 1]      # interface-demanded wet-canopy evaporation, positive up
    end
    FT    = eltype(grid)
    # `Time(time)` so a time-varying (`FieldTimeSeries`) LAI interpolates to the clock;
    # a `Number`/`Field` LAI ignores the time argument.
    LAI   = convert(FT, stateindex(h.leaf_area_index, i, j, 1, grid, Time(time), (Center, Center, Center)))
    Wcᵐᵃˣ = h.capacity_per_leaf_area * LAI
    f_int = 1 - exp(-h.extinction * LAI * h.clumping)

    Wcⁿ⁺¹, _, Pˡ, E_wet_r = canopy_store_update(Wcⁿ, rain, E_wet, Wcᵐᵃˣ, f_int,
                                                h.drainage_smoothing_width, Δt)

    @inbounds begin
        Wc[i, j, 1]            = Wcⁿ⁺¹
        Pl[i, j, 1]            = Pˡ
        throughfall[i, j, 1]   = Pˡ
        Ewet_realized[i, j, 1] = E_wet_r
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

# Refresh the wrapped soil's diagnostics, then publish the store capacity `Wᶜᵐᵃˣ = c·LAI`
# the interface reads to normalize `f_wet` — kept fresh here so a time-varying LAI stays
# current and the first coupled flux computation already sees a nonzero capacity.
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
    @inbounds capacity[i, j, 1] = h.capacity_per_leaf_area * LAI
end

saturation(h::InterceptingHydrology, land) = saturation(h.soil, land)

Base.summary(h::InterceptingHydrology) =
    string("InterceptingHydrology(soil=", summary(h.soil),
           ", c=", prettysummary(h.capacity_per_leaf_area), ")")
