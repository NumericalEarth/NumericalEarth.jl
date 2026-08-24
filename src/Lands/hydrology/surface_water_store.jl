#####
##### `SurfaceWaterStore` — a prognostic surface pond wrapping a soil hydrology.
#####
##### Water a runoff closure rejects at the surface (the wrapped soil's
##### `surface_runoff`) has not left the column — it ponds. This closure adds one
##### prognostic, the surface water store `Sˢᶠᶜ` (kg m⁻²), that catches everything
##### the soil rejects, drains to true runoff on a timescale `τ`, and re-offers the
##### remainder to infiltration. Each step, around the delegated soil step:
#####
#####   R    = Sⁿ (1 − e^{−Δt/τ}) / Δt   # true runoff (exact linear drain)
#####   Pˡ  += Sⁿ e^{−Δt/τ} / Δt         # remainder re-offered to infiltration
#####   (soil step: infiltrates up to its cap, rejects the rest into `surface_runoff`)
#####   Sⁿ⁺¹ = Δt · Rˢᶠᶜ                 # rejected water refills the pond
#####
##### Nothing is created or destroyed — pond → offer → infiltrated or rejected →
##### pond — so the surface water balance closes to machine precision and the store
##### stays non-negative without a clamp. The re-offer passes through the soil's own
##### runoff cap, so the pond needs no knowledge of the infiltration model.
#####

"""
    SurfaceWaterStore(FT = Oceananigans.defaults.FloatType;
                      soil,
                      drainage_timescale = 3600)

Prognostic surface pond `Sˢᶠᶜ` (kg m⁻²) wrapping a soil hydrology `soil` (typically a
[`VariablySaturatedHydrology`](@ref) whose runoff closure publishes `surface_runoff`).
Rejected infiltration lands in the pond instead of vanishing; the pond drains to true
runoff on the e-folding timescale `drainage_timescale` (`τ`, s) — published as
`surface_water_runoff`, the flux that actually leaves the column — and re-offers what
remains to infiltration through `liquid_precipitation_flux`. The drain is the exact
solution of the linear reservoir `dS/dt = −S/τ` over the step.

The pond carries no internal energy (a 5 mm pond is ~2% of a 0.1 m dry slab's areal
heat capacity), so pond drainage advects no heat yet.

```jldoctest
julia> using NumericalEarth

julia> soil = VariablySaturatedHydrology(slab_depth = 1, porosity = 0.4, storage_height = 0.1,
                                         retention_curve = VanGenuchtenRetention(α = 2, n = 1.5),
                                         hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 1.5),
                                         runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3));

julia> summary(SurfaceWaterStore(soil = soil))
"SurfaceWaterStore(soil=VariablySaturatedHydrology(slab_depth=1.0, porosity=0.4, retention=VanGenuchtenRetention(α=2.0, n=1.5), deep=NoDeepLiquidFlux, runoff=InfiltrationCapacityRunoff(infiltration_capacity=0.001)), τ=3600.0)"
```
"""
struct SurfaceWaterStore{S, FT} <: AbstractHydrology
    soil               :: S
    drainage_timescale :: FT
end

SurfaceWaterStore(FT::Type = Oceananigans.defaults.FloatType;
                  soil,
                  drainage_timescale = 3600) =
    SurfaceWaterStore(soil, convert(FT, drainage_timescale))

Adapt.adapt_structure(to, h::SurfaceWaterStore) =
    SurfaceWaterStore(Adapt.adapt(to, h.soil), h.drainage_timescale)

prognostic_variables(h::SurfaceWaterStore) =
    merge_unique(prognostic_variables(h.soil), (:surface_water_storage,))

flux_variables(h::SurfaceWaterStore) =
    merge_unique(flux_variables(h.soil), (:liquid_precipitation_flux,))

diagnostic_variables(h::SurfaceWaterStore) =
    merge_unique(diagnostic_variables(h.soil),
                 (:surface_water_runoff, :surface_water_storage_tendency))

initial_flux(h::SurfaceWaterStore, name::Symbol, grid) = initial_flux(h.soil, name, grid)
initial_diagnostic(h::SurfaceWaterStore, name::Symbol, grid) = initial_diagnostic(h.soil, name, grid)

update_diagnostics!(h::SurfaceWaterStore, land) = update_diagnostics!(h.soil, land)

saturation(h::SurfaceWaterStore, land) = saturation(h.soil, land)

#####
##### Pond drain (before the soil step) and refill (after it).
#####

@kernel function _surface_water_drain!(S, Pl, R, τ, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Sⁿ = S[i, j, 1]
        Sᵈ = Sⁿ * exp(-Δt / τ)   # store left after the exact linear drain
        Pl[i, j, 1] += Sᵈ / Δt
        R[i, j, 1]   = (Sⁿ - Sᵈ) / Δt
    end
end

@kernel function _surface_water_refill!(S, Rˢᶠᶜ, dSdt, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Sⁿ   = S[i, j, 1]
        Sⁿ⁺¹ = Δt * Rˢᶠᶜ[i, j, 1]
        S[i, j, 1]    = Sⁿ⁺¹
        dSdt[i, j, 1] = (Sⁿ⁺¹ - Sⁿ) / Δt
    end
end

EarthSystemModels.surface_retention_curve(h::SurfaceWaterStore) =
    EarthSystemModels.surface_retention_curve(h.soil)

function time_step!(h::SurfaceWaterStore, land, Δt, time)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _surface_water_drain!,
            land.prognostic.surface_water_storage,
            land.fluxes.liquid_precipitation_flux,
            land.diagnostics.surface_water_runoff,
            h.drainage_timescale, Δt)
    time_step!(h.soil, land, Δt, time)
    launch!(arch, land.grid, :xy, _surface_water_refill!,
            land.prognostic.surface_water_storage,
            land.diagnostics.surface_runoff,
            land.diagnostics.surface_water_storage_tendency,
            Δt)
    return nothing
end

Base.summary(h::SurfaceWaterStore) =
    string("SurfaceWaterStore(soil=", summary(h.soil),
           ", τ=", prettysummary(h.drainage_timescale), ")")
