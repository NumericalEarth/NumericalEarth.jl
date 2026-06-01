#####
##### `VariablySaturatedBucketHydrology` — conservative variably saturated slab
##### hydrology.
#####
##### Replaces `BucketHydrology`'s `clamp(M + (P − E) Δt, 0, M⁺)` update with
##### a signed-flux conservative budget:
#####
#####     dMˡᵃ/dt = Jˡ_b − Jˡ_s − Jᵛ − Rᴹ_lat
#####
##### where every term is positive-upward (`Jˡ_s = −Pˡ + Rᴹ_sfc`). The
##### conservative storage variable is the *augmented* liquid fraction
##### `ϑˡ = θˡ + Sₛ·max(Π,0)`, so `Mˡᵃ > Mˡᵃ⁺` is admitted and corresponds to
##### saturated positive-pressure storage (`Π > 0`).
#####
##### Diagnostics published every step:
#####   * `deep_liquid_flux`        (Jˡ_b, positive upward)
#####   * `surface_liquid_flux`     (Jˡ_s, positive upward)
#####   * `surface_runoff`          (Rᴹ_sfc, ≥ 0; rejected input)
#####   * `subsurface_runoff`       (Rᴹ_lat, ≥ 0; storage export)
#####   * `water_storage_tendency`  (dMˡᵃ/dt, kg m⁻² s⁻¹)
#####
##### The diagnostic surface saturation `𝒮 = clamp(θˡ/ν, 0, 1)` is written into
##### `land.saturation` by `update_diagnostics!` (read by surface-property and
##### humidity closures) and `moisture_availability = β = clamp(𝒮/𝒮ᶜ, 0, 1)` is
##### the `saturation(...)` accessor result.
#####

"""
    VariablySaturatedBucketHydrology(FT = Oceananigans.defaults.FloatType;
                                     slab_depth,
                                     porosity,
                                     residual_liquid_fraction = 0,
                                     specific_storage,
                                     critical_saturation,
                                     liquid_density = 1000,
                                     retention_curve,
                                     hydraulic_conductivity,
                                     deep_liquid_flux = NoDeepLiquidFlux(),
                                     deep_pressure_head = 0,
                                     runoff = NoRunoff())

Conservative slab hydrology with an augmented liquid fraction `ϑˡ`. The
storage variable `Mˡᵃ` evolves under signed fluxes (positive upward at the
surface; positive upward at the bottom).

* `slab_depth` (`D`, m) — slab thickness.
* `porosity` (`ν`) — total pore fraction.
* `residual_liquid_fraction` (`θʳ`) — minimum pore liquid (default 0).
* `specific_storage` (`Sₛ`, m⁻¹) — saturated pressure-storage coefficient.
* `critical_saturation` (`𝒮ᶜ`) — saturation above which `β = 1`.
* `retention_curve` — e.g. [`VanGenuchtenRetention`](@ref).
* `hydraulic_conductivity` — e.g. [`VanGenuchtenConductivity`](@ref).
* `deep_liquid_flux` — bottom-boundary closure: [`NoDeepLiquidFlux`](@ref),
  [`FreeDrainageFlux`](@ref), [`DarcyDeepLiquidFlux`](@ref), or
  [`LinearReservoirDrainage`](@ref).
* `deep_pressure_head` — the deep-reservoir pressure head (m), passed to the
  deep-flux closure as `Π_D`. Default 0.
* `runoff` — runoff closure: [`NoRunoff`](@ref) or
  [`InfiltrationCapacityRunoff`](@ref).
"""
struct VariablySaturatedBucketHydrology{FT, R, C, DF, PD, RO} <: AbstractHydrology
    slab_depth               :: FT
    porosity                 :: FT
    residual_liquid_fraction :: FT
    specific_storage         :: FT
    critical_saturation      :: FT
    liquid_density           :: FT
    retention_curve          :: R
    hydraulic_conductivity   :: C
    deep_liquid_flux         :: DF
    deep_pressure_head       :: PD
    runoff                   :: RO
end

function VariablySaturatedBucketHydrology(FT::Type = Oceananigans.defaults.FloatType;
                                          slab_depth,
                                          porosity,
                                          residual_liquid_fraction = 0,
                                          specific_storage,
                                          critical_saturation,
                                          liquid_density = 1000,
                                          retention_curve,
                                          hydraulic_conductivity,
                                          deep_liquid_flux = NoDeepLiquidFlux(),
                                          deep_pressure_head = 0,
                                          runoff = NoRunoff())
    return VariablySaturatedBucketHydrology(convert(FT, slab_depth),
                                            convert(FT, porosity),
                                            convert(FT, residual_liquid_fraction),
                                            convert(FT, specific_storage),
                                            convert(FT, critical_saturation),
                                            convert(FT, liquid_density),
                                            retention_curve,
                                            hydraulic_conductivity,
                                            deep_liquid_flux,
                                            normalize_property(FT, deep_pressure_head),
                                            runoff)
end

# Coupler writes vapor flux and liquid precipitation into `land.fluxes`.
# `vapor_flux` is the signed `Jᵛ` (positive upward) — replaces the legacy
# `precipitation/evaporation` positive-part fields for this closure.
# `liquid_precipitation_flux` is `Pˡ` (positive downward).
flux_variables(::VariablySaturatedBucketHydrology) =
    (:vapor_flux, :liquid_precipitation_flux)

diagnostic_variables(::VariablySaturatedBucketHydrology) =
    (:deep_liquid_flux,
     :surface_liquid_flux,
     :surface_runoff,
     :subsurface_runoff,
     :water_storage_tendency)

#####
##### Per-cell helpers — used by both `update_diagnostics!` and `time_step!`.
#####

@inline function augmented_liquid_fraction(h, M)
    FT = typeof(M)
    return M / (convert(FT, h.liquid_density) * convert(FT, h.slab_depth))
end

@inline function liquid_fraction(h, M)
    ϑˡ = augmented_liquid_fraction(h, M)
    return min(ϑˡ, convert(typeof(ϑˡ), h.porosity))
end

@inline function liquid_saturation(h, θˡ)
    FT  = typeof(θˡ)
    ν   = convert(FT, h.porosity)
    θʳ  = convert(FT, h.residual_liquid_fraction)
    Δ   = ν - θʳ
    return clamp((θˡ - θʳ) / Δ, zero(FT), one(FT))
end

@inline function pressure_head_diag(h, M, θˡ, 𝒮)
    FT  = typeof(M)
    ν   = convert(FT, h.porosity)
    ρˡ  = convert(FT, h.liquid_density)
    D   = convert(FT, h.slab_depth)
    Sₛ  = convert(FT, h.specific_storage)
    Mˡᵃ⁺ = ρˡ * ν * D
    # Unsaturated branch: Π = Π_m(𝒮). Saturated branch: Π = (M − M⁺)/(ρˡ D Sₛ).
    return ifelse(M < Mˡᵃ⁺,
                  pressure_head(h.retention_curve, 𝒮),
                  (M - Mˡᵃ⁺) / (ρˡ * D * Sₛ))
end

@inline moisture_availability(h, 𝒮) =
    clamp(𝒮 / convert(typeof(𝒮), h.critical_saturation), zero(𝒮), one(𝒮))

#####
##### Diagnostic-saturation kernel — refreshes `land.saturation` from `M`.
#####

@kernel function _variably_saturated_saturation!(saturation, M, h)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Mij = M[i, j, 1]
        θˡ  = liquid_fraction(h, Mij)
        𝒮   = liquid_saturation(h, θˡ)
        saturation[i, j, 1] = 𝒮
    end
end

function update_diagnostics!(h::VariablySaturatedBucketHydrology, land)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _variably_saturated_saturation!,
            land.saturation, land.water_storage, h)
    return nothing
end

saturation(h::VariablySaturatedBucketHydrology, land) = land.saturation

#####
##### Time-step kernel.
#####

@kernel function _variably_saturated_step!(M, sat,
                                           Jˡb_diag, Jˡs_diag, Rsfc_diag, Rlat_diag, dMdt_diag,
                                           Jv, Pl, h, deep_pressure_head,
                                           Δt, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Mij = M[i, j, 1]
        Jvij = Jv[i, j, 1]
        Plij = Pl[i, j, 1]
        ψ_D  = stateindex(deep_pressure_head, i, j, 1, grid, time, (Center, Center, Center))
    end

    θˡ = liquid_fraction(h, Mij)
    𝒮  = liquid_saturation(h, θˡ)
    Π  = pressure_head_diag(h, Mij, θˡ, 𝒮)
    K  = hydraulic_conductivity(h.hydraulic_conductivity, 𝒮)

    Jˡs, Rsfc = surface_liquid_flux_and_runoff(h.runoff, Plij, Mij, θˡ, 𝒮, Π, K)
    Jˡb       = deep_liquid_flux(h.deep_liquid_flux, Mij, θˡ, 𝒮, Π, K, ψ_D, time)
    Rlat      = subsurface_runoff(h.runoff, Mij, Π, K)

    dMdt = Jˡb - Jˡs - Jvij - Rlat

    @inbounds begin
        Mnew = max(Mij + Δt * dMdt, zero(Mij))
        M[i, j, 1]         = Mnew
        Jˡb_diag[i, j, 1]  = Jˡb
        Jˡs_diag[i, j, 1]  = Jˡs
        Rsfc_diag[i, j, 1] = Rsfc
        Rlat_diag[i, j, 1] = Rlat
        dMdt_diag[i, j, 1] = dMdt
        # Refresh saturation immediately so the energy step (which runs after
        # hydrology) sees state consistent with the new M.
        θˡn = liquid_fraction(h, Mnew)
        sat[i, j, 1] = liquid_saturation(h, θˡn)
    end
end

function time_step!(h::VariablySaturatedBucketHydrology, land, Δt, time)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _variably_saturated_step!,
            land.water_storage, land.saturation,
            land.diagnostics.deep_liquid_flux,
            land.diagnostics.surface_liquid_flux,
            land.diagnostics.surface_runoff,
            land.diagnostics.subsurface_runoff,
            land.diagnostics.water_storage_tendency,
            land.fluxes.vapor_flux,
            land.fluxes.liquid_precipitation_flux,
            h, h.deep_pressure_head, Δt, land.grid, time)
    return nothing
end

Base.summary(h::VariablySaturatedBucketHydrology) =
    string("VariablySaturatedBucketHydrology(",
           "slab_depth=", prettysummary(h.slab_depth),
           ", porosity=", prettysummary(h.porosity),
           ", retention=", summary(h.retention_curve),
           ", deep=", summary(h.deep_liquid_flux),
           ", runoff=", summary(h.runoff), ")")
