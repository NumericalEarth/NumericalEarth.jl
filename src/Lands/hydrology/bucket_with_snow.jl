#####
##### `BucketWithSnow` — single-bucket soil moisture + SWE + Jarvis-modulated β.
#####
##### Prognostic state:
#####   :W                     — column-integrated soil moisture [kg m⁻²]
#####   :SWE                   — snow water equivalent [m LWE]
#####   :snow_fraction         — diagnostic, tanh(SWE / SWE_crit)
#####   :moisture_availability — diagnostic β consumed by the atmosphere
#####
##### Flux inputs (NamedTuple keys the coupler must write each step):
#####   :rainfall_rate    [kg m⁻² s⁻¹]
#####   :snowfall_rate    [kg m⁻² s⁻¹]
#####   :evaporation      [kg m⁻² s⁻¹]   bare-soil evap, upward positive
#####   :sublimation      [kg m⁻² s⁻¹]   over snow-covered fraction
#####   :transpiration    [kg m⁻² s⁻¹]   plant canopy
#####   :solar_irradiance, :air_temperature, :air_humidity, :surface_pressure  (Jarvis)
#####
##### Cross-axis writes: melt cools `state.T` to close the latent-heat
##### budget for the consumed snow. This crosses the energy/water axis
##### through the shared `state` `NamedTuple` (§2.4 of the parent design
##### doc), not via a direct reference to the energy closure.
#####
##### References (active):
#####   Manabe (1969)              soil-moisture bucket + critical wetness β
#####   Lynch-Stieglitz (1994)     single-pack SWE bucket
#####   Niu & Yang (2007)          tanh snow-cover fraction
#####   Jarvis (1976)              stomatal-resistance stress functions
#####   Stewart (1988)             functional form of Jarvis applied to forests
#####   Noilhan & Planton (1989)   bare/canopy β composition by vegfrac
#####   Buck (1981)                saturation specific humidity
#####

"""
    BucketWithSnow(FT = Float64;
                   W_max = 150.0,
                   W_crit_frac = 0.75,
                   W_wilt_frac = 0.10,
                   SWE_crit = 0.05,
                   L_f = 3.337e5,
                   ρ_w = 1000.0,
                   ground_heat_capacity = 1.0e6,
                   r_smax = 5000.0,
                   solar_radiation_scale = 100.0,
                   vpd_scale = 4.0e-3,
                   T_opt = 298.0)

Combined soil-moisture bucket + minimal snow pack + Jarvis-modulated β.
`W_max` is the bucket capacity in kg m⁻² (≈ 15 cm of water);
`W_crit_frac` and `W_wilt_frac` are fractions of `W_max`.
`SWE_crit` controls the `tanh` snow-cover fraction. `ground_heat_capacity`
must match the value used in the paired `SlabEnergy` so the melt kernel
closes the latent-heat budget consistently.
"""
struct BucketWithSnow{FT} <: AbstractHydrology
    W_max                 :: FT
    W_crit_frac           :: FT
    W_wilt_frac           :: FT
    SWE_crit              :: FT
    L_f                   :: FT
    ρ_w                   :: FT
    ground_heat_capacity  :: FT
    r_smax                :: FT
    solar_radiation_scale :: FT
    vpd_scale             :: FT
    T_opt                 :: FT
end

function BucketWithSnow(FT::Type = Float64;
                        W_max = 150.0,
                        W_crit_frac = 0.75,
                        W_wilt_frac = 0.10,
                        SWE_crit = 0.05,
                        L_f = 3.337e5,
                        ρ_w = 1000.0,
                        ground_heat_capacity = 1.0e6,
                        r_smax = 5000.0,
                        solar_radiation_scale = 100.0,
                        vpd_scale = 4.0e-3,
                        T_opt = 298.0)
    return BucketWithSnow{FT}(convert.(FT,
        (W_max, W_crit_frac, W_wilt_frac, SWE_crit,
         L_f, ρ_w, ground_heat_capacity,
         r_smax, solar_radiation_scale, vpd_scale, T_opt))...)
end

prognostic_variables(::BucketWithSnow) = (:W, :SWE, :snow_fraction, :moisture_availability)
flux_variables(::BucketWithSnow)       = (:rainfall_rate, :snowfall_rate,
                                           :evaporation, :sublimation, :transpiration,
                                           :solar_irradiance, :air_temperature,
                                           :air_humidity, :surface_pressure)

function initial_state(b::BucketWithSnow, name::Symbol, grid)
    f = CenterField(grid)
    if name === :moisture_availability
        fill!(f, 1)
    end
    return f
end

wetness(::BucketWithSnow, state, parameters) = state.moisture_availability

#####
##### `step!` — water mass balance, snow accumulation, energy-limited melt
#####

@kernel function _bucket_with_snow_step!(W, SWE, T,
                                          P_rain, P_snow, E, Sub, T_t,
                                          Δt, W_max,
                                          L_f, ρ_w, ρcH_g)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(W)

        # 1. Snow mass balance — snowfall in, sublimation out.
        Δsnow_in  = max(zero(FT), P_snow[i, j, 1]) * FT(Δt) / ρ_w     # m LWE
        Δsub      = max(zero(FT), Sub[i, j, 1])    * FT(Δt) / ρ_w
        SWE_after = max(zero(FT), SWE[i, j, 1] + Δsnow_in - Δsub)

        # 2. Energy-limited melt when T > 273.15 K.
        Tg = T[i, j, 1]
        Δmelt = zero(FT)
        if SWE_after > zero(FT) && Tg > FT(273.15)
            Δmelt_max = ρcH_g * (Tg - FT(273.15)) / (L_f * ρ_w)
            Δmelt = min(Δmelt_max, SWE_after)
            SWE_after  -= Δmelt
            T[i, j, 1]  = Tg - Δmelt * L_f * ρ_w / ρcH_g
        end
        SWE[i, j, 1] = SWE_after

        # 3. Soil water balance.
        # All rainfall reaches the soil (snow-covered fraction is treated
        # as routing rain-on-snow directly through the pack); meltwater
        # also feeds the bucket.
        ΔW_in  = max(zero(FT), P_rain[i, j, 1]) * FT(Δt) + Δmelt * ρ_w
        ΔW_out = (max(zero(FT), E[i, j, 1]) + max(zero(FT), T_t[i, j, 1])) * FT(Δt)
        Wnew = clamp(W[i, j, 1] + ΔW_in - ΔW_out, zero(FT), W_max)
        W[i, j, 1] = Wnew
    end
end

function step!(b::BucketWithSnow, state, fluxes, surface, parameters, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _bucket_with_snow_step!,
            state.W, state.SWE, state.T,
            fluxes.rainfall_rate, fluxes.snowfall_rate,
            fluxes.evaporation, fluxes.sublimation, fluxes.transpiration,
            Δt, b.W_max,
            b.L_f, b.ρ_w, b.ground_heat_capacity)
    return nothing
end

#####
##### `update_diagnostics!` — snow_fraction, β with Jarvis-Stewart canopy term
#####

@inline function _esat_buck(T)
    FT = typeof(T)
    if T > FT(273.15)
        return FT(6.1121) * exp(FT(17.502) * (T - FT(273.15)) / (T - FT(32.18)))
    else
        return FT(6.1115) * exp(FT(22.452) * (T - FT(273.15)) / (T - FT(0.61)))
    end
end

@inline function _qsat_buck(T, p_hPa)
    FT = typeof(T)
    e = _esat_buck(T)
    ε = FT(0.622)
    return ε * e / (p_hPa - (one(FT) - ε) * e)
end

@inline function _jarvis_factor(rg, qa, Ta, p_hPa, β_bare, lai,
                                 solar_scale, vpd_scale, T_opt,
                                 W_crit_frac, W_wilt_frac)
    FT = typeof(rg)
    if lai ≤ zero(FT)
        return zero(FT)
    end
    F1 = (rg / solar_scale) / (one(FT) + rg / solar_scale); F1 = max(F1, FT(1e-3))
    qsat_air = _qsat_buck(Ta, p_hPa)
    F2 = one(FT) / (one(FT) + max(zero(FT), qsat_air - qa) / vpd_scale)
    F2 = clamp(F2, FT(1e-3), one(FT))
    # β_bare = W / (W_crit_frac · W_max). F3 maps wilting (W_wilt/W_crit
    # in β_bare units) → 1e-3 and saturation (β_bare ≥ 1) → 1.
    wilt_over_crit = W_wilt_frac / W_crit_frac
    F3 = clamp((β_bare - wilt_over_crit) / (one(FT) - wilt_over_crit), FT(1e-3), one(FT))
    F4 = one(FT) - FT(0.0016) * (T_opt - Ta)^2; F4 = clamp(F4, FT(1e-3), one(FT))
    return clamp(F1 * F2 * F3 * F4, FT(1e-3), one(FT))
end

@kernel function _bucket_with_snow_diagnostics!(snow_fraction, β_out,
                                                 W, SWE, T, p_surf, qa, Ta, rg,
                                                 vegfrac, lai,
                                                 W_max, W_crit_frac, W_wilt_frac, SWE_crit,
                                                 solar_scale, vpd_scale, T_opt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(β_out)
        f_sn = tanh(SWE[i, j, 1] / SWE_crit)
        snow_fraction[i, j, 1] = f_sn

        β_bare = clamp(W[i, j, 1] / (W_crit_frac * W_max), zero(FT), one(FT))
        vf = vegfrac[i, j, 1]
        L  = lai[i, j, 1]

        ps = p_surf[i, j, 1] > one(FT) ? p_surf[i, j, 1] : FT(1013.25)
        J = _jarvis_factor(rg[i, j, 1], qa[i, j, 1], Ta[i, j, 1], ps,
                           β_bare, L,
                           solar_scale, vpd_scale, T_opt,
                           W_crit_frac, W_wilt_frac)

        β_canopy = β_bare * J
        β_snow_free = (one(FT) - vf) * β_bare + vf * β_canopy
        β_out[i, j, 1] = (one(FT) - f_sn) * β_snow_free + f_sn
    end
end

function update_diagnostics!(b::BucketWithSnow, state, fluxes, surface, parameters, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _bucket_with_snow_diagnostics!,
            state.snow_fraction, state.moisture_availability,
            state.W, state.SWE, state.T,
            fluxes.surface_pressure, fluxes.air_humidity, fluxes.air_temperature,
            fluxes.solar_irradiance,
            surface.vegfrac, surface.lai,
            b.W_max, b.W_crit_frac, b.W_wilt_frac, b.SWE_crit,
            b.solar_radiation_scale, b.vpd_scale, b.T_opt)
    return nothing
end

Base.summary(b::BucketWithSnow{FT}) where FT =
    string("BucketWithSnow{$FT}(W_max=", b.W_max,
           " kg/m², SWE_crit=", b.SWE_crit, " m)")
