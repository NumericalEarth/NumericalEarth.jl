#####
##### `RucHydrology` — RUC-LSM-style hydrology closure for `SlabLand`.
#####
##### Owns the snow + canopy-water + soil-moisture state and closes the
##### water budget every time step. Implements the slab-compatible
##### subset of the RUC LSM (Smirnova et al. 1997, 2016) plus
##### complementary parameterizations from ISBA (Noilhan-Planton 1989,
##### Mahfouf-Noilhan 1991), Clapp-Hornberger (1978), Anderson (1976),
##### Koren et al. (1999), Niu-Yang (2007), and Jarvis-Stewart
##### (Jarvis 1976, Stewart 1988).
#####
##### Cross-axis writes: this closure mutates `state.T` to close the
##### latent-heat budget of snow melt (`_melt_snow!`) and soil
##### freeze/thaw (`_freeze_thaw_soil!`). These are conservative
##### exchanges between the water and energy reservoirs that physically
##### belong to the hydrology step; the design doc's "no
##### closure-to-closure direct reads" invariant is preserved because
##### the channel is the shared `state` `NamedTuple`, not a reference to
##### the energy closure.
#####
##### References:
#####   Anderson, E. A., 1976: A point energy and mass balance model of
#####     a snow cover. NOAA Tech. Rep. NWS 19.
#####   Manabe, S., 1969: Climate and the ocean circulation: I. The
#####     atmospheric circulation and the hydrology of the Earth's
#####     surface. Mon. Wea. Rev., 97, 739–774.
#####   Jarvis, P. G., 1976: The interpretation of the variations in
#####     leaf water potential and stomatal conductance found in
#####     canopies in the field. Phil. Trans. R. Soc. London B, 273,
#####     593–610.
#####   Clapp, R. B., and G. M. Hornberger, 1978: Empirical equations
#####     for some soil hydraulic properties. Water Resour. Res., 14,
#####     601–604.
#####   Buck, A. L., 1981: New equations for computing vapor pressure
#####     and enhancement factor. J. Appl. Meteor., 20, 1527–1532.
#####   Stewart, J. B., 1988: Modelling surface conductance of pine
#####     forest. Agric. For. Meteor., 43, 19–35.
#####   Noilhan, J., and S. Planton, 1989: A simple parameterization of
#####     land surface processes for meteorological models. Mon. Wea.
#####     Rev., 117, 536–549.
#####   Mahfouf, J.-F., and J. Noilhan, 1991: Comparative study of
#####     various formulations of evaporation from bare soil using in
#####     situ data. J. Appl. Meteor., 30, 1354–1365.
#####   Smirnova, T. G., J. M. Brown, and S. G. Benjamin, 1997:
#####     Performance of different soil model configurations in
#####     simulating ground surface temperature and surface fluxes.
#####     Mon. Wea. Rev., 125, 1870–1884.
#####   Koren, V., J. Schaake, K. Mitchell, Q.-Y. Duan, F. Chen, and
#####     J. M. Baker, 1999: A parameterization of snowpack and frozen
#####     ground intended for NCEP weather and climate models. J.
#####     Geophys. Res., 104, 19569–19585.
#####   Niu, G.-Y., and Z.-L. Yang, 2007: An observation-based
#####     formulation of snow cover fraction and its evaluation over
#####     large North American river basins. J. Geophys. Res., 112,
#####     D21101, doi:10.1029/2007JD008674.
#####   Smirnova, T. G., J. M. Brown, S. G. Benjamin, and J. S. Kenyon,
#####     2016: Modifications to the Rapid Update Cycle Land Surface
#####     Model (RUC LSM) available in the WRF model. Mon. Wea. Rev.,
#####     144, 1851–1865, doi:10.1175/MWR-D-15-0198.1.
#####

"""
    RucHydrology(parameters::RucSlabLandParameters)

Build a `RucHydrology` from the RUC parameter bag. All snow, canopy,
soil-moisture, freeze/thaw, and Jarvis-Stewart resistance parameters
live on the closure itself. The ground areal heat capacity
`(ρ c H)_g` is duplicated here from the energy closure so this
closure's snow-melt / freeze-thaw kernels can close the latent-heat
budget without a direct reference to `RucEnergy`.
"""
struct RucHydrology{FT} <: AbstractHydrology
    # Soil bucket
    soil_depth     :: FT
    theta_saturation      :: FT
    theta_field_capacity       :: FT
    theta_wilting_point     :: FT
    theta_air_dry  :: FT
    # Soil hydraulics (Clapp-Hornberger)
    psi_saturation        :: FT
    clapp_hornberger_exponent           :: FT
    # Snow-cover fraction
    snow_cover_scale_factor       :: FT
    snow_cover_option   :: Int
    # Snow density compaction
    c1_compaction  :: FT
    c2_compaction  :: FT
    snow_density_min      :: FT
    snow_density_max      :: FT
    # Snow liquid retention + phase change
    snow_liquid_capacity_fraction :: FT
    melt_factor               :: FT
    snow_retention_min_fraction  :: FT
    snow_retention_max_fraction  :: FT
    snow_retention_depth_scale  :: FT
    snow_retention_depth_factor :: FT
    # Phase-change and water properties
    latent_heat_fusion :: FT
    density_of_water   :: FT
    # Canopy
    canopy_water_capacity :: FT
    # Jarvis-Stewart resistances
    stomatal_resistance_min   :: FT
    stomatal_resistance_max   :: FT
    solar_radiation_limit   :: FT
    vapor_pressure_deficit_limit  :: FT
    temperature_optimum    :: FT
    # Ground areal heat capacity (duplicated from RucEnergy so the
    # snow-melt / freeze-thaw kernels can close the latent-heat budget).
    ground_heat_capacity :: FT
end

function RucHydrology(p::RucSlabLandParameters{FT}) where FT
    ρcH_g = p.density * p.heat_capacity * p.depth
    return RucHydrology{FT}(p.soil_depth, p.theta_saturation, p.theta_field_capacity,
                            p.theta_wilting_point, p.theta_air_dry,
                            p.psi_saturation, p.clapp_hornberger_exponent,
                            p.snow_cover_scale_factor, p.snow_cover_option,
                            p.c1_compaction, p.c2_compaction,
                            p.snow_density_min, p.snow_density_max,
                            p.snow_liquid_capacity_fraction, p.melt_factor,
                            p.snow_retention_min_fraction, p.snow_retention_max_fraction,
                            p.snow_retention_depth_scale, p.snow_retention_depth_factor,
                            p.latent_heat_fusion, FT(1000),
                            p.canopy_water_capacity,
                            p.stomatal_resistance_min, p.stomatal_resistance_max, p.solar_radiation_limit, p.vapor_pressure_deficit_limit, p.temperature_optimum,
                            ρcH_g)
end

#####
##### Prognostic state and flux declarations
#####

const _RUC_HYDROLOGY_STATE_KEYS = (
    # Soil moisture
    :θ, :θ_ice,
    # Snow column
    :snwe, :snhei, :rhosn, :rhonewsn, :rhosn_step_start,
    :snowfrac, :newsn, :snowfallac, :snowfracnewsn, :keep_snow_albedo,
    :swl, :swe_inflow, :swl_overflow,
    # Canopy
    :cst, :drip, :interw, :intersn, :infwater, :intwratio, :canopy_capacity,
    # Resistance / wetness diagnostics
    :moisture_availability, :soil_resistance, :stomatal_resistance,
)

const _RUC_HYDROLOGY_FLUX_KEYS = (
    :snowfall_rate, :rainfall_rate, :moisture_flux,
    :canopy_evaporation, :transpiration,
    :solar_irradiance, :air_temperature, :air_humidity, :surface_pressure,
)

prognostic_variables(::RucHydrology) = _RUC_HYDROLOGY_STATE_KEYS
flux_variables(::RucHydrology)       = _RUC_HYDROLOGY_FLUX_KEYS

function initial_state(h::RucHydrology, name::Symbol, grid)
    f = CenterField(grid)
    if name === :θ
        fill!(f, h.theta_field_capacity)
    elseif name === :rhosn || name === :rhosn_step_start
        fill!(f, 250)
    elseif name === :rhonewsn
        fill!(f, 100)
    elseif name === :soil_resistance
        fill!(f, 1)
    elseif name === :stomatal_resistance
        fill!(f, h.stomatal_resistance_min)
    end
    return f
end

#####
##### Snow physics — kernels (relocated from `ruc_slab_land.jl`)
#####

# Snow-cover fraction (Koren et al. 1999, Niu and Yang 2007). Mirrors
# RUC LSM `compute_snow_fraction`.
@inline function compute_snow_fraction(opt::Int, snhei, snhei_crit, roughness_length,
                                       rhosn, rhonewsn, snow_cover_scale_factor)
    FT = typeof(snhei)
    if opt == 1
        return min(one(FT), snhei / (FT(2) * snhei_crit))
    elseif opt == 2
        f1 = min(one(FT), snhei / (FT(2) * snhei_crit))
        z₀ = min(FT(0.2), roughness_length)
        f2 = tanh(snhei / (FT(2.5) * z₀ * (rhosn / rhonewsn)))
        return FT(0.5) * (f1 + f2)
    else
        return tanh(snhei / (FT(10) * snow_cover_scale_factor * (rhosn / rhonewsn)))
    end
end

# Per-cell snapshot of the start-of-step `rhosn`. RUC LSM evaluates
# `snhei_crit_newsn` at the start-of-step density, before compaction or
# the new-snow density blend mutate it; this snapshot lets
# `_accumulate_new_snow!` reproduce that ordering.
@kernel function _snapshot_rhosn!(dst, src)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j, 1] = src[i, j, 1]
end

# Snow density compaction following Anderson (1976).
@kernel function _compact_snow!(rhosn, snwe, snhei, swl, T,
                                Δt, c1, c2, ρ_min, ρ_max)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(rhosn)
        ρ  = max(rhosn[i, j, 1], ρ_min)
        h  = snhei[i, j, 1]
        total_swe = snwe[i, j, 1] + swl[i, j, 1]
        if h > FT(0.0081) * FT(1000) / ρ
            T_C = min(zero(FT), T[i, j, 1] - FT(273.15))
            bsn = (FT(Δt) / FT(3600)) * c1 *
                  exp(FT(0.08) * T_C - c2 * ρ * FT(1e-3))
            arg = bsn * total_swe * FT(100)
            if arg ≥ FT(1e-4)
                xsn = ρ * (exp(arg) - one(FT)) / arg
                rhosn[i, j, 1] = clamp(xsn, ρ_min, ρ_max)
            end
        end
        snhei[i, j, 1] = total_swe > zero(FT) ?
                         total_swe * FT(1000) / max(rhosn[i, j, 1], ρ_min) :
                         zero(FT)
    end
end

# Atmosphere-supplied vapor flux applied to the snow-covered fraction of
# the pack, regardless of pack temperature. RUC LSM applies the
# `(snwe -= β·epot·ras·Δt)` mass sink in addition to any melt sink; the
# associated latent heat enters the slab through `temperature_flux`
# upstream, so this kernel only updates mass.
@kernel function _apply_sublimation!(snwe, snhei, swl, F_v, snowfrac, rhosn,
                                     Δt, ρ_w, ρ_min)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snwe)
        if snwe[i, j, 1] > zero(FT) && snowfrac[i, j, 1] > zero(FT)
            Δswe = F_v[i, j, 1] * snowfrac[i, j, 1] * FT(Δt) / ρ_w
            snwe[i, j, 1]  = max(zero(FT), snwe[i, j, 1] - Δswe)
            snhei[i, j, 1] = (snwe[i, j, 1] + swl[i, j, 1]) * FT(1000) /
                             max(rhosn[i, j, 1], ρ_min)
        end
    end
end

# Snow melt when `T_g > 273.15 K`. Slab heat surplus → melt mass, with
# RUC Egglston cap and retained-melt liquid split. Slab cools by exactly
# the latent heat consumed.
@kernel function _melt_snow!(snwe, snhei, swl, swl_overflow, T,
                             rhosn, newsn, rhonewsn,
                             ρcH_ground, L_f, ρ_w, ρ_min, ρ_max,
                             Δt, melt_factor,
                             rsm_min_frac, rsm_max_frac,
                             rsm_depth_scale, rsm_depth_factor)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snwe)
        if snwe[i, j, 1] > zero(FT) && T[i, j, 1] > FT(273.15)
            snwepr = snwe[i, j, 1]
            snwepr_total = snwepr + swl[i, j, 1]
            ρ_sn = max(rhosn[i, j, 1], ρ_min)
            Δenergy = ρcH_ground * (T[i, j, 1] - FT(273.15))   # J m⁻²
            Δmass_max = Δenergy / L_f                          # kg m⁻²
            Δswe_max  = Δmass_max / ρ_w                        # m LWE

            smelt = Δswe_max / FT(Δt)
            if (ρ_sn < FT(350) ||
                (newsn[i, j, 1] > zero(FT) && rhonewsn[i, j, 1] < FT(450))) &&
               T[i, j, 1] < FT(283)
                smelt_cap = FT(Δt) / FT(60) * FT(5.6e-8) *
                            melt_factor * max(one(FT), T[i, j, 1] - FT(273.15))
                smelt = min(smelt, smelt_cap)
            end

            Δswe = min(smelt * FT(Δt), snwepr)
            rsmfrac = zero(FT)
            if snhei[i, j, 1] > FT(0.01) && ρ_sn < FT(350)
                rsmfrac = min(rsm_max_frac,
                              max(rsm_min_frac,
                                   snwepr_total / rsm_depth_scale * rsm_depth_factor))
            end

            retained = rsmfrac * Δswe
            overflow = Δswe - retained

            snwe[i, j, 1] = max(zero(FT), snwepr - Δswe)
            swl[i, j, 1] += retained
            swl_overflow[i, j, 1] += overflow
            T[i, j, 1]     -= Δswe * ρ_w * L_f / ρcH_ground

            # RUC folds retained liquid into `snwe`; the split Julia state
            # must still mass-average the full pack and use that total for
            # snow height. Only the freshly retained liquid from this step
            # (`retained`) blends in at `ρ_w`; previously retained `swl` is
            # already folded into the bulk `ρ_sn`. This is RUC's
            #     `xsn = (rhosn*(snwe-rsm) + 1.e3*rsm) / snwe`
            # rewritten under the split SWE/SWL representation.
            total_pack_swe = snwe[i, j, 1] + swl[i, j, 1]
            if total_pack_swe > zero(FT)
                solid_plus_old_liq = max(zero(FT),
                                         total_pack_swe - retained)
                xsn = (ρ_sn * solid_plus_old_liq + ρ_w * retained) /
                      total_pack_swe
                rhosn[i, j, 1] = clamp(xsn, ρ_min, ρ_max)
            end

            # A single-T slab has no separate sub-snow soil temperature.
            snhei[i, j, 1] = total_pack_swe * FT(1000) /
                             max(rhosn[i, j, 1], ρ_min)
        end
    end
end

# Drain carried-over liquid water in the slab pack to the soil bucket
# once it exceeds the configured capacity.
@kernel function _drain_swl!(swl, snwe, swl_overflow, retention_frac)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(swl)
        cap = retention_frac * snwe[i, j, 1]
        if swl[i, j, 1] > cap
            swl_overflow[i, j, 1] = swl[i, j, 1] - cap
            swl[i, j, 1]          = cap
        else
            swl_overflow[i, j, 1] = zero(FT)
        end
    end
end

# Add new snowfall (post-canopy throughfall in `swe_inflow`) to the pack.
@kernel function _accumulate_new_snow!(snwe, snhei, rhosn, swl, rhonewsn,
                                       newsn, snowfracnewsn, keep_snow_albedo,
                                       snowfallac,
                                       swe_inflow, T_air, rhosn_step_start,
                                       ρ_min, ρ_max)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snwe)
        Δswe = max(zero(FT), swe_inflow[i, j, 1])      # already × Δt

        if Δswe > zero(FT)
            T_K = T_air[i, j, 1]
            ρ_new = clamp(FT(1000) /
                           max(FT(8), FT(17) * tanh((FT(276.65) - T_K) * FT(0.15))),
                           ρ_min, FT(125))
            rhonewsn[i, j, 1]  = ρ_new

            old_swe = snwe[i, j, 1] + swl[i, j, 1]
            old_ρ   = max(rhosn[i, j, 1], ρ_min)
            total   = old_swe + Δswe

            # SWE-weighted bulk-density blend uses the post-canopy `Δswe`,
            # consistent with the actual mass added to `snwe`. (RUC LSM
            # uses the raw pre-canopy `newsn` for this blend, which over-
            # counts mass relative to its own subsequent `snwe += newsn -
            # intersn` update; the Julia formulation is internally mass-
            # balanced.)
            xsn = (old_ρ * old_swe + ρ_new * Δswe) / total
            rhosn[i, j, 1] = clamp(xsn, ρ_min, ρ_max)

            snwe[i, j, 1] += Δswe
            snhei[i, j, 1] = (snwe[i, j, 1] + swl[i, j, 1]) * FT(1000) /
                             max(rhosn[i, j, 1], ρ_min)

            new_depth = Δswe * FT(1000) / ρ_new
            newsn[i, j, 1] = new_depth

            # `snhei_crit_newsn = 0.0005·ρ_w/ρ_sn` is evaluated at the
            # **pre-compaction, pre-blend** `ρ_sn` snapshot taken at the
            # start of the time step. RUC LSM captures it once at the top
            # of its surface routine before any in-step modifications.
            ρ_pre = max(rhosn_step_start[i, j, 1], ρ_min)
            snhei_crit_newsn_dyn = FT(0.0005) * FT(1000) / ρ_pre

            # `snowfracnewsn` is evaluated against the **pre-increment**
            # `snowfallac`; the increment `snowfallac += newsn` happens
            # afterwards. This ordering keeps `keep_snow_albedo` from
            # latching one step too early when `snowfallac` first crosses
            # the `snhei_crit_newsn` threshold.
            snowfracnewsn[i, j, 1] = min(one(FT),
                                         snowfallac[i, j, 1] / snhei_crit_newsn_dyn)
            snowfallac[i, j, 1]   += new_depth

            keep_snow_albedo[i, j, 1] = (snowfracnewsn[i, j, 1] > FT(0.99) &&
                                         ρ_new < FT(450)) ? one(FT) : zero(FT)
        end
    end
end

# Recompute snow fraction from the post-step pack state.
@kernel function _finalize_snow_cover!(snowfrac, snhei, roughness_length, rhosn, rhonewsn,
                                       is_urban, snow_cover_scale_factor, snow_cover_option)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snowfrac)
        if snhei[i, j, 1] == zero(FT)
            snowfrac[i, j, 1] = zero(FT)
        else
            ρ_sn  = max(rhosn[i, j, 1],    FT(58.8))
            ρ_new = max(rhonewsn[i, j, 1], FT(58.8))
            snhei_crit_dyn = FT(0.01601) * FT(1000) / ρ_sn
            f = compute_snow_fraction(snow_cover_option,
                                      snhei[i, j, 1], snhei_crit_dyn,
                                      roughness_length[i, j, 1],
                                      ρ_sn, ρ_new, snow_cover_scale_factor)
            if is_urban[i, j, 1] > FT(0.5)
                f = min(f, FT(0.75))
            end
            snowfrac[i, j, 1] = f
        end
    end
end

#####
##### Canopy water balance
#####

@inline canopy_capacity(::FT, ::FT, capacity::FT) where FT = capacity

@kernel function _intercept_precip!(cst, drip, interw, intersn,
                                    infwater, intwratio,
                                    canopy_cap, vegfrac, lai, snowfrac,
                                    rainfall_rate, snowfall_rate,
                                    swe_inflow, Δt, canopy_water_capacity)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(cst)
        vf = vegfrac[i, j, 1]
        L  = lai[i, j, 1]
        sat = canopy_capacity(L, vf, FT(canopy_water_capacity))
        canopy_cap[i, j, 1] = sat

        Δrain = max(zero(FT), rainfall_rate[i, j, 1] * FT(Δt))
        Δsnow = max(zero(FT), snowfall_rate[i, j, 1] * FT(Δt))

        if vf > FT(0.01)
            transmission = one(FT) - exp(-FT(0.5) * L)
            iw = FT(0.25) * Δrain * transmission * vf
            is = FT(0.25) * Δsnow * transmission * vf
            interw[i, j, 1]  = iw
            intersn[i, j, 1] = is

            ratio = (iw + is) > zero(FT) ? iw / (iw + is) : zero(FT)
            intwratio[i, j, 1] = ratio

            cst_new = cst[i, j, 1] + iw + is
            if cst_new > sat
                drip[i, j, 1] = cst_new - sat
                cst[i, j, 1]  = sat
            else
                drip[i, j, 1] = zero(FT)
                cst[i, j, 1]  = cst_new
            end

            d = drip[i, j, 1]
            # Drip routing branches on snow coverage:
            #   snow_mosaic == 1 (snowfrac < 0.75) splits drip by intwratio
            #   between liquid throughfall and snow pack inflow;
            #   snow_mosaic == 0 (snowfrac ≥ 0.75) sends ALL drip to the
            #   snow pack since the surface is essentially uniform snow.
            if snowfrac[i, j, 1] < FT(0.75)
                infwater[i, j, 1]   = max(zero(FT), Δrain - iw) + d * ratio
                swe_inflow[i, j, 1] = max(zero(FT), Δsnow - is) +
                                      d * (one(FT) - ratio)
            else
                infwater[i, j, 1]   = max(zero(FT), Δrain - iw)
                swe_inflow[i, j, 1] = max(zero(FT), Δsnow - is) + d
            end
        else
            cst[i, j, 1]        = zero(FT)
            drip[i, j, 1]       = zero(FT)
            interw[i, j, 1]     = zero(FT)
            intersn[i, j, 1]    = zero(FT)
            intwratio[i, j, 1]  = zero(FT)
            infwater[i, j, 1]   = Δrain
            swe_inflow[i, j, 1] = Δsnow
        end
    end
end

@kernel function _evaporate_canopy!(cst, F_v_canopy, Δt, ρ_w)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(cst)
        if cst[i, j, 1] > zero(FT)
            Δ = max(zero(FT), F_v_canopy[i, j, 1] * FT(Δt) / ρ_w)
            cst[i, j, 1] = max(zero(FT), cst[i, j, 1] - Δ)
        end
    end
end

#####
##### Soil moisture, freeze/thaw, RUC `moisture_availability`/`soil_resistance`, Jarvis stomatal_resistance
#####

@inline function ruc_moisture_availability(θ, snowfrac, θ_air_dry, θ_fc)
    FT = typeof(θ)
    ref = max(θ_fc - θ_air_dry, eps(FT))
    residual = max(zero(FT), θ - θ_air_dry)
    return clamp(residual / ref * (one(FT) - snowfrac) + snowfrac,
                 FT(1e-5), one(FT))
end

@inline function ruc_soil_resistance(θ, qa, qg, θ_air_dry, θ_fc)
    FT = typeof(θ)
    fc = max(θ_air_dry, FT(0.5) * θ_fc)
    if θ > fc || qa > qg
        return one(FT)
    else
        fex_fc = clamp(θ / fc, FT(0.01), one(FT))
        return FT(0.25) * (one(FT) - cos(FT(π) * fex_fc))^2
    end
end

# Saturation specific humidity from Magnus / Buck (1981).
@inline function esat_buck(T)
    FT = typeof(T)
    if T > FT(273.15)
        return FT(6.1121) * exp(FT(17.502) * (T - FT(273.15)) / (T - FT(32.18)))
    else
        return FT(6.1115) * exp(FT(22.452) * (T - FT(273.15)) / (T - FT(0.61)))
    end
end

@inline function qsat_buck(T, p_hPa)
    FT = typeof(T)
    e = esat_buck(T)
    ε = FT(0.622)
    return ε * e / (p_hPa - (one(FT) - ε) * e)
end

@inline function jarvis_resistance(rg, qa, Ta, p_hPa, θ, lai,
                                   stomatal_resistance_min, stomatal_resistance_max, solar_radiation_limit, vapor_pressure_deficit_limit, temperature_optimum,
                                   θ_wilt, θ_fc)
    FT = typeof(rg)
    if lai ≤ zero(FT)
        return stomatal_resistance_max
    end
    F1 = (rg / solar_radiation_limit) / (one(FT) + rg / solar_radiation_limit);  F1 = max(F1, FT(1e-3))
    qsat_air = qsat_buck(Ta, p_hPa)
    F2 = one(FT) / (one(FT) + max(zero(FT), (qsat_air - qa)) / vapor_pressure_deficit_limit)
    F2 = clamp(F2, FT(1e-3), one(FT))
    if θ ≥ θ_fc
        F3 = one(FT)
    elseif θ ≤ θ_wilt
        F3 = FT(1e-3)
    else
        F3 = clamp((θ - θ_wilt) / (θ_fc - θ_wilt), FT(1e-3), one(FT))
    end
    F4 = one(FT) - FT(0.0016) * (temperature_optimum - Ta)^2;  F4 = clamp(F4, FT(1e-3), one(FT))
    return clamp(stomatal_resistance_min / (lai * F1 * F2 * F3 * F4), stomatal_resistance_min, stomatal_resistance_max)
end

@kernel function _update_moisture_availability_rs!(moisture_availability, soil_resistance, stomatal_resistance,
                                    soil_moisture, snowfrac,
                                    T, p_surf, vegfrac, lai,
                                    rg_irr, qa, Ta,
                                    stomatal_resistance_min_field,
                                    θ_wilt, θ_fc, θ_air_dry,
                                    stomatal_resistance_max, solar_radiation_limit, vapor_pressure_deficit_limit, temperature_optimum)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(moisture_availability)
        θ  = soil_moisture[i, j, 1]
        f_sn = snowfrac[i, j, 1]
        m  = ruc_moisture_availability(θ, f_sn, θ_air_dry, θ_fc)
        ps = p_surf[i, j, 1] > one(FT) ? p_surf[i, j, 1] : FT(1013.25)
        qg = qsat_buck(T[i, j, 1], ps) * m
        sr = ruc_soil_resistance(θ, qa[i, j, 1], qg, θ_air_dry, θ_fc)
        L  = lai[i, j, 1]
        moisture_availability[i, j, 1]  = m
        soil_resistance[i, j, 1] = sr
        stomatal_resistance[i, j, 1]    = jarvis_resistance(rg_irr[i, j, 1],
                                            qa[i, j, 1],
                                            Ta[i, j, 1], ps,
                                            θ, L,
                                            stomatal_resistance_min_field[i, j, 1], stomatal_resistance_max,
                                            solar_radiation_limit, vapor_pressure_deficit_limit,
                                            temperature_optimum, θ_wilt, θ_fc)
    end
end

@kernel function _step_soil_moisture!(soil_moisture, infwater, swl_overflow,
                                      F_v_total, transpiration,
                                      vegfrac, soil_resistance, snowfrac,
                                      Δt, ρ_w, soil_depth,
                                      θ_sat, θ_air_dry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(soil_moisture)
        vf = vegfrac[i, j, 1]
        snow_free = one(FT) - snowfrac[i, j, 1]

        Δθ_in = (infwater[i, j, 1] + swl_overflow[i, j, 1]) / soil_depth

        bare_share = snow_free * (one(FT) - vf) * soil_resistance[i, j, 1]
        E_g  = max(zero(FT), F_v_total[i, j, 1]) * bare_share
        E_t  = max(zero(FT), transpiration[i, j, 1])
        Δθ_out = (E_g + E_t) * FT(Δt) / (ρ_w * soil_depth)

        θ_new = soil_moisture[i, j, 1] + Δθ_in - Δθ_out
        soil_moisture[i, j, 1] = clamp(θ_new, θ_air_dry, θ_sat)
    end
end

# Equilibrium unfrozen water content `θ_liq*` at sub-zero soil
# temperature, from Clapp-Hornberger soil-water retention combined with
# Clausius-Clapeyron freezing-point depression.
@inline function unfrozen_liquid_eq(T, θ_sat, θ_air_dry,
                                    L_f, ψ_sat, clapp_hornberger_exponent)
    FT = typeof(T)
    if T ≥ FT(273.15)
        return θ_sat
    end
    g = FT(9.81)
    arg = L_f * (FT(273.15) - T) / (T * g * ψ_sat)
    base = θ_sat * arg^(-one(FT) / clapp_hornberger_exponent) - θ_air_dry
    return clamp(base, zero(FT), θ_sat)
end

@kernel function _freeze_thaw_soil!(θ_liq, θ_ice, T,
                                    ρcH_ground, L_f, ρ_w, soil_depth,
                                    θ_sat, θ_air_dry, ψ_sat, clapp_hornberger_exponent)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(θ_liq)
        Tg  = T[i, j, 1]
        θ_l = θ_liq[i, j, 1]
        θ_i = θ_ice[i, j, 1]
        θ_total = θ_l + θ_i

        # Equilibrium liquid-water target, clamped by total available water.
        θ_liq_eq = min(unfrozen_liquid_eq(Tg, θ_sat, θ_air_dry,
                                          L_f, ψ_sat, clapp_hornberger_exponent),
                       θ_total)

        # Δθ_freeze > 0 ⇒ freeze; < 0 ⇒ thaw.
        Δθ_freeze = θ_l - θ_liq_eq

        # Cap by available water reservoir.
        Δθ_freeze = clamp(Δθ_freeze, -θ_i, θ_l)

        # Cap by slab heat budget so the exchange relaxes `T_g` toward
        # `T_f = 273.15 K` and never overshoots. Energy released by the
        # phase change: `ΔE = Δθ_freeze · L_f · ρ_w · H_s [J m⁻²]`; slab
        # temperature response: `ΔT = ΔE / ρcH_g`. The admissible range
        # is therefore `|Δθ_freeze| ≤ Δθ_cap`, with sign restricted to the
        # direction that drives `T_g` toward `T_f`:
        #   `T_g < T_f` ⇒ only freeze (Δθ ≥ 0);
        #   `T_g > T_f` ⇒ only thaw (Δθ ≤ 0);
        #   `T_g = T_f` ⇒ no exchange.
        # The opposite direction (e.g. spontaneous thaw at sub-zero) is
        # an implicit transient that a simultaneous T-θ solve would
        # handle, but for the explicit split step here it is suppressed
        # for numerical stability.
        ΔT_avail = abs(FT(273.15) - Tg)
        Δθ_cap   = ρcH_ground * ΔT_avail / (L_f * ρ_w * soil_depth)

        if Tg < FT(273.15)
            Δθ_freeze = clamp(Δθ_freeze, zero(FT),  Δθ_cap)
        elseif Tg > FT(273.15)
            Δθ_freeze = clamp(Δθ_freeze, -Δθ_cap, zero(FT))
        else
            Δθ_freeze = zero(FT)
        end

        θ_liq[i, j, 1] = θ_l - Δθ_freeze
        θ_ice[i, j, 1] = θ_i + Δθ_freeze
        T[i, j, 1]     = Tg + Δθ_freeze * L_f * ρ_w * soil_depth / ρcH_ground
    end
end

#####
##### `step!` — full RUC water-balance sequence
#####

function step!(h::RucHydrology, state, fluxes, surface, parameters, grid, Δt)
    arch = architecture(grid)
    FT   = eltype(grid)
    ρ_w  = h.density_of_water
    L_f  = h.latent_heat_fusion
    ρcH_g = h.ground_heat_capacity

    # 1 — Drain retained liquid from previous steps above capacity.
    launch!(arch, grid, :xy, _drain_swl!,
            state.swl, state.snwe, state.swl_overflow,
            h.snow_liquid_capacity_fraction)

    # Snapshot start-of-step `rhosn` for `snhei_crit_newsn`.
    launch!(arch, grid, :xy, _snapshot_rhosn!,
            state.rhosn_step_start, state.rhosn)

    # 2 — Compaction of the existing pack, before current snowfall is added.
    launch!(arch, grid, :xy, _compact_snow!,
            state.rhosn, state.snwe, state.snhei, state.swl, state.T, Δt,
            h.c1_compaction, h.c2_compaction, h.snow_density_min, h.snow_density_max)

    # 3 — Canopy interception (drip routing depends on prior snowfrac).
    launch!(arch, grid, :xy, _intercept_precip!,
            state.cst, state.drip, state.interw, state.intersn,
            state.infwater, state.intwratio,
            state.canopy_capacity, surface.vegfrac, surface.lai,
            state.snowfrac,
            fluxes.rainfall_rate, fluxes.snowfall_rate,
            state.swe_inflow, Δt, h.canopy_water_capacity)

    # 4 — Wet-canopy direct evaporation.
    launch!(arch, grid, :xy, _evaporate_canopy!,
            state.cst, fluxes.canopy_evaporation, Δt, ρ_w)

    # 5 — New-snow accumulation.
    fill!(state.newsn, 0)
    fill!(state.snowfracnewsn, 0)
    fill!(state.keep_snow_albedo, 0)
    fill!(state.rhonewsn, 100)
    launch!(arch, grid, :xy, _accumulate_new_snow!,
            state.snwe, state.snhei, state.rhosn, state.swl, state.rhonewsn,
            state.newsn, state.snowfracnewsn, state.keep_snow_albedo,
            state.snowfallac,
            state.swe_inflow, fluxes.air_temperature, state.rhosn_step_start,
            h.snow_density_min, h.snow_density_max)

    # 6 — Sublimation / deposition over snow-covered fraction. Latent heat
    # already enters the slab heat budget through `temperature_flux`.
    launch!(arch, grid, :xy, _apply_sublimation!,
            state.snwe, state.snhei, state.swl, fluxes.moisture_flux,
            state.snowfrac, state.rhosn, Δt, ρ_w, h.snow_density_min)

    # 7 — Melt (warm pack), with RUC melt cap and retained-water split.
    launch!(arch, grid, :xy, _melt_snow!,
            state.snwe, state.snhei, state.swl, state.swl_overflow, state.T,
            state.rhosn, state.newsn, state.rhonewsn,
            ρcH_g, L_f, ρ_w, h.snow_density_min, h.snow_density_max,
            Δt, h.melt_factor,
            h.snow_retention_min_fraction,
            h.snow_retention_max_fraction,
            h.snow_retention_depth_scale,
            h.snow_retention_depth_factor)

    # 8 — Soil moisture update.
    launch!(arch, grid, :xy, _step_soil_moisture!,
            state.θ, state.infwater, state.swl_overflow,
            fluxes.moisture_flux, fluxes.transpiration,
            surface.vegfrac, state.soil_resistance, state.snowfrac,
            Δt, ρ_w, h.soil_depth,
            h.theta_saturation, h.theta_air_dry)

    # 9 — Soil freeze/thaw equilibration.
    launch!(arch, grid, :xy, _freeze_thaw_soil!,
            state.θ, state.θ_ice, state.T,
            ρcH_g, L_f, ρ_w, h.soil_depth,
            h.theta_saturation, h.theta_air_dry, h.psi_saturation, h.clapp_hornberger_exponent)

    return nothing
end

#####
##### `update_diagnostics!` — snow-cover fraction + Jarvis stomatal_resistance, moisture_availability, soil_resistance
#####

function update_diagnostics!(h::RucHydrology, state, fluxes, surface, parameters, grid)
    arch = architecture(grid)

    launch!(arch, grid, :xy, _finalize_snow_cover!,
            state.snowfrac, state.snhei, surface.roughness_length,
            state.rhosn, state.rhonewsn,
            surface.is_urban, h.snow_cover_scale_factor, h.snow_cover_option)

    launch!(arch, grid, :xy, _update_moisture_availability_rs!,
            state.moisture_availability, state.soil_resistance, state.stomatal_resistance,
            state.θ, state.snowfrac,
            state.T, fluxes.surface_pressure,
            surface.vegfrac, surface.lai,
            fluxes.solar_irradiance, fluxes.air_humidity, fluxes.air_temperature,
            surface.stomatal_resistance_min,
            h.theta_wilting_point, h.theta_field_capacity, h.theta_air_dry,
            h.stomatal_resistance_max, h.solar_radiation_limit, h.vapor_pressure_deficit_limit, h.temperature_optimum)

    return nothing
end

# Atmosphere-facing wetness factor.
wetness(::RucHydrology, state, parameters) = state.moisture_availability

Base.summary(h::RucHydrology{FT}) where FT =
    string("RucHydrology{$FT}(soil_depth=", h.soil_depth,
           " m, θ ∈ [", h.theta_wilting_point, ", ", h.theta_saturation, "])")
