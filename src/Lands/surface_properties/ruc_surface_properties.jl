#####
##### `RucSurfaceProperties` — RUC-style snow / vegetation surface
##### property closure.
#####
##### Owns three end-member sets blended by `vegfrac` and then by
##### `snowfrac`:
#####   - scalar snow end-members (`alb_snow`, `emiss_snow`, `z0_snow`)
#####   - scalar bare-soil end-members (`alb_bare`, `emiss_bare`, `z0_bare`)
#####   - per-cell vegetation end-members (`albedo_veg`, `emissivity_veg`,
#####     `z0_veg`) — typically populated from a USGS / MODIS lookup
#####     table via `apply_land_classifications!`
#####
##### Also owns the per-cell vegetation/land-class fields
##### (`vegfrac`, `lai`, `r_smin`, `is_urban`) that `RucHydrology`
##### reads through the `surface` argument to its kernels — these are
##### "shared surface properties" in the sense of the design doc's §2
##### invariant #4.
#####
##### Output Fields (`alb`, `emiss`, `znt`) are refreshed by
##### `update_diagnostics!` and exposed through the
##### `albedo` / `emissivity` / `momentum_roughness_length` /
##### `scalar_roughness_length` accessors.
#####
##### References:
#####   Robinson, D. A., and G. Kukla, 1985: Maximum surface albedo of
#####     seasonally snow-covered lands in the Northern Hemisphere.
#####     J. Climate Appl. Meteor., 24, 402–411.
#####   Smirnova, T. G., J. M. Brown, and S. G. Benjamin, 1997:
#####     Performance of different soil model configurations in
#####     simulating ground surface temperature and surface fluxes.
#####     Mon. Wea. Rev., 125, 1870–1884.
#####   Smirnova, T. G., J. M. Brown, S. G. Benjamin, and J. S. Kenyon,
#####     2016: Modifications to the Rapid Update Cycle Land Surface
#####     Model (RUC LSM) available in the WRF model. Mon. Wea. Rev.,
#####     144, 1851–1865, doi:10.1175/MWR-D-15-0198.1.
#####

"""
    RucSurfaceProperties(grid, parameters::RucSlabLandParameters)

Allocate the per-cell vegetation config + output Fields and wrap them
together with the scalar snow / bare-soil end-members from
`parameters`. The vegetation fields are filled from the scalar
parameters by default; call `apply_land_classifications!(land, …)` to
populate them from a USGS / MODIS lookup table.
"""
struct RucSurfaceProperties{FT, F} <: AbstractSurfaceProperties
    # Scalar snow / bare-soil end-members
    alb_snow     :: FT
    alb_bare     :: FT
    emiss_snow   :: FT
    emiss_bare   :: FT
    z0_snow      :: FT
    z0_bare      :: FT
    # Per-cell vegetation end-members and land-class fields
    vegfrac        :: F
    lai            :: F
    albedo_veg     :: F
    emissivity_veg :: F
    z0_veg         :: F
    r_smin         :: F
    is_urban       :: F
    # Output Fields populated by `update_diagnostics!`
    alb   :: F
    emiss :: F
    znt   :: F
end

function RucSurfaceProperties(grid, p::RucSlabLandParameters{FT}) where FT
    vegfrac        = CenterField(grid)
    lai            = CenterField(grid)
    albedo_veg     = CenterField(grid); fill!(albedo_veg,     p.alb_veg)
    emissivity_veg = CenterField(grid); fill!(emissivity_veg, p.emiss_veg)
    z0_veg         = CenterField(grid); fill!(z0_veg,         p.z0_veg)
    r_smin         = CenterField(grid); fill!(r_smin,         p.r_smin)
    is_urban       = CenterField(grid)

    alb   = CenterField(grid); fill!(alb,   p.alb_bare)
    emiss = CenterField(grid); fill!(emiss, p.emiss_bare)
    znt   = CenterField(grid); fill!(znt,   p.z0_bare)

    return RucSurfaceProperties{FT, typeof(alb)}(
        p.alb_snow, p.alb_bare,
        p.emiss_snow, p.emiss_bare,
        p.z0_snow, p.z0_bare,
        vegfrac, lai, albedo_veg, emissivity_veg, z0_veg, r_smin, is_urban,
        alb, emiss, znt)
end

prognostic_variables(::RucSurfaceProperties) = ()
flux_variables(::RucSurfaceProperties)       = ()

#####
##### Property-blending kernel
#####

@kernel function _update_surface_properties!(alb, emiss, znt,
                                             snowfrac, keep_snow_albedo,
                                             T, newsn, snhei,
                                             vegfrac,
                                             albedo_veg, emissivity_veg, z0_veg_field,
                                             alb_snow, alb_bare,
                                             emiss_snow, emiss_bare,
                                             z0_snow, z0_bare)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT   = eltype(alb)
        f_sn = snowfrac[i, j, 1]
        keep = keep_snow_albedo[i, j, 1]
        h    = snhei[i, j, 1]
        vf   = vegfrac[i, j, 1]
        α_sn = FT(alb_snow)

        # Composite snow-free end-members weighted by vegfrac.
        # The `_veg` end-members are per-cell Fields populated either
        # from scalar parameters or from a vegetation lookup table.
        α_veg_ij  = albedo_veg[i, j, 1]
        ε_veg_ij  = emissivity_veg[i, j, 1]
        z₀_veg_ij = z0_veg_field[i, j, 1]
        α_free  = (one(FT) - vf) * alb_bare   + vf * α_veg_ij
        ε_free  = (one(FT) - vf) * emiss_bare + vf * ε_veg_ij
        z₀_free = (one(FT) - vf) * z0_bare    + vf * z₀_veg_ij

        # Roughness: default to the snow-free composite. RUC's
        # snow-blending branch only applies when the surface is short
        # (z₀_free ≤ 0.2 m, i.e. grass / shrub / cropland) and there is
        # no fresh snow this step; tall canopies (forests with
        # z₀_free > 0.2 m) keep the canopy roughness.
        znt_new = z₀_free
        if h > zero(FT) && newsn[i, j, 1] == zero(FT) && z₀_free ≤ FT(0.2)
            if h ≤ FT(2) * z₀_free
                znt_new = FT(0.55) * z₀_free + FT(0.45) * z0_snow
            elseif h ≤ FT(4) * z₀_free
                znt_new = FT(0.2) * z₀_free + FT(0.8) * z0_snow
            else
                znt_new = z0_snow
            end
        end
        znt[i, j, 1] = znt_new

        # Shortwave-albedo blend (Robinson and Kukla 1985), with the
        # warm-T attenuation branch below 0.4 disabled when fresh snow has
        # latched `keep_snow_albedo = 1`.
        α_blend = max(keep * α_sn,
                      min(α_free + (α_sn - α_free) * f_sn, α_sn))
        if α_blend < FT(0.4) || keep == one(FT)
            alb[i, j, 1] = α_blend
        else
            T_C = T[i, j, 1] - FT(273.15)
            alb[i, j, 1] = min(α_blend,
                               max(α_blend - FT(0.1) * (T_C + FT(10)) /
                                             FT(10) * α_blend,
                                   α_blend - FT(0.05)))
        end

        emiss[i, j, 1] = max(keep * emiss_snow,
                             min(ε_free + (emiss_snow - ε_free) * f_sn,
                                 emiss_snow))
    end
end

function update_diagnostics!(s::RucSurfaceProperties, state, fluxes, surface, parameters, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _update_surface_properties!,
            s.alb, s.emiss, s.znt,
            state.snowfrac, state.keep_snow_albedo,
            state.T, state.newsn, state.snhei,
            s.vegfrac,
            s.albedo_veg, s.emissivity_veg, s.z0_veg,
            s.alb_snow, s.alb_bare,
            s.emiss_snow, s.emiss_bare,
            s.z0_snow, s.z0_bare)
    return nothing
end

#####
##### Atmosphere-facing accessors
#####

albedo(s::RucSurfaceProperties, state, parameters)                    = s.alb
emissivity(s::RucSurfaceProperties, state, parameters)                = s.emiss
momentum_roughness_length(s::RucSurfaceProperties, state, parameters) = s.znt
# RUC follows Garratt 1992: scalar roughness is `znt / 10`. Keep that
# convention here so the atmosphere's MOST scalar-roughness consumer
# (`LandRoughnessLength(multiplier=0.1)`) remains numerically equivalent.
scalar_roughness_length(s::RucSurfaceProperties, state, parameters) = s.znt

Base.summary(::RucSurfaceProperties{FT}) where FT = "RucSurfaceProperties{$FT}"
