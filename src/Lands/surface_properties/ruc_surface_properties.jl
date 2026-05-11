#####
##### `RucSurfaceProperties` — RUC-style snow / vegetation surface
##### property closure.
#####
##### Owns three end-member sets blended by `vegfrac` and then by
##### `snowfrac`:
#####   - scalar snow end-members (`albedo_snow`, `emissivity_snow`, `roughness_length_snow`)
#####   - scalar bare-soil end-members (`albedo_bare`, `emissivity_bare`, `roughness_length_bare`)
#####   - per-cell vegetation end-members (`albedo_vegetation`, `emissivity_vegetation`,
#####     `roughness_length_vegetation`) — typically populated from a USGS / MODIS lookup
#####     table via `apply_land_classifications!`
#####
##### Also owns the per-cell vegetation/land-class fields
##### (`vegfrac`, `lai`, `stomatal_resistance_min`, `is_urban`) that `RucHydrology`
##### reads through the `surface` argument to its kernels — these are
##### "shared surface properties" in the sense of the design doc's §2
##### invariant #4.
#####
##### Output Fields (`albedo`, `emissivity`, `roughness_length`) are
##### refreshed by `update_diagnostics!` and exposed through the
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
    albedo_snow                 :: FT
    albedo_bare                 :: FT
    emissivity_snow             :: FT
    emissivity_bare             :: FT
    roughness_length_snow       :: FT
    roughness_length_bare       :: FT
    # Per-cell vegetation end-members and land-class fields
    vegfrac                     :: F
    lai                         :: F
    albedo_vegetation           :: F
    emissivity_vegetation       :: F
    roughness_length_vegetation :: F
    stomatal_resistance_min     :: F
    is_urban                    :: F
    # Output Fields populated by `update_diagnostics!`
    albedo                      :: F
    emissivity                  :: F
    roughness_length            :: F
end

function RucSurfaceProperties(grid, p::RucSlabLandParameters{FT}) where FT
    vegfrac                     = CenterField(grid)
    lai                         = CenterField(grid)
    albedo_vegetation           = CenterField(grid); fill!(albedo_vegetation,           p.albedo_vegetation)
    emissivity_vegetation       = CenterField(grid); fill!(emissivity_vegetation,       p.emissivity_vegetation)
    roughness_length_vegetation = CenterField(grid); fill!(roughness_length_vegetation, p.roughness_length_vegetation)
    stomatal_resistance_min     = CenterField(grid); fill!(stomatal_resistance_min,     p.stomatal_resistance_min)
    is_urban                    = CenterField(grid)

    albedo           = CenterField(grid); fill!(albedo,           p.albedo_bare)
    emissivity       = CenterField(grid); fill!(emissivity,       p.emissivity_bare)
    roughness_length = CenterField(grid); fill!(roughness_length, p.roughness_length_bare)

    return RucSurfaceProperties{FT, typeof(albedo)}(
        p.albedo_snow, p.albedo_bare,
        p.emissivity_snow, p.emissivity_bare,
        p.roughness_length_snow, p.roughness_length_bare,
        vegfrac, lai,
        albedo_vegetation, emissivity_vegetation, roughness_length_vegetation,
        stomatal_resistance_min, is_urban,
        albedo, emissivity, roughness_length)
end

prognostic_variables(::RucSurfaceProperties) = ()
flux_variables(::RucSurfaceProperties)       = ()

#####
##### Property-blending kernel
#####

@kernel function _update_surface_properties!(albedo_field, emissivity_field, roughness_length_field,
                                             snowfrac, keep_snow_albedo,
                                             T, newsn, snhei,
                                             vegfrac,
                                             albedo_vegetation, emissivity_vegetation, roughness_length_vegetation_field,
                                             albedo_snow, albedo_bare,
                                             emissivity_snow, emissivity_bare,
                                             roughness_length_snow, roughness_length_bare)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT   = eltype(albedo_field)
        f_sn = snowfrac[i, j, 1]
        keep = keep_snow_albedo[i, j, 1]
        h    = snhei[i, j, 1]
        vf   = vegfrac[i, j, 1]
        α_sn = FT(albedo_snow)

        # Composite snow-free end-members weighted by vegfrac.
        # The vegetation end-members are per-cell Fields populated either
        # from scalar parameters or from a vegetation lookup table.
        α_veg_ij  = albedo_vegetation[i, j, 1]
        ε_veg_ij  = emissivity_vegetation[i, j, 1]
        z₀_veg_ij = roughness_length_vegetation_field[i, j, 1]
        α_free  = (one(FT) - vf) * albedo_bare   + vf * α_veg_ij
        ε_free  = (one(FT) - vf) * emissivity_bare + vf * ε_veg_ij
        z₀_free = (one(FT) - vf) * roughness_length_bare    + vf * z₀_veg_ij

        # Roughness: default to the snow-free composite. RUC's
        # snow-blending branch only applies when the surface is short
        # (z₀_free ≤ 0.2 m, i.e. grass / shrub / cropland) and there is
        # no fresh snow this step; tall canopies (forests with
        # z₀_free > 0.2 m) keep the canopy roughness.
        z₀_new = z₀_free
        if h > zero(FT) && newsn[i, j, 1] == zero(FT) && z₀_free ≤ FT(0.2)
            if h ≤ FT(2) * z₀_free
                z₀_new = FT(0.55) * z₀_free + FT(0.45) * roughness_length_snow
            elseif h ≤ FT(4) * z₀_free
                z₀_new = FT(0.2) * z₀_free + FT(0.8) * roughness_length_snow
            else
                z₀_new = roughness_length_snow
            end
        end
        roughness_length_field[i, j, 1] = z₀_new

        # Shortwave-albedo blend (Robinson and Kukla 1985), with the
        # warm-T attenuation branch below 0.4 disabled when fresh snow has
        # latched `keep_snow_albedo = 1`.
        α_blend = max(keep * α_sn,
                      min(α_free + (α_sn - α_free) * f_sn, α_sn))
        if α_blend < FT(0.4) || keep == one(FT)
            albedo_field[i, j, 1] = α_blend
        else
            T_C = T[i, j, 1] - FT(273.15)
            albedo_field[i, j, 1] = min(α_blend,
                               max(α_blend - FT(0.1) * (T_C + FT(10)) /
                                             FT(10) * α_blend,
                                   α_blend - FT(0.05)))
        end

        emissivity_field[i, j, 1] = max(keep * emissivity_snow,
                             min(ε_free + (emissivity_snow - ε_free) * f_sn,
                                 emissivity_snow))
    end
end

function update_diagnostics!(s::RucSurfaceProperties, state, fluxes, surface, parameters, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _update_surface_properties!,
            s.albedo, s.emissivity, s.roughness_length,
            state.snowfrac, state.keep_snow_albedo,
            state.T, state.newsn, state.snhei,
            s.vegfrac,
            s.albedo_vegetation, s.emissivity_vegetation, s.roughness_length_vegetation,
            s.albedo_snow, s.albedo_bare,
            s.emissivity_snow, s.emissivity_bare,
            s.roughness_length_snow, s.roughness_length_bare)
    return nothing
end

#####
##### Atmosphere-facing accessors
#####

albedo(s::RucSurfaceProperties, state, parameters)                    = s.albedo
emissivity(s::RucSurfaceProperties, state, parameters)                = s.emissivity
momentum_roughness_length(s::RucSurfaceProperties, state, parameters) = s.roughness_length
# RUC follows Garratt 1992: scalar roughness is `roughness_length / 10`.
# Keep that convention here so the atmosphere's MOST scalar-roughness
# consumer (`LandRoughnessLength(multiplier=0.1)`) remains numerically
# equivalent.
scalar_roughness_length(s::RucSurfaceProperties, state, parameters) = s.roughness_length

Base.summary(::RucSurfaceProperties{FT}) where FT = "RucSurfaceProperties{$FT}"
