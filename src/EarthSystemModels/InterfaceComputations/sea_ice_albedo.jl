"""
    SeaIceAlbedo{FT, HI, HS, TS}

Sea ice albedo parameterization following the CCSM3 scheme (Briegleb et al. 2004).

Computes broadband albedo as a function of ice thickness, snow depth, and surface
temperature. The scheme blends between bare ice and snow-covered albedos, with
a temperature-dependent reduction near the melting point to implicitly represent
melt pond formation.

Algorithm:
1. Base cold albedos: bare ice (0.53) and snow-covered (0.82)
2. Temperature reduction within 1C of melting: Δα_ice = 0.075, Δα_snow = 0.10
3. Thin-ice transition to ocean albedo below h_amin = 0.5 m
4. Snow cover interpolation: full snow albedo at h_snow > h_smin = 0.02 m

References:
- Briegleb, B.P., C.M. Bitz, E.C. Hunke, W.H. Lipscomb, and M.M. Schramm (2004):
  Scientific description of the sea ice component in CCSM3. NCAR Tech Note.
- Briegleb, B.P. and B. Light (2007): NCAR/TN-472+STR.
"""
struct SeaIceAlbedo{FT, HI, HS, TS}
    # Cold base albedos (broadband, approx 0.52 * vis + 0.48 * nir)
    ice_albedo :: FT    # 0.52*0.73 + 0.48*0.33 = 0.538 ≈ 0.54
    snow_albedo :: FT   # 0.52*0.96 + 0.48*0.68 = 0.825 ≈ 0.83
    # Melt reduction
    ice_melt_reduction :: FT    # 0.075
    snow_melt_reduction :: FT   # 0.10
    melting_temperature :: FT   # 0 C
    temperature_range :: FT     # 1 C
    # Thickness scales
    ocean_albedo :: FT          # 0.06
    minimum_ice_thickness :: FT # 0.5 m
    minimum_snow_depth :: FT    # 0.02 m
    # References to model fields
    ice_thickness :: HI
    snow_thickness :: HS
    surface_temperature :: TS
end

Adapt.adapt_structure(to, α::SeaIceAlbedo) =
    SeaIceAlbedo(α.ice_albedo,
                      α.snow_albedo,
                      α.ice_melt_reduction,
                      α.snow_melt_reduction,
                      α.melting_temperature,
                      α.temperature_range,
                      α.ocean_albedo,
                      α.minimum_ice_thickness,
                      α.minimum_snow_depth,
                      Adapt.adapt(to, α.ice_thickness),
                      Adapt.adapt(to, α.snow_thickness),
                      Adapt.adapt(to, α.surface_temperature))

"""
    SeaIceAlbedo(ice_thickness, snow_thickness, surface_temperature;
                      ice_albedo = 0.54,
                      snow_albedo = 0.83,
                      ice_melt_reduction = 0.075,
                      snow_melt_reduction = 0.10,
                      melting_temperature = 0.0,
                      temperature_range = 1.0,
                      ocean_albedo = 0.06,
                      minimum_ice_thickness = 0.5,
                      minimum_snow_depth = 0.02)

Construct a CCSM3 sea ice albedo parameterization. Requires references to the sea ice
model's ice thickness, snow thickness, and surface temperature fields.

Broadband albedos are approximate averages of the visible and near-IR bands
weighted by solar spectrum (52% visible, 48% near-IR):
- ice_albedo ≈ 0.52 x 0.73 + 0.48 x 0.33 ≈ 0.54
- snow_albedo ≈ 0.52 x 0.96 + 0.48 x 0.68 ≈ 0.83
"""
function SeaIceAlbedo(ice_thickness, snow_thickness, surface_temperature;
                      FT = Float64,
                      ice_albedo = 0.54,
                      snow_albedo = 0.83,
                      ice_melt_reduction = 0.075,
                      snow_melt_reduction = 0.10,
                      melting_temperature = 0.0,
                      temperature_range = 1.0,
                      ocean_albedo = 0.06,
                      minimum_ice_thickness = 0.5,
                      minimum_snow_depth = 0.02)

    return SeaIceAlbedo(convert(FT, ice_albedo),
                        convert(FT, snow_albedo),
                        convert(FT, ice_melt_reduction),
                        convert(FT, snow_melt_reduction),
                        convert(FT, melting_temperature),
                        convert(FT, temperature_range),
                        convert(FT, ocean_albedo),
                        convert(FT, minimum_ice_thickness),
                        convert(FT, minimum_snow_depth),
                        ice_thickness,
                        snow_thickness,
                        surface_temperature)
end

Base.summary(::SeaIceAlbedo{FT}) where FT = "SeaIceAlbedo{$FT}"
Base.show(io::IO, α::SeaIceAlbedo{FT}) where FT =
    print(io, "SeaIceAlbedo{$FT}(ice=", α.ice_albedo,
              ", snow=", α.snow_albedo, ")")

@inline function stateindex(α::SeaIceAlbedo, i, j, k, grid, time, loc, args...)
    @inbounds hi = α.ice_thickness[i, j, 1]
    @inbounds Ts = α.surface_temperature[i, j, 1]

    # Snow thickness: may be nothing (no snow model)
    hs = get_snow_thickness(α.snow_thickness, i, j)

    # Temperature-dependent reduction (implicit melt ponds)
    Tm = α.melting_temperature
    ΔT = α.temperature_range
    fT = clamp((Ts - Tm + ΔT) / ΔT, zero(Ts), one(Ts))

    αi = α.ice_albedo  - α.ice_melt_reduction  * fT
    αs = α.snow_albedo - α.snow_melt_reduction * fT

    # Thin ice → transition to ocean albedo
    αo = α.ocean_albedo
    fh = clamp(hi / α.minimum_ice_thickness, zero(hi), one(hi))
    αi = αo + (αi - αo) * fh

    # Snow cover blending
    fs = clamp(hs / α.minimum_snow_depth, zero(hs), one(hs))
    return fs * αs + (1 - fs) * αi
end

# Helper to handle nothing snow thickness (no snow model)
@inline get_snow_thickness(hs::Nothing, i, j, grid) = zero(grid)
@inline get_snow_thickness(hs, i, j, grid) = @inbounds hs[i, j, 1]
