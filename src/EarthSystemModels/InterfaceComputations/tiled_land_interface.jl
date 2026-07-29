#####
##### `TiledLandInterface` — subgrid vegetation/bare-soil tiling (parallel fluxes).
#####
##### At 100 m–1 km a cell is rarely pure canopy or pure bare soil. A `TiledLandInterface`
##### makes it a **mosaic** of a vegetated fraction `f = f_veg` and a bare-soil fraction
##### `1 − f`: each tile runs the *same* single-tile interface solve independently against
##### the *same* atmosphere (different roughness / stability), and the fluxes are
##### area-weighted into the boundary condition the atmosphere and slab read,
#####
#####     𝒬 = f · 𝒬_veg + (1 − f) · 𝒬_bare.
#####
##### This is the SOTA parallel/tiled-flux scheme (Avissar & Pielke 1989; Koster–Suarez
##### 1992; Noah-MP `F_veg·veg + (1−F_veg)·bare`; ClimaLand v1 `(1−σ)E_soil + σ·canopy`).
##### It is *complementary* to the `CanopyAirSpace` composite: the vegetated tile is a full
##### CAS footprint (canopy over its own shaded soil); the bare tile is open soil talking
##### straight to the atmosphere. At `f = 1` the blend reduces to pure CAS; at `f = 0` to
##### pure bare soil.
#####
##### Both tiles are `CanopyAirSpace` objects so they emit the *same currency* — an
##### internalized-radiation, conduction-driven slab energy input (`Gcond`) and an
##### upwelling-longwave (LST) — and the blend is a clean area-weight (no radiation
##### double-count; `apply_air_land_radiative_fluxes!` stays a no-op). The bare tile is a
##### canopy-free CAS (LAI = 0), derived from the vegetated tile by default. The two tiles
##### share one soil column (one `Tˡᵃ`, `Mˡᵃ`, `𝒮`), matching Noah-MP / JULES; both deplete
##### the same water store through the area-weighted vapor flux.
#####

"""
    struct TiledLandInterface

A subgrid mosaic of a vegetated tile and a bare-soil tile over a shared soil column.
Holds two `CanopyAirSpace` sub-interfaces (each a full [`atmosphere_land_interface`](@ref)
with its own roughness), the vegetation fraction `f_veg`, and the area-weighted buffers
the atmosphere and slab read. Build it with [`TiledLandInterface(grid, atmosphere, land; …)`](@ref)
and pass it as `atmosphere_land_interface = …` to `AtmosphereLandModel` / `ComponentInterfaces`.

Fields:
- `vegetated`   : the vegetated tile, an `AtmosphereInterface` with a [`CanopyAirSpace`](@ref).
- `bare`        : the bare-soil tile, an `AtmosphereInterface` with a canopy-free `CanopyAirSpace`.
- `fraction`    : `f_veg ∈ [0, 1]` — a `Number`, `Field`, or `FieldTimeSeries`.
- `fluxes`      : the blended [`AtmosphereSurfaceFluxes`](@ref) the atmosphere/slab read.
- `temperature` : the blended diagnostic temperatures/fluxes (a `CanopyAirSpace`-shaped NamedTuple).
"""
struct TiledLandInterface{V, B, F, FL, T}
    vegetated   :: V
    bare        :: B
    fraction    :: F
    fluxes      :: FL
    temperature :: T
end

# A canopy-free canopy: LAI = 0 collapses the leaf branch (no transpiration, no canopy
# shortwave/longwave, `gˡʰ = 0`), leaving the soil-skin balance the bare tile needs.
zero_leaf_area_index_canopy(q::CanopyConductanceHumidity) =
    CanopyConductanceHumidity(zero(q.atmospheric_co2), q.photosynthesis, q.conductance,
                              q.moisture_stress, q.absorbed_par, q.atmospheric_co2, q.phase)

"""
    bare_canopy_air_space(vegetated::CanopyAirSpace; undercanopy_conductance)

Derive the bare-soil tile from the vegetated `CanopyAirSpace`: the same soil vapor branch,
skin conduction, albedos, emissivities, and optics, but with a canopy-free (LAI = 0) leaf
branch and no interception. The soil then talks straight to the atmosphere (all shortwave
reaches the ground, the ground sees the sky, no transpiration). `undercanopy_conductance`
sets the soil↔canopy-air coupling for the bare tile (defaults to the vegetated value; a
larger value pushes the soil resistance toward the pure-aerodynamic limit).
"""
function bare_canopy_air_space(c::CanopyAirSpace; undercanopy_conductance = c.undercanopy_conductance)
    FT = typeof(c.undercanopy_conductance)
    return CanopyAirSpace(c.soil, zero_leaf_area_index_canopy(c.canopy), c.soil_skin_flux,
                          c.leaf_albedo, c.ground_albedo, c.canopy_emissivity_max, c.ground_emissivity,
                          c.extinction, c.clumping, c.leaf_boundary_conductance,
                          convert(FT, undercanopy_conductance),
                          c.inner_iterations, c.relaxation, nothing, c.phase)
end

"""
    TiledLandInterface(grid, atmosphere, land;
                       vegetated,
                       fraction,
                       bare              = bare_canopy_air_space(vegetated),
                       vegetated_fluxes  = default_atmosphere_land_fluxes(land, eltype(grid)),
                       bare_fluxes       = default_atmosphere_land_fluxes(land, eltype(grid)),
                       velocity_difference = RelativeVelocity())

Build a two-tile (vegetated + bare) land interface. `vegetated` is a [`CanopyAirSpace`](@ref);
`bare` defaults to its canopy-free counterpart. `fraction` is `f_veg` (a `Number`, `Field`, or
`FieldTimeSeries`). Pass `vegetated_fluxes` / `bare_fluxes` to give the tiles a roughness
contrast (forest z₀ ≫ bare z₀) — a first-order control on inland wind decay.

```julia
model = AtmosphereLandModel(atmosphere, land; radiation,
    atmosphere_land_interface = TiledLandInterface(grid, atmosphere, land;
                                                   vegetated = canopy_air_space,
                                                   fraction  = 0.6))
```
"""
function TiledLandInterface(grid, atmosphere, land;
                            vegetated,
                            fraction,
                            bare                = bare_canopy_air_space(vegetated),
                            vegetated_fluxes    = default_atmosphere_land_fluxes(land, eltype(grid)),
                            bare_fluxes         = default_atmosphere_land_fluxes(land, eltype(grid)),
                            velocity_difference = RelativeVelocity())

    vegetated_interface = atmosphere_land_interface(grid, atmosphere, land;
                                                    fluxes              = vegetated_fluxes,
                                                    temperature         = vegetated,
                                                    velocity_difference = velocity_difference,
                                                    specific_humidity   = vegetated)

    bare_interface = atmosphere_land_interface(grid, atmosphere, land;
                                               fluxes              = bare_fluxes,
                                               temperature         = bare,
                                               velocity_difference = velocity_difference,
                                               specific_humidity   = bare)

    fluxes      = AtmosphereSurfaceFluxes(grid)
    temperature = build_interface_temperature(vegetated, grid)

    return TiledLandInterface(vegetated_interface, bare_interface, fraction, fluxes, temperature)
end

Base.summary(::TiledLandInterface) = "TiledLandInterface"
Base.show(io::IO, ti::TiledLandInterface) =
    print(io, "TiledLandInterface(vegetated=", summary(ti.vegetated.properties.temperature_formulation),
              ", bare=", summary(ti.bare.properties.temperature_formulation), ")")

@inline computed_fluxes(ti::TiledLandInterface) = ti.fluxes

# The atmosphere-facing surface temperature is the blended canopy-air node (the same
# NamedTuple signal a single CanopyAirSpace uses).
EarthSystemModels.surface_temperature(ti::TiledLandInterface) = interface_node_temperature(ti.temperature)

"""
    leaf_area_index_cover_fraction(leaf_area_index; extinction=0.5, clumping=1)

Beer–Lambert vegetation cover fraction `f_veg = 1 − exp(−K·Ω·LAI)` — the fraction of the
cell shaded by foliage, and Noah-MP's `1 − gap`. A data-free default for the tiling
`fraction` (the same relation the `CanopyAirSpace` uses for its canopy shortwave split).
"""
@inline leaf_area_index_cover_fraction(leaf_area_index; extinction = 0.5, clumping = 1) =
    1 - exp(-extinction * clumping * leaf_area_index)

#####
##### Two-pass parallel fluxes: run each tile's existing single-tile solve, then blend.
#####

function compute_atmosphere_land_fluxes!(coupled_model, ti::TiledLandInterface)
    # Pass 1 & 2: each tile writes its own turbulent fluxes and diagnostic temperatures,
    # both reading the shared land exchanger state (𝒮, Tˡᵃ, Wᶜ).
    compute_atmosphere_land_fluxes!(coupled_model, ti.vegetated)
    compute_atmosphere_land_fluxes!(coupled_model, ti.bare)

    grid  = coupled_model.interfaces.exchanger.grid
    arch  = architecture(grid)
    clock = coupled_model.clock

    fraction, fraction_time_interpolator = kernel_surface_field(ti.fraction, arch, clock.time)

    launch!(arch, grid, :xy, _blend_tiled_land_fluxes!,
            ti.fluxes, ti.temperature,
            ti.vegetated.fluxes, ti.vegetated.temperature,
            ti.bare.fluxes, ti.bare.temperature,
            fraction, fraction_time_interpolator, grid)

    return nothing
end

@kernel function _blend_tiled_land_fluxes!(blended_fluxes, blended_temperature,
                                           veg_fluxes, veg_temperature,
                                           bare_fluxes, bare_temperature,
                                           fraction, fraction_time_interpolator, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    f = clamp(convert(FT, surface_field_value(fraction, i, j, fraction_time_interpolator)), zero(FT), one(FT))
    g = one(FT) - f

    @inbounds begin
        blended_fluxes.latent_heat[i, j, 1]       = f * veg_fluxes.latent_heat[i, j, 1]       + g * bare_fluxes.latent_heat[i, j, 1]
        blended_fluxes.sensible_heat[i, j, 1]     = f * veg_fluxes.sensible_heat[i, j, 1]     + g * bare_fluxes.sensible_heat[i, j, 1]
        blended_fluxes.water_vapor[i, j, 1]       = f * veg_fluxes.water_vapor[i, j, 1]       + g * bare_fluxes.water_vapor[i, j, 1]
        blended_fluxes.x_momentum[i, j, 1]        = f * veg_fluxes.x_momentum[i, j, 1]        + g * bare_fluxes.x_momentum[i, j, 1]
        blended_fluxes.y_momentum[i, j, 1]        = f * veg_fluxes.y_momentum[i, j, 1]        + g * bare_fluxes.y_momentum[i, j, 1]
        blended_fluxes.friction_velocity[i, j, 1] = f * veg_fluxes.friction_velocity[i, j, 1] + g * bare_fluxes.friction_velocity[i, j, 1]
        blended_fluxes.temperature_scale[i, j, 1] = f * veg_fluxes.temperature_scale[i, j, 1] + g * bare_fluxes.temperature_scale[i, j, 1]
        blended_fluxes.water_vapor_scale[i, j, 1] = f * veg_fluxes.water_vapor_scale[i, j, 1] + g * bare_fluxes.water_vapor_scale[i, j, 1]

        blended_temperature.interface[i, j, 1]              = f * veg_temperature.interface[i, j, 1]              + g * bare_temperature.interface[i, j, 1]
        blended_temperature.canopy[i, j, 1]                 = f * veg_temperature.canopy[i, j, 1]                 + g * bare_temperature.canopy[i, j, 1]
        blended_temperature.soil_skin[i, j, 1]              = f * veg_temperature.soil_skin[i, j, 1]              + g * bare_temperature.soil_skin[i, j, 1]
        blended_temperature.ground_heat_flux[i, j, 1]        = f * veg_temperature.ground_heat_flux[i, j, 1]        + g * bare_temperature.ground_heat_flux[i, j, 1]
        blended_temperature.canopy_latent_heat[i, j, 1]     = f * veg_temperature.canopy_latent_heat[i, j, 1]     + g * bare_temperature.canopy_latent_heat[i, j, 1]
        blended_temperature.soil_latent_heat[i, j, 1]       = f * veg_temperature.soil_latent_heat[i, j, 1]       + g * bare_temperature.soil_latent_heat[i, j, 1]
        blended_temperature.canopy_sensible_heat[i, j, 1]   = f * veg_temperature.canopy_sensible_heat[i, j, 1]   + g * bare_temperature.canopy_sensible_heat[i, j, 1]
        blended_temperature.soil_sensible_heat[i, j, 1]     = f * veg_temperature.soil_sensible_heat[i, j, 1]     + g * bare_temperature.soil_sensible_heat[i, j, 1]
        blended_temperature.canopy_evaporation[i, j, 1]     = f * veg_temperature.canopy_evaporation[i, j, 1]     + g * bare_temperature.canopy_evaporation[i, j, 1]
        blended_temperature.canopy_wet_latent_heat[i, j, 1] = f * veg_temperature.canopy_wet_latent_heat[i, j, 1] + g * bare_temperature.canopy_wet_latent_heat[i, j, 1]

        # Effective (LST) temperature: area-weight in radiance (T⁴) space (σ cancels),
        # σ Teff⁴ = f · LWu_veg + (1−f) · LWu_bare.
        Teffᵛ = veg_temperature.effective[i, j, 1]
        Teffᵇ = bare_temperature.effective[i, j, 1]
        blended_temperature.effective[i, j, 1] = (f * Teffᵛ^4 + g * Teffᵇ^4)^convert(FT, 1//4)
    end
end
