#####
##### `TiledLandInterface` — a mosaic of a vegetated `CanopyGroundSurface` tile and a bare-soil
##### skin tile over one soil column; each tile runs the single-tile interface solve and
##### the fluxes are area-weighted, 𝒬 = f 𝒬ᵛᵉᵍ + (1 − f) 𝒬ᵇᵃʳᵉ.
#####

"""
    struct TiledLandInterface

A forested tile and a bare-soil tile over a shared soil column, each a full
[`atmosphere_land_interface`](@ref) with its own roughness, with the area-weighted fluxes
and temperatures the atmosphere and slab read. Build it with
[`TiledLandInterface(grid, atmosphere, land; …)`](@ref) and pass it as
`atmosphere_land_interface = …` to `AtmosphereLandModel` / `ComponentInterfaces`.

Fields:
- `vegetated`   : the forested tile, an `AtmosphereInterface` with a [`CanopyGroundSurface`](@ref).
- `bare`        : the bare-soil tile, an `AtmosphereInterface` with an [`EnergyBalanceTemperature`](@ref).
- `fraction`    : the forested fraction `f ∈ [0, 1]` — a `Number`, `Field`, or `FieldTimeSeries`.
- `fluxes`      : the blended [`AtmosphereSurfaceFluxes`](@ref).
- `temperature` : the blended [`CanopyGroundSurfaceDiagnostics`](@ref).
"""
struct TiledLandInterface{V, B, F, FL, T}
    vegetated   :: V
    bare        :: B
    fraction    :: F
    fluxes      :: FL
    temperature :: T
end

"""
$(TYPEDSIGNATURES)

Build the exchange interface for a land component configured with `TiledLandSurface`.
`vegetated_fluxes` and `bare_fluxes` specify the turbulent transport for each tile.
"""
function TiledLandInterface(grid, atmosphere, land;
                            vegetated_fluxes       = default_atmosphere_land_fluxes(land, eltype(grid)),
                            bare_fluxes            = default_atmosphere_land_fluxes(land, eltype(grid)),
                            velocity_difference    = RelativeVelocity())

    surface = default_al_temperature(land)
    vegetated, bare = surface.vegetated, surface.bare
    skin_conductance(bare.temperature) == skin_conductance(vegetated.temperature) ||
        throw(ArgumentError("the bare and vegetated tiles must share the skin conductance Λᵍ"))

    vegetated_interface = atmosphere_land_interface(grid, atmosphere, land;
                                                    fluxes              = vegetated_fluxes,
                                                    temperature         = vegetated.temperature,
                                                    specific_humidity   = vegetated.specific_humidity,
                                                    velocity_difference = velocity_difference)

    bare_interface = atmosphere_land_interface(grid, atmosphere, land;
                                               fluxes              = bare_fluxes,
                                               temperature         = bare.temperature,
                                               velocity_difference = velocity_difference,
                                               specific_humidity   = bare.specific_humidity)

    fluxes      = AtmosphereSurfaceFluxes(grid)
    temperature = build_interface_temperature(vegetated.temperature, grid)

    return TiledLandInterface(vegetated_interface, bare_interface, surface.fraction, fluxes, temperature)
end

Base.summary(::TiledLandInterface) = "TiledLandInterface"
Base.show(io::IO, ti::TiledLandInterface) =
    print(io, "TiledLandInterface(vegetated=", summary(ti.vegetated.properties.temperature_formulation),
              ", bare=", summary(ti.bare.properties.temperature_formulation), ")")

@inline computed_fluxes(ti::TiledLandInterface) = ti.fluxes

skin_conductance(ti::TiledLandInterface) = skin_conductance(ti.vegetated)

validate_zero_plane_displacement(ti::TiledLandInterface, zᵃᵗ) =
    foreach(tile -> validate_zero_plane_displacement(tile, zᵃᵗ), (ti.vegetated, ti.bare))

EarthSystemModels.surface_temperature(ti::TiledLandInterface) = interface_node_temperature(ti.temperature)
EarthSystemModels.surface_temperature(ti::TiledLandInterface, ocean_interface) =
    EarthSystemModels.surface_temperature(ti)

function compute_atmosphere_land_fluxes!(coupled_model, ti::TiledLandInterface, Δt)
    compute_atmosphere_land_fluxes!(coupled_model, ti.vegetated, Δt)
    compute_atmosphere_land_fluxes!(coupled_model, ti.bare, Δt)

    exchanger = coupled_model.interfaces.exchanger
    grid  = exchanger.grid
    arch  = architecture(grid)
    clock = coupled_model.clock

    fraction, fraction_time_interpolator = kernel_surface_field(ti.fraction, arch, clock.time)
    radiation_state = isnothing(exchanger.radiation) ? nothing : exchanger.radiation.state

    launch!(arch, grid, :xy, _blend_tiled_land_fluxes!,
            ti.fluxes, ti.temperature,
            ti.vegetated.fluxes, ti.vegetated.temperature,
            ti.bare.fluxes, ti.bare.temperature, skin_conductance(ti.bare), exchanger.land.state.T,
            fraction, fraction_time_interpolator, grid, Time(clock.time),
            kernel_radiation_properties(coupled_model.radiation, exchanger.radiation), radiation_state)

    return nothing
end

@kernel function _blend_tiled_land_fluxes!(blended_fluxes, blended_temperature,
                                           veg_fluxes, veg_temperature,
                                           bare_fluxes, bare_temperature, Λ, T,
                                           fraction, fraction_time_interpolator, grid, time,
                                           radiation_properties, radiation_state)
    i, j = @index(Global, NTuple)
    f = clamp(surface_field_value(fraction, i, j, fraction_time_interpolator), 0, 1)
    g = 1 - f
    bare_radiation = air_land_interface_radiation_state(radiation_properties, radiation_state,
                                                       i, j, size(grid, 3), grid, time)

    @inbounds begin
        blended_fluxes.latent_heat[i, j, 1]       = f * veg_fluxes.latent_heat[i, j, 1]       + g * bare_fluxes.latent_heat[i, j, 1]
        blended_fluxes.sensible_heat[i, j, 1]     = f * veg_fluxes.sensible_heat[i, j, 1]     + g * bare_fluxes.sensible_heat[i, j, 1]
        blended_fluxes.water_vapor[i, j, 1]       = f * veg_fluxes.water_vapor[i, j, 1]       + g * bare_fluxes.water_vapor[i, j, 1]
        blended_fluxes.x_momentum[i, j, 1]        = f * veg_fluxes.x_momentum[i, j, 1]        + g * bare_fluxes.x_momentum[i, j, 1]
        blended_fluxes.y_momentum[i, j, 1]        = f * veg_fluxes.y_momentum[i, j, 1]        + g * bare_fluxes.y_momentum[i, j, 1]
        blended_fluxes.friction_velocity[i, j, 1] = f * veg_fluxes.friction_velocity[i, j, 1] + g * bare_fluxes.friction_velocity[i, j, 1]
        blended_fluxes.temperature_scale[i, j, 1] = f * veg_fluxes.temperature_scale[i, j, 1] + g * bare_fluxes.temperature_scale[i, j, 1]
        blended_fluxes.water_vapor_scale[i, j, 1] = f * veg_fluxes.water_vapor_scale[i, j, 1] + g * bare_fluxes.water_vapor_scale[i, j, 1]

        # The bare tile is a single skin: its temperature is both its ground surface and
        # its skin, and its turbulent fluxes are all soil fluxes. The canopy water store is
        # kept per unit vegetated area, so its evaporation is not area-weighted.
        Tᵇ = bare_temperature[i, j, 1]
        blended_temperature.interface[i, j, 1]              = f * veg_temperature.interface[i, j, 1] + g * Tᵇ
        blended_temperature.canopy[i, j, 1]                 = veg_temperature.canopy[i, j, 1]
        blended_temperature.ground[i, j, 1]                 = f * veg_temperature.ground[i, j, 1] + g * Tᵇ
        blended_temperature.ground_heat_flux[i, j, 1]       = f * veg_temperature.ground_heat_flux[i, j, 1] + g * Λ * (Tᵇ - T[i, j, 1])
        blended_temperature.canopy_latent_heat[i, j, 1]     = f * veg_temperature.canopy_latent_heat[i, j, 1]
        blended_temperature.soil_latent_heat[i, j, 1]       = f * veg_temperature.soil_latent_heat[i, j, 1] + g * bare_fluxes.latent_heat[i, j, 1]
        blended_temperature.canopy_sensible_heat[i, j, 1]   = f * veg_temperature.canopy_sensible_heat[i, j, 1]
        blended_temperature.soil_sensible_heat[i, j, 1]     = f * veg_temperature.soil_sensible_heat[i, j, 1] + g * bare_fluxes.sensible_heat[i, j, 1]
        blended_temperature.canopy_evaporation[i, j, 1]     = veg_temperature.canopy_evaporation[i, j, 1]
        blended_temperature.canopy_wet_latent_heat[i, j, 1] = f * veg_temperature.canopy_wet_latent_heat[i, j, 1]
        blended_temperature.land_vapor_flux[i, j, 1]        = f * veg_temperature.land_vapor_flux[i, j, 1] + g * bare_fluxes.water_vapor[i, j, 1]
        blended_temperature.transpiration[i, j, 1]          = f * veg_temperature.transpiration[i, j, 1]

        # Skin (effective radiating) temperature of the mosaic: σ T⁴ = f · LWꜛ(vegetated) +
        # (1 − f) · LWꜛ(bare), where the bare tile emits ϵ σ Tᵇ⁴ and reflects (1 − ϵ) of the downwelling.
        σ, ϵ = bare_radiation.σ, bare_radiation.ϵ
        Tᵛ   = veg_temperature.effective[i, j, 1]
        LWꜛᵇ = ϵ * σ * Tᵇ^4 - (1 - ϵ) * bare_radiation.ℐꜜˡʷ
        T⁴   = f * Tᵛ^4 + g * ifelse(σ > 0, LWꜛᵇ / σ, Tᵇ^4)
        blended_temperature.effective[i, j, 1] = sqrt(sqrt(T⁴))

        # Both tiles see the same incident shortwave, so the mosaic albedo is their area weight.
        blended_temperature.effective_albedo[i, j, 1] =
            f * veg_temperature.effective_albedo[i, j, 1] + g * bare_radiation.α
    end
end

land_surface_fluxes(interface::TiledLandInterface, bulk_temperature) = land_surface_fluxes(interface.temperature)
radiative_surface(interface::TiledLandInterface, albedo, emissivity) = radiative_surface(interface.temperature, albedo, emissivity)
