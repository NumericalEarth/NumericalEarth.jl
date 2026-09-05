#####
##### `TiledLandInterface` — a mosaic of a vegetated `CanopyAirSpace` tile and a bare-soil
##### skin tile over one soil column; each tile runs the single-tile interface solve and
##### the fluxes are area-weighted, 𝒬 = f 𝒬ᵛᵉᵍ + (1 − f) 𝒬ᵇᵃʳᵉ.
#####

"""
    struct TiledLandInterface

A vegetated tile and a bare-soil tile over a shared soil column, each a full
[`atmosphere_land_interface`](@ref) with its own roughness, with the area-weighted fluxes
and temperatures the atmosphere and slab read. Build it with
[`TiledLandInterface(grid, atmosphere, land; …)`](@ref) and pass it as
`atmosphere_land_interface = …` to `AtmosphereLandModel` / `ComponentInterfaces`.

Fields:
- `vegetated`   : the vegetated tile, an `AtmosphereInterface` with a [`CanopyAirSpace`](@ref).
- `bare`        : the bare-soil tile, an `AtmosphereInterface` with an [`EnergyBalanceTemperature`](@ref).
- `fraction`    : the vegetated fraction `f ∈ [0, 1]` — a `Number`, `Field`, or `FieldTimeSeries`.
- `fluxes`      : the blended [`AtmosphereSurfaceFluxes`](@ref).
- `temperature` : the blended [`CanopyAirSpaceDiagnostics`](@ref).
"""
struct TiledLandInterface{V, B, F, FL, T}
    vegetated   :: V
    bare        :: B
    fraction    :: F
    fluxes      :: FL
    temperature :: T
end

"""
    TiledLandInterface(grid, atmosphere, land;
                       vegetated,
                       fraction,
                       bare                = EnergyBalanceTemperature(vegetated.soil_skin_flux),
                       bare_specific_humidity = vegetated.soil,
                       vegetated_fluxes    = default_atmosphere_land_fluxes(land, eltype(grid)),
                       bare_fluxes         = default_atmosphere_land_fluxes(land, eltype(grid)),
                       velocity_difference = RelativeVelocity())

Build a two-tile (vegetated + bare) land interface. `vegetated` is a [`CanopyAirSpace`](@ref);
the bare tile is a soil skin with the same skin→bulk conduction and soil vapor branch by
default. `fraction` is the vegetated fraction (a `Number`, `Field`, or `FieldTimeSeries`).
Pass `vegetated_fluxes` / `bare_fluxes` to give the tiles a roughness contrast.
"""
function TiledLandInterface(grid, atmosphere, land;
                            vegetated,
                            fraction,
                            bare                   = EnergyBalanceTemperature(vegetated.soil_skin_flux),
                            bare_specific_humidity = vegetated.soil,
                            vegetated_fluxes       = default_atmosphere_land_fluxes(land, eltype(grid)),
                            bare_fluxes            = default_atmosphere_land_fluxes(land, eltype(grid)),
                            velocity_difference    = RelativeVelocity())

    skin_conductance(bare) == skin_conductance(vegetated) ||
        throw(ArgumentError("the bare and vegetated tiles must share the skin conductance Λᵍ"))

    vegetated_interface = atmosphere_land_interface(grid, atmosphere, land;
                                                    fluxes              = vegetated_fluxes,
                                                    temperature         = vegetated,
                                                    velocity_difference = velocity_difference)

    bare_interface = atmosphere_land_interface(grid, atmosphere, land;
                                               fluxes              = bare_fluxes,
                                               temperature         = bare,
                                               velocity_difference = velocity_difference,
                                               specific_humidity   = bare_specific_humidity)

    fluxes      = AtmosphereSurfaceFluxes(grid)
    temperature = build_interface_temperature(vegetated, grid)

    return TiledLandInterface(vegetated_interface, bare_interface, fraction, fluxes, temperature)
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

interface_prognostic_state(ti::TiledLandInterface) =
    (; vegetated = interface_prognostic_state(ti.vegetated),
       bare      = interface_prognostic_state(ti.bare))

function restore_interface_state!(ti::TiledLandInterface, state)
    restore_interface_state!(ti.vegetated, state.vegetated)
    restore_interface_state!(ti.bare, state.bare)
    return nothing
end

function compute_atmosphere_land_fluxes!(coupled_model, ti::TiledLandInterface, Δt)
    compute_atmosphere_land_fluxes!(coupled_model, ti.vegetated, Δt)
    compute_atmosphere_land_fluxes!(coupled_model, ti.bare, Δt)

    exchanger = coupled_model.interfaces.exchanger
    grid  = exchanger.grid
    arch  = architecture(grid)
    clock = coupled_model.clock

    fraction, fraction_time_interpolator = kernel_surface_field(ti.fraction, arch, clock.time)

    launch!(arch, grid, :xy, _blend_tiled_land_fluxes!,
            ti.fluxes, ti.temperature,
            ti.vegetated.fluxes, ti.vegetated.temperature,
            ti.bare.fluxes, ti.bare.temperature, skin_conductance(ti.bare), exchanger.land.state.T,
            fraction, fraction_time_interpolator)

    return nothing
end

@kernel function _blend_tiled_land_fluxes!(blended_fluxes, blended_temperature,
                                           veg_fluxes, veg_temperature,
                                           bare_fluxes, bare_temperature, Λ, T,
                                           fraction, fraction_time_interpolator)
    i, j = @index(Global, NTuple)
    f = clamp(surface_field_value(fraction, i, j, fraction_time_interpolator), 0, 1)
    g = 1 - f

    @inbounds begin
        blended_fluxes.latent_heat[i, j, 1]       = f * veg_fluxes.latent_heat[i, j, 1]       + g * bare_fluxes.latent_heat[i, j, 1]
        blended_fluxes.sensible_heat[i, j, 1]     = f * veg_fluxes.sensible_heat[i, j, 1]     + g * bare_fluxes.sensible_heat[i, j, 1]
        blended_fluxes.water_vapor[i, j, 1]       = f * veg_fluxes.water_vapor[i, j, 1]       + g * bare_fluxes.water_vapor[i, j, 1]
        blended_fluxes.x_momentum[i, j, 1]        = f * veg_fluxes.x_momentum[i, j, 1]        + g * bare_fluxes.x_momentum[i, j, 1]
        blended_fluxes.y_momentum[i, j, 1]        = f * veg_fluxes.y_momentum[i, j, 1]        + g * bare_fluxes.y_momentum[i, j, 1]
        blended_fluxes.friction_velocity[i, j, 1] = f * veg_fluxes.friction_velocity[i, j, 1] + g * bare_fluxes.friction_velocity[i, j, 1]
        blended_fluxes.temperature_scale[i, j, 1] = f * veg_fluxes.temperature_scale[i, j, 1] + g * bare_fluxes.temperature_scale[i, j, 1]
        blended_fluxes.water_vapor_scale[i, j, 1] = f * veg_fluxes.water_vapor_scale[i, j, 1] + g * bare_fluxes.water_vapor_scale[i, j, 1]

        # The bare tile is a single skin: its own temperature is its soil skin and its
        # turbulent fluxes are all soil fluxes. The canopy water store is kept per unit
        # vegetated area, so its evaporation is not area-weighted.
        Tᵇ = bare_temperature[i, j, 1]
        blended_temperature.interface[i, j, 1]              = f * veg_temperature.interface[i, j, 1] + g * Tᵇ
        blended_temperature.canopy[i, j, 1]                 = veg_temperature.canopy[i, j, 1]
        blended_temperature.soil_skin[i, j, 1]              = f * veg_temperature.soil_skin[i, j, 1] + g * Tᵇ
        blended_temperature.ground_heat_flux[i, j, 1]       = f * veg_temperature.ground_heat_flux[i, j, 1] + g * Λ * (Tᵇ - T[i, j, 1])
        blended_temperature.canopy_latent_heat[i, j, 1]     = f * veg_temperature.canopy_latent_heat[i, j, 1]
        blended_temperature.soil_latent_heat[i, j, 1]       = f * veg_temperature.soil_latent_heat[i, j, 1] + g * bare_fluxes.latent_heat[i, j, 1]
        blended_temperature.canopy_sensible_heat[i, j, 1]   = f * veg_temperature.canopy_sensible_heat[i, j, 1]
        blended_temperature.soil_sensible_heat[i, j, 1]     = f * veg_temperature.soil_sensible_heat[i, j, 1] + g * bare_fluxes.sensible_heat[i, j, 1]
        blended_temperature.canopy_evaporation[i, j, 1]     = veg_temperature.canopy_evaporation[i, j, 1]
        blended_temperature.canopy_wet_latent_heat[i, j, 1] = f * veg_temperature.canopy_wet_latent_heat[i, j, 1]
        blended_temperature.land_vapor_flux[i, j, 1]        = f * veg_temperature.land_vapor_flux[i, j, 1] + g * bare_fluxes.water_vapor[i, j, 1]

        # Radiating temperature: area-weighted in σT⁴.
        blended_temperature.effective[i, j, 1] = sqrt(sqrt(f * veg_temperature.effective[i, j, 1]^4 + g * Tᵇ^4))
    end
end
