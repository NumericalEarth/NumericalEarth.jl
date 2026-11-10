#####
##### `TiledLandInterface` — a mosaic of a vegetated `CanopyInterface` tile and a bare-soil
##### skin tile over one soil column; each tile runs the single-tile interface solve and
##### the vegetated and bare fluxes are area-weighted by the vegetated cover fraction f.
#####

"""
$(TYPEDSIGNATURES)

Specific humidity of a bare-soil skin that evaporates through the ground vapor path of a
[`CanopyInterface`](@ref), with the aerodynamic conductance `gᵃᵛ` in place of the undercanopy
path: the `ground_surface_resistance` `rᵍʳⁿᵈ` lies in series on both branches of `soil_humidity`,
and the `wet_soil_resistance` `rˢᵒⁱ` on its moist-soil branch. The skin humidity carries the flux
of the whole path through `gᵃᵛ` alone.
"""
struct BareSoilHumidity{S, W, L}
    soil_humidity             :: S
    wet_soil_resistance       :: W
    ground_surface_resistance :: L
end

Adapt.@adapt_structure BareSoilHumidity

land_state_accessors(q::BareSoilHumidity) = land_state_accessors(q.soil_humidity)
@inline interface_phase(q::BareSoilHumidity) = interface_phase(q.soil_humidity)

@inline local_interface_formulation(q::BareSoilHumidity, i, j) =
    BareSoilHumidity(q.soil_humidity, q.wet_soil_resistance, local_interface_formulation(q.ground_surface_resistance, i, j))

@inline function compute_interface_humidity(q::BareSoilHumidity, Tⁱⁿ, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    ℂᵃᵗ   = ℙₐ.thermodynamics_parameters
    ρᵃᵗ   = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    gᵃᵛ   = aerodynamic_vapor_conductance(Ψⁱⁿ, Ψₐ, ℂᵃᵗ)
    rᵍʳⁿᵈ = ground_surface_resistance(q.ground_surface_resistance, Ψⁱⁿ.fluxes.u★)
    rˢᵒⁱ  = soil_surface_resistance(q.wet_soil_resistance, Ψˡᵃ.𝒮)

    gᵈᵇʳ    = gᵃᵛ / (1 + gᵃᵛ * rᵍʳⁿᵈ / ρᵃᵗ)
    gᵍʳⁿᵈᵛ⁺ = gᵃᵛ / (1 + gᵃᵛ * (rᵍʳⁿᵈ + rˢᵒⁱ) / ρᵃᵗ)
    gᵍʳⁿᵈᵛ, qᵍʳⁿᵈ, _ = soil_vapor_terms(q.soil_humidity, Tⁱⁿ, gᵍʳⁿᵈᵛ⁺, gᵈᵇʳ, Ψⁱⁿ, Ψₐ, Ψˡᵃ, ℙₐ)

    # gᵍʳⁿᵈᵛ / gᵃᵛ → 1 as gᵃᵛ → 0
    return Ψₐ.q + ifelse(gᵃᵛ > 0, gᵍʳⁿᵈᵛ / gᵃᵛ, one(gᵃᵛ)) * (qᵍʳⁿᵈ - Ψₐ.q)
end

"""
    struct TiledLandInterface

A forested tile and a bare-soil tile over a shared soil column, each a full
[`atmosphere_land_interface`](@ref) with its own roughness, with the area-weighted fluxes
and temperatures the atmosphere and slab read. Build it with
[`TiledLandInterface(grid, atmosphere, land; …)`](@ref) and pass it as
`atmosphere_land_interface = …` to `AtmosphereLandModel` / `ComponentInterfaces`.

Fields:
- `vegetated`      : the forested tile, an `AtmosphereInterface` with a [`CanopyInterface`](@ref).
- `bare`           : the bare-soil tile, an `AtmosphereInterface` with an [`EnergyBalanceTemperature`](@ref).
- `cover_fraction` : the vegetated cover fraction `f ∈ [0, 1]` — a `Number`, `Field`, or `FieldTimeSeries`.
- `fluxes`         : the blended [`AtmosphereSurfaceFluxes`](@ref).
- `temperature`    : the blended [`CanopyInterfaceDiagnostics`](@ref). Canopy quantities (`canopy`,
                     `canopy_latent_heat`, `canopy_sensible_heat`, `canopy_evaporation`,
                     `wet_canopy_latent_heat`, `transpiration`) are those of the vegetated tile, per unit
                     vegetated area; the others are per unit cell area.
"""
struct TiledLandInterface{V, B, F, FL, T}
    vegetated      :: V
    bare           :: B
    cover_fraction :: F
    fluxes         :: FL
    temperature    :: T
end

"""
$(TYPEDSIGNATURES)

Build a tiled atmosphere–land interface from the `vegetated` canopy formulation and the
vegetated `cover_fraction`. `bare` defaults to a soil skin that shares the canopy's ground heat
flux and its ground vapor path, a [`BareSoilHumidity`](@ref). The land's surface storage belongs
to the vegetated tile. `vegetated_fluxes` and `bare_fluxes` specify the turbulent transport for
each tile.
"""
function TiledLandInterface(grid, atmosphere, land;
                            vegetated,
                            cover_fraction,
                            bare                = (temperature       = EnergyBalanceTemperature(vegetated.ground_heat_flux),
                                                   specific_humidity = BareSoilHumidity(vegetated.soil_humidity,
                                                                                        vegetated.wet_soil_resistance,
                                                                                        vegetated.ground_surface_resistance)),
                            vegetated_fluxes    = default_atmosphere_land_fluxes(land, eltype(grid)),
                            bare_fluxes         = default_atmosphere_land_fluxes(land, eltype(grid)),
                            velocity_difference = RelativeVelocity())

    ground_conductance(temperature_formulation(bare)) == ground_conductance(vegetated) ||
        throw(ArgumentError("the bare and vegetated tiles must share the ground conductance Λᵍʳⁿᵈ"))
    validate_surface_storage(land, vegetated)

    vegetated_interface = build_atmosphere_land_interface(grid, vegetated_fluxes, vegetated, velocity_difference)
    bare_interface      = build_atmosphere_land_interface(grid, bare_fluxes, bare, velocity_difference)

    return TiledLandInterface(vegetated_interface, bare_interface, cover_fraction,
                              AtmosphereSurfaceFluxes(grid), CanopyInterfaceDiagnostics(grid))
end

Base.summary(::TiledLandInterface) = "TiledLandInterface"
Base.show(io::IO, interface::TiledLandInterface) =
    print(io, "TiledLandInterface(vegetated=", summary(temperature_formulation(interface.vegetated.formulation.state)),
              ", bare=", summary(temperature_formulation(interface.bare.formulation.state)), ")")

@inline computed_fluxes(interface::TiledLandInterface) = interface.fluxes

ground_conductance(interface::TiledLandInterface) = ground_conductance(interface.vegetated)

# The land publishes the state both tiles read.
land_state_accessors(interface::TiledLandInterface) =
    merge(land_state_accessors(interface.vegetated), land_state_accessors(interface.bare))

validate_zero_plane_displacement(interface::TiledLandInterface, zᵃᵗ) =
    foreach(tile -> validate_zero_plane_displacement(tile, zᵃᵗ), (interface.vegetated, interface.bare))

EarthSystemModels.surface_temperature(interface::TiledLandInterface) = interface_node_temperature(interface.temperature)
EarthSystemModels.surface_temperature(interface::TiledLandInterface, ocean_interface) =
    EarthSystemModels.surface_temperature(interface)

# The land's surface storage belongs to the vegetated tile; the bare tile is a diagnostic skin.
function compute_atmosphere_land_fluxes!(coupled_model, interface::TiledLandInterface, Δt)
    compute_atmosphere_land_fluxes!(coupled_model, interface.vegetated, Δt)
    compute_atmosphere_land_fluxes!(coupled_model, interface.bare, Δt, land_surface_state(nothing))
    blend_tiles!(coupled_model, interface)
    return nothing
end

function initialize_land_interface!(coupled_model, interface::TiledLandInterface)
    initialize_land_interface!(coupled_model, interface.vegetated)
    initialize_land_interface!(coupled_model, interface.bare, land_surface_state(nothing))
    blend_tiles!(coupled_model, interface)
    return nothing
end

function blend_tiles!(coupled_model, interface)
    exchanger = coupled_model.interfaces.exchanger
    grid  = exchanger.grid
    arch  = architecture(grid)
    clock = coupled_model.clock

    cover_fraction, cover_fraction_time_interpolator = kernel_surface_field(interface.cover_fraction, arch, clock.time)
    radiation_state = isnothing(exchanger.radiation) ? nothing : exchanger.radiation.state

    launch!(arch, grid, :xy, _blend_tiled_land_fluxes!,
            interface.fluxes, interface.temperature,
            interface.vegetated.fluxes, interface.vegetated.temperature,
            interface.bare.fluxes, interface.bare.temperature, ground_conductance(interface.bare), exchanger.land.state.T,
            cover_fraction, cover_fraction_time_interpolator, grid, Time(clock.time),
            kernel_radiation_properties(coupled_model.radiation, exchanger.radiation), radiation_state)

    return nothing
end

@kernel function _blend_tiled_land_fluxes!(blended_fluxes, blended_temperature,
                                           vegetated_fluxes, vegetated_temperature,
                                           bare_fluxes, bare_temperature, Λ, T,
                                           cover_fraction, cover_fraction_time_interpolator, grid, time,
                                           radiation_properties, radiation_state)
    i, j = @index(Global, NTuple)
    f = clamp(stateindex(cover_fraction, i, j, 1, cover_fraction_time_interpolator), 0, 1)
    bare_radiation = air_land_interface_radiation_state(radiation_properties, radiation_state,
                                                       i, j, size(grid, 3), grid, time)

    @inbounds begin
        blended_fluxes.latent_heat[i, j, 1]       = f * vegetated_fluxes.latent_heat[i, j, 1]       + (1 - f) * bare_fluxes.latent_heat[i, j, 1]
        blended_fluxes.sensible_heat[i, j, 1]     = f * vegetated_fluxes.sensible_heat[i, j, 1]     + (1 - f) * bare_fluxes.sensible_heat[i, j, 1]
        blended_fluxes.water_vapor[i, j, 1]       = f * vegetated_fluxes.water_vapor[i, j, 1]       + (1 - f) * bare_fluxes.water_vapor[i, j, 1]
        blended_fluxes.x_momentum[i, j, 1]        = f * vegetated_fluxes.x_momentum[i, j, 1]        + (1 - f) * bare_fluxes.x_momentum[i, j, 1]
        blended_fluxes.y_momentum[i, j, 1]        = f * vegetated_fluxes.y_momentum[i, j, 1]        + (1 - f) * bare_fluxes.y_momentum[i, j, 1]
        blended_fluxes.friction_velocity[i, j, 1] = f * vegetated_fluxes.friction_velocity[i, j, 1] + (1 - f) * bare_fluxes.friction_velocity[i, j, 1]
        blended_fluxes.temperature_scale[i, j, 1] = f * vegetated_fluxes.temperature_scale[i, j, 1] + (1 - f) * bare_fluxes.temperature_scale[i, j, 1]
        blended_fluxes.water_vapor_scale[i, j, 1] = f * vegetated_fluxes.water_vapor_scale[i, j, 1] + (1 - f) * bare_fluxes.water_vapor_scale[i, j, 1]

        # The bare tile is a single skin: its temperature is both its ground surface and
        # its skin, and its turbulent fluxes are all ground fluxes. Canopy quantities are
        # per unit vegetated area, so they are not area-weighted.
        bare_skin_temperature = bare_temperature[i, j, 1]
        blended_temperature.interface[i, j, 1]              = f * vegetated_temperature.interface[i, j, 1] + (1 - f) * bare_skin_temperature
        blended_temperature.canopy[i, j, 1]                 = vegetated_temperature.canopy[i, j, 1]
        blended_temperature.ground[i, j, 1]                 = f * vegetated_temperature.ground[i, j, 1] + (1 - f) * bare_skin_temperature
        blended_temperature.ground_heat_flux[i, j, 1]       = f * vegetated_temperature.ground_heat_flux[i, j, 1] + (1 - f) * Λ * (bare_skin_temperature - T[i, j, 1])
        blended_temperature.canopy_latent_heat[i, j, 1]     = vegetated_temperature.canopy_latent_heat[i, j, 1]
        blended_temperature.ground_latent_heat[i, j, 1]     = f * vegetated_temperature.ground_latent_heat[i, j, 1] + (1 - f) * bare_fluxes.latent_heat[i, j, 1]
        blended_temperature.canopy_sensible_heat[i, j, 1]   = vegetated_temperature.canopy_sensible_heat[i, j, 1]
        blended_temperature.ground_sensible_heat[i, j, 1]   = f * vegetated_temperature.ground_sensible_heat[i, j, 1] + (1 - f) * bare_fluxes.sensible_heat[i, j, 1]
        blended_temperature.canopy_evaporation[i, j, 1]     = vegetated_temperature.canopy_evaporation[i, j, 1]
        blended_temperature.wet_canopy_latent_heat[i, j, 1] = vegetated_temperature.wet_canopy_latent_heat[i, j, 1]
        blended_temperature.land_vapor_flux[i, j, 1]        = f * vegetated_temperature.land_vapor_flux[i, j, 1] + (1 - f) * bare_fluxes.water_vapor[i, j, 1]
        blended_temperature.transpiration[i, j, 1]          = vegetated_temperature.transpiration[i, j, 1]

        # Skin (effective radiating) temperature of the mosaic: σ T⁴ = f · ℐꜛˡʷ(vegetated) +
        # (1 − f) · ℐꜛˡʷ(bare), where the bare tile emits ϵ σ T⁴ at its skin temperature and
        # reflects (1 − ϵ) of the downwelling.
        σ, ϵ = bare_radiation.σ, bare_radiation.ϵ
        vegetated_effective_temperature = vegetated_temperature.effective[i, j, 1]
        ℐꜛˡʷ = ϵ * σ * bare_skin_temperature^4 - (1 - ϵ) * bare_radiation.ℐꜜˡʷ
        T⁴   = f * vegetated_effective_temperature^4 + (1 - f) * ifelse(σ > 0, ℐꜛˡʷ / σ, bare_skin_temperature^4)
        blended_temperature.effective[i, j, 1] = sqrt(sqrt(T⁴))

        # Both tiles see the same incident shortwave, so the mosaic albedo is their area weight.
        blended_temperature.effective_albedo[i, j, 1] =
            f * vegetated_temperature.effective_albedo[i, j, 1] + (1 - f) * bare_radiation.α
    end
end

land_surface_fluxes(interface::TiledLandInterface, bulk_temperature) =
    land_surface_fluxes(interface.temperature, interface, bulk_temperature)
radiative_surface(interface::TiledLandInterface, albedo, emissivity) = radiative_surface(interface.temperature, albedo, emissivity)
