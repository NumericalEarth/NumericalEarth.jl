#####
##### The surface that the Breeze `RadiativeTransferModel` sees over an ocean: in each cell a mosaic of
##### land (1 - ℵᵒᶜ), open ocean ℵᵒᶜ (1 - ℵˢⁱ) and sea ice ℵᵒᶜ ℵˢⁱ.
#####
##### The RTM takes one temperature, one emissivity and one direct and one diffuse albedo per column.
##### With the fractions f, the mosaic's are
#####
#####     ϵ = Σ f ϵ,    T = (Σ f ϵ T⁴ / ϵ)^¼,    α = Σ f α,
#####
##### so that the RTM's upwelling longwave, ϵ σ T⁴ - (1 - ϵ) ℐꜜˡʷ, and its reflected shortwave are the
##### fraction-weighted sums of what each surface emits and reflects, and the RTM's net surface flux is
##### the fraction-weighted sum of what the surfaces absorb. Each surface in turn absorbs the RTM's
##### downwelling fluxes with the albedo and emissivity that were published to the RTM.
#####

using Adapt: Adapt
using Oceananigans.Fields: ZeroField
using Oceananigans.Units: Time
using NumericalEarth.EarthSystemModels: EarthSystemModel, DegreesCelsius, convert_to_kelvin
using NumericalEarth.Radiations: Radiations, default_ocean_surface_radiation, default_sea_ice_surface_radiation,
                                 emitted_longwave_radiation, absorbed_longwave_radiation,
                                 transmitted_shortwave_radiation

"""
    PublishedAlbedo(field)

The albedo of a surface as it was last published to the radiative transfer model. The interface kernels
read it in place of the albedo model it was evaluated from, so that a surface absorbs the downwelling
shortwave with the same albedo that the radiative transfer model reflected it with.
"""
struct PublishedAlbedo{F}
    field :: F
end

Adapt.adapt_structure(to, albedo::PublishedAlbedo) = PublishedAlbedo(Adapt.adapt(to, albedo.field))

@inline NumericalEarth.stateindex(albedo::PublishedAlbedo, i, j, k, args...) = @inbounds albedo.field[i, j, 1]

# The ocean and sea ice optics that the interfaces hold for the RTM, which has room for one surface only.
# Ocean: `LatitudeDependentAlbedo` on a `LatitudeLongitudeGrid`, 0.06 elsewhere; emissivity 0.97.
# Sea ice: the component's own (`SeaIceAlbedo` for `PrescribedSeaIce`), otherwise albedo 0.7; emissivity 1.
NumericalEarth.EarthSystemModels.InterfaceComputations.default_surface_radiation_properties(::BreezeRTM, ocean, sea_ice, grid) =
    (ocean = default_ocean_surface_radiation(grid),
     sea_ice = default_sea_ice_surface_radiation(sea_ice, grid))

Radiations.interface_radiation_fluxes(::BreezeRTM, radiation_exchanger) = radiation_exchanger.state.interface_fluxes

# What `update_state!` reads and does not recompute: the albedos last published to the RTM, with which the
# surfaces absorb the RTM's downwelling shortwave, and the mosaic that the RTM last saw. The radiative fluxes
# at the land surface are diagnosed with the land fluxes at the start of a step, not by `update_state!`.
function NumericalEarth.EarthSystemModels.exchanged_radiation_state(::BreezeRTM, radiation_exchanger)
    state = radiation_exchanger.state
    land = state.interface_fluxes.land

    return (; ocean_albedo = state.surface_properties.ocean.albedo.field,
              sea_ice_albedo = state.surface_properties.sea_ice.albedo.field,
              radiative_surface = state.radiative_surface,
              land_fluxes = (; land.upwelling_longwave, land.downwelling_longwave, land.downwelling_shortwave))
end

#####
##### Publish the mosaic
#####

# The land's radiating surface; without land its fraction is zero.
mosaic_land_surface(::Nothing, configured) = (temperature = 0, emissivity = 0, direct_albedo = 0, diffuse_albedo = 0)

function mosaic_land_surface(land_interface, configured)
    direct = radiative_surface(land_interface, configured.direct_albedo, configured.emissivity)
    diffuse = radiative_surface(land_interface, configured.diffuse_albedo, configured.emissivity)
    return (temperature = direct.temperature,
            emissivity = direct.emissivity,
            direct_albedo = direct.albedo,
            diffuse_albedo = diffuse.albedo)
end

# Interface temperatures of the ocean and the sea ice, with their units
function mosaic_sea_surface(coupled_model)
    interfaces = coupled_model.interfaces
    ai_interface = interfaces.atmosphere_sea_ice_interface
    sea_ice_properties = interfaces.sea_ice_properties

    return (ocean_fraction = interfaces.surface_partition.ocean_fraction,
            sea_ice_concentration = sea_ice_concentration(coupled_model.sea_ice),
            ocean_temperature = interfaces.atmosphere_ocean_interface.temperature,
            sea_ice_temperature = isnothing(ai_interface) ? ZeroField(eltype(interfaces.exchanger.grid)) : ai_interface.temperature,
            ocean_units = interfaces.ocean_properties.temperature_units,
            sea_ice_units = isnothing(sea_ice_properties) ? DegreesCelsius() : sea_ice_properties.temperature_units)
end

NumericalEarth.EarthSystemModels.update_radiative_surface!(coupled_model::EarthSystemModel{<:BreezeRTM}) =
    update_radiative_surface!(coupled_model, coupled_model.interfaces.atmosphere_ocean_interface)

# Without an ocean the RTM reads the land's radiating surface directly
update_radiative_surface!(coupled_model, ::Nothing) = nothing

function update_radiative_surface!(coupled_model, atmosphere_ocean_interface)
    interfaces = coupled_model.interfaces
    grid = interfaces.exchanger.grid
    state = interfaces.exchanger.radiation.state

    launch!(architecture(grid), grid, :xy, _update_radiative_surface!,
            state.radiative_surface,
            state.surface_properties,
            state.surface_albedo_models,
            mosaic_land_surface(interfaces.atmosphere_land_interface, state.configured_surface),
            mosaic_sea_surface(coupled_model),
            state.ℐꜜˢʷ,
            grid,
            coupled_model.clock)

    return nothing
end

@kernel function _update_radiative_surface!(surface, properties, albedo_models, land, sea, ℐꜜˢʷ, grid, clock)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)
    location = (Center, Center, Center)

    @inbounds begin
        ℵᵒᶜ = sea.ocean_fraction[i, j, 1]
        ℵˢⁱ = sea.sea_ice_concentration[i, j, 1]
        Tᵒᶜ = convert_to_kelvin(sea.ocean_units, sea.ocean_temperature[i, j, 1])
        Tˢⁱ = convert_to_kelvin(sea.sea_ice_units, sea.sea_ice_temperature[i, j, 1])
        ℐꜜˢʷᵢ = ℐꜜˢʷ[i, j, 1]
    end

    # Evaluate the albedo models, and publish their values to the interface kernels too
    αᵒᶜ = stateindex(albedo_models.ocean, i, j, kᴺ, grid, time, location, ℐꜜˢʷᵢ)
    αˢⁱ = stateindex(albedo_models.sea_ice, i, j, kᴺ, grid, time, location, ℐꜜˢʷᵢ)
    ϵᵒᶜ = stateindex(properties.ocean.emissivity, i, j, kᴺ, grid, time, location)
    ϵˢⁱ = stateindex(properties.sea_ice.emissivity, i, j, kᴺ, grid, time, location)

    Tˡᵃ = stateindex(land.temperature, i, j, 1)
    ϵˡᵃ = stateindex(land.emissivity, i, j, 1)
    αˡᵃ_direct = stateindex(land.direct_albedo, i, j, 1)
    αˡᵃ_diffuse = stateindex(land.diffuse_albedo, i, j, 1)

    land_fraction = 1 - ℵᵒᶜ
    ocean_fraction = ℵᵒᶜ * (1 - ℵˢⁱ)
    sea_ice_fraction = ℵᵒᶜ * ℵˢⁱ

    # Emissivity, and the temperature that emits the mosaic's longwave through it
    ϵ = land_fraction * ϵˡᵃ + ocean_fraction * ϵᵒᶜ + sea_ice_fraction * ϵˢⁱ
    ϵT⁴ = land_fraction * ϵˡᵃ * Tˡᵃ^4 + ocean_fraction * ϵᵒᶜ * Tᵒᶜ^4 + sea_ice_fraction * ϵˢⁱ * Tˢⁱ^4
    T⁴ = ifelse(ϵ > 0, ϵT⁴ / ϵ, land_fraction * Tˡᵃ^4 + ocean_fraction * Tᵒᶜ^4 + sea_ice_fraction * Tˢⁱ^4)

    sea_albedo = ocean_fraction * αᵒᶜ + sea_ice_fraction * αˢⁱ

    @inbounds begin
        properties.ocean.albedo.field[i, j, 1] = αᵒᶜ
        properties.sea_ice.albedo.field[i, j, 1] = αˢⁱ
        surface.temperature[i, j, 1] = sqrt(sqrt(T⁴))
        surface.emissivity[i, j, 1] = ϵ
        surface.direct_albedo[i, j, 1] = land_fraction * αˡᵃ_direct + sea_albedo
        surface.diffuse_albedo[i, j, 1] = land_fraction * αˡᵃ_diffuse + sea_albedo
        surface.sea_albedo[i, j, 1] = sea_albedo
    end
end

#####
##### Radiative fluxes at the ocean and sea ice surfaces
#####

# The air–sea analog of the land's radiative fluxes: an ocean that responds takes its radiative heating,
# a prescribed one (no net fluxes) takes none.
NumericalEarth.EarthSystemModels.apply_air_sea_radiative_fluxes!(coupled_model::EarthSystemModel{<:BreezeRTM}) =
    apply_breeze_air_sea_radiative_fluxes!(coupled_model, coupled_model.ocean)

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ::Nothing) = nothing

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ocean) =
    invoke(NumericalEarth.EarthSystemModels.apply_air_sea_radiative_fluxes!, Tuple{EarthSystemModel}, coupled_model)

# Likewise for sea ice. The diagnostics of both surfaces are written last, whether or not they respond.
function NumericalEarth.EarthSystemModels.apply_air_sea_ice_radiative_fluxes!(coupled_model::EarthSystemModel{<:BreezeRTM})
    invoke(NumericalEarth.EarthSystemModels.apply_air_sea_ice_radiative_fluxes!, Tuple{EarthSystemModel}, coupled_model)
    compute_sea_surface_radiative_fluxes!(coupled_model, coupled_model.interfaces.atmosphere_ocean_interface)
    return nothing
end

compute_sea_surface_radiative_fluxes!(coupled_model, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Write the radiative fluxes at the ocean and sea ice surfaces under a Breeze `RadiativeTransferModel` into
the `interface_fluxes` of the radiation exchanger: per unit area of each surface, positive upward, at the
interface temperatures of this `update_state!` and with the albedos published to the radiative transfer model
before its last solve.
"""
function compute_sea_surface_radiative_fluxes!(coupled_model, atmosphere_ocean_interface)
    interfaces = coupled_model.interfaces
    grid = interfaces.exchanger.grid
    exchanger = interfaces.exchanger.radiation
    state = exchanger.state
    rk = NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(coupled_model.radiation, exchanger)

    launch!(architecture(grid), grid, :xy, _compute_sea_surface_radiative_fluxes!,
            state.interface_fluxes.ocean,
            state.interface_fluxes.sea_ice,
            rk,
            mosaic_sea_surface(coupled_model),
            state.ℐꜜˢʷ,
            state.ℐꜜˡʷ,
            grid,
            coupled_model.clock)

    return nothing
end

@kernel function _compute_sea_surface_radiative_fluxes!(ocean_fluxes, sea_ice_fluxes, rk, sea, ℐꜜˢʷ, ℐꜜˡʷ, grid, clock)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)
    location = (Center, Center, Center)
    ocean = rk.surface_properties.ocean
    sea_ice = rk.surface_properties.sea_ice

    @inbounds begin
        Tᵒᶜ = convert_to_kelvin(sea.ocean_units, sea.ocean_temperature[i, j, 1])
        Tˢⁱ = convert_to_kelvin(sea.sea_ice_units, sea.sea_ice_temperature[i, j, 1])
        ℐꜜˢʷᵢ = ℐꜜˢʷ[i, j, 1]
        ℐꜜˡʷᵢ = ℐꜜˡʷ[i, j, 1]
    end

    αᵒᶜ = stateindex(ocean.albedo, i, j, kᴺ, grid, time, location, ℐꜜˢʷᵢ)
    αˢⁱ = stateindex(sea_ice.albedo, i, j, kᴺ, grid, time, location, ℐꜜˢʷᵢ)
    ϵᵒᶜ = stateindex(ocean.emissivity, i, j, kᴺ, grid, time, location)
    ϵˢⁱ = stateindex(sea_ice.emissivity, i, j, kᴺ, grid, time, location)

    @inbounds begin
        ocean_fluxes.upwelling_longwave[i, j, 1]    = emitted_longwave_radiation(Tᵒᶜ, rk.σ, ϵᵒᶜ)
        ocean_fluxes.downwelling_longwave[i, j, 1]  = absorbed_longwave_radiation(ϵᵒᶜ, ℐꜜˡʷᵢ)
        ocean_fluxes.downwelling_shortwave[i, j, 1] = transmitted_shortwave_radiation(αᵒᶜ, ℐꜜˢʷᵢ)

        sea_ice_fluxes.upwelling_longwave[i, j, 1]    = emitted_longwave_radiation(Tˢⁱ, rk.σ, ϵˢⁱ)
        sea_ice_fluxes.downwelling_longwave[i, j, 1]  = absorbed_longwave_radiation(ϵˢⁱ, ℐꜜˡʷᵢ)
        sea_ice_fluxes.downwelling_shortwave[i, j, 1] = transmitted_shortwave_radiation(αˢⁱ, ℐꜜˢʷᵢ)
    end
end
