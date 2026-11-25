#####
##### Surface coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: ConstantField, Field
using Oceananigans.Grids: Center
using NumericalEarth.EarthSystemModels.InterfaceComputations: radiative_surface
using NumericalEarth: stateindex
using NumericalEarth.Radiations: SurfaceRadiationProperties, InterfaceRadiationFlux, default_stefan_boltzmann_constant

const BreezeRTM = Breeze.RadiativeTransferModel
const GrayBreezeRTM = Breeze.RadiativeTransferModel{<:Number, <:Any, <:Any, Nothing}

# With an ocean, each cell is a mosaic of land, open ocean and sea ice, and the RTM reads the mosaic's
# temperature, emissivity and albedos from the exchanger (see `update_radiative_surface!`).
# Without one: a canopy column radiates σ (Tʳᵃᵈ)⁴ and reflects αʳᵃᵈ; any other land radiates at its
# interface temperature through the configured optics. Without land either, a configured temperature is kept.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_properties(rtm::BreezeRTM, interfaces)
    if !isnothing(interfaces.atmosphere_ocean_interface)
        surface = interfaces.exchanger.radiation.state.radiative_surface
        return @set rtm.surface_radiation = Breeze.AtmosphereModels.SurfaceRadiation(
            surface.temperature, surface.emissivity, surface.direct_albedo, surface.diffuse_albedo)
    end

    land = interfaces.atmosphere_land_interface
    if !isnothing(land)
        properties = radiative_surface(land, rtm.surface_radiation.direct_surface_albedo,
                                       rtm.surface_radiation.surface_emissivity)
        diffuse = radiative_surface(land, rtm.surface_radiation.diffuse_surface_albedo,
                                    rtm.surface_radiation.surface_emissivity)
        emissivity = properties.emissivity isa Number ? ConstantField(properties.emissivity) : properties.emissivity
        return @set rtm.surface_radiation = Breeze.AtmosphereModels.SurfaceRadiation(
            properties.temperature, emissivity, properties.albedo, diffuse.albedo)
    end

    !isnothing(rtm.surface_radiation.surface_temperature) && return rtm
    Tˢ = NumericalEarth.EarthSystemModels.surface_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return @set rtm.surface_radiation.surface_temperature = Tˢ
end

# The exchanger holds what the RTM has no room for:
#
# - `surface_properties`: the albedo and emissivity that the interface kernels read for each surface.
#   The land's are the RTM's configured optics, which the land kernels read as the ground's. The ocean's
#   and the sea ice's albedos are the values published to the RTM (`PublishedAlbedo`), evaluated from the
#   `surface_albedo_models`.
# - `configured_albedo` and `configured_surface`: the RTM's configured optics, captured here before the
#   radiating surface replaces them in the RTM.
# - `radiative_surface`: the mosaic that the RTM sees when the model has an ocean, and `sea_albedo`, the part
#   of its albedo that the ocean and the sea ice contribute, from which the land's albedo is recovered.
# - `interface_fluxes`: the radiative fluxes at each surface, per unit area of that surface.
function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(rtm::BreezeRTM, exchange_grid;
                                                                                   surface_radiation_properties, kw...)
    surface_field() = Field{Center, Center, Nothing}(exchange_grid)
    configured = rtm.surface_radiation
    ocean = surface_radiation_properties.ocean
    sea_ice = surface_radiation_properties.sea_ice

    state = (; ℐꜜˢʷ = surface_field(),
               ℐꜜˡʷ = surface_field(),
               surface_properties = (; ocean = SurfaceRadiationProperties(PublishedAlbedo(surface_field()), ocean.emissivity),
                                       sea_ice = SurfaceRadiationProperties(PublishedAlbedo(surface_field()), sea_ice.emissivity),
                                       land = SurfaceRadiationProperties(surface_field(), configured.surface_emissivity)),
               configured_albedo = blue_sky_albedo(rtm),
               configured_surface = (; direct_albedo = configured.direct_surface_albedo,
                                       diffuse_albedo = configured.diffuse_surface_albedo,
                                       emissivity = configured.surface_emissivity),
               surface_albedo_models = (; ocean = ocean.albedo, sea_ice = sea_ice.albedo),
               radiative_surface = (; temperature = surface_field(),
                                      emissivity = surface_field(),
                                      direct_albedo = surface_field(),
                                      diffuse_albedo = surface_field(),
                                      sea_albedo = surface_field()),
               interface_fluxes = (; ocean = InterfaceRadiationFlux(exchange_grid),
                                     sea_ice = InterfaceRadiationFlux(exchange_grid),
                                     land = InterfaceRadiationFlux(exchange_grid)))

    return ComponentExchanger(state, nothing)
end

# Downwelling fluxes are negative in both Breeze and the interface state.
@kernel function _interpolate_breeze_radiation_state!(state, ℐꜜˢʷ, ℐꜜˡʷ, albedo, ℵᵒᶜ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        state.ℐꜜˢʷ[i, j, 1] = ℐꜜˢʷ[i, j, 1]
        state.ℐꜜˡʷ[i, j, 1] = ℐꜜˡʷ[i, j, 1]
        sea_albedo = state.radiative_surface.sea_albedo[i, j, 1]
        state.surface_properties.land.albedo[i, j, 1] = land_albedo(albedo, i, j, ℵᵒᶜ[i, j, 1], sea_albedo)
    end
end

# The land's albedo is recovered from the surface as it was published to the RTM, whatever the sea ice
# concentration has become since.
function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, rtm::BreezeRTM, coupled_model)
    state = exchanger.state

    launch!(architecture(exchange_grid), exchange_grid, :xy,
            _interpolate_breeze_radiation_state!,
            state,
            rtm.downwelling_shortwave_flux,
            rtm.downwelling_longwave_flux,
            state.configured_albedo,
            coupled_model.interfaces.surface_partition.ocean_fraction)

    return nothing
end

# σ is NumericalEarth's default: Breeze's `stefan_bolzmann_constant` is not reachable from the
# model, so land emission pairs with atmospheric absorption only while Breeze keeps that default.
function NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(rtm::BreezeRTM, exchanger)
    FT = eltype(rtm.downwelling_shortwave_flux)
    return (σ = convert(FT, default_stefan_boltzmann_constant),
            surface_properties = exchanger.state.surface_properties)
end

# Gray optics transmits the direct beam only.
blue_sky_albedo(rtm::GrayBreezeRTM) = rtm.surface_radiation.direct_surface_albedo

# Two-stream optics reflect the direct and diffuse beams with their own albedos;
# their blend is the ratio of upwelling to downwelling shortwave at the surface face k = 1.
blue_sky_albedo(rtm::BreezeRTM) =
    KernelFunctionOperation{Center, Center, Nothing}(reflected_shortwave_fraction,
                                                     rtm.downwelling_shortwave_flux.grid,
                                                     rtm.upwelling_shortwave_flux,
                                                     rtm.downwelling_shortwave_flux)

@inline reflected_shortwave_fraction(i, j, k, grid, ℐꜛˢʷ, ℐꜜˢʷ) =
    @inbounds ifelse(ℐꜜˢʷ[i, j, 1] < 0, - ℐꜛˢʷ[i, j, 1] / ℐꜜˢʷ[i, j, 1], zero(grid))

# The land's albedo is the configured one. The reflected fraction of a two-stream RTM is that of the whole
# mosaic, αˡᵃ (1 - ℵᵒᶜ) + ℵᵒᶜ ((1 - ℵˢⁱ) αᵒᶜ + ℵˢⁱ αˢⁱ), from which the land's blue-sky albedo is what the
# published contribution of the ocean and the sea ice, `sea_albedo`, leaves. A cell without land keeps the
# reflected fraction.
@inline land_albedo(albedo, i, j, ℵᵒᶜ, sea_albedo) = stateindex(albedo, i, j, 1)

@inline function land_albedo(reflected_fraction::KernelFunctionOperation, i, j, ℵᵒᶜ, sea_albedo)
    α = stateindex(reflected_fraction, i, j, 1)
    αˡᵃ = (α - sea_albedo) / (1 - ℵᵒᶜ)
    return ifelse(ℵᵒᶜ < 1, max(0, αˡᵃ), α)
end
