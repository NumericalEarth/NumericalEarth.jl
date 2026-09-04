#####
##### Surface coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### The RTM reads the coupled surface through its `surface_properties`, bound at construction,
##### and publishes its surface downwelling fluxes to the interface radiation state each step.
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Center, ConstantField, Field
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant

const BreezeRTM = Breeze.RadiativeTransferModel

# A canopy column radiates as a blackbody at Tᵉᶠᶠ — σ (Tᵉᶠᶠ)⁴ is its total upwelling longwave —
# and reflects αᵉᶠᶠ, so it overrides configured optics. Any other surface keeps the configured
# optics and an explicitly configured temperature.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_properties(rtm::BreezeRTM, interfaces)
    land = interfaces.atmosphere_land_interface
    temperature = isnothing(land) ? nothing : land.temperature

    if temperature isa CanopyAirSpaceDiagnostics
        rtm = @set rtm.surface_properties.surface_temperature = temperature.effective
        rtm = @set rtm.surface_properties.surface_emissivity = ConstantField(one(eltype(temperature.effective)))
        rtm = @set rtm.surface_properties.direct_surface_albedo = temperature.effective_albedo
        return @set rtm.surface_properties.diffuse_surface_albedo = temperature.effective_albedo
    end

    isnothing(rtm.surface_properties.surface_temperature) || return rtm
    Tˢ = NumericalEarth.EarthSystemModels.surface_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return @set rtm.surface_properties.surface_temperature = Tˢ
end

# `interpolate_state!` copies index for index, so the RTM's horizontal grid has to be the exchange
# grid: the ocean's where there is an ocean, the land's otherwise.
function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(rtm::BreezeRTM, exchange_grid; kw...)
    ℐ = rtm.downwelling_shortwave_flux
    radiation_size = (size(ℐ, 1), size(ℐ, 2))
    exchange_size = (size(exchange_grid, 1), size(exchange_grid, 2))

    radiation_size == exchange_size ||
        throw(ArgumentError("The Breeze RadiativeTransferModel's horizontal grid $radiation_size does not " *
                            "match the exchange grid $exchange_size. The surface fluxes are copied index " *
                            "for index, so the two have to agree."))

    state = (; ℐꜜˢʷ = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(exchange_grid))

    return ComponentExchanger(state, nothing)
end

# Breeze stores fluxes positive-up; the interface state holds positive-down magnitudes.
@kernel function _interpolate_breeze_radiation_state!(state, ℐꜜˢʷ, ℐꜜˡʷ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        state.ℐꜜˢʷ[i, j, 1] = -ℐꜜˢʷ[i, j, 1]
        state.ℐꜜˡʷ[i, j, 1] = -ℐꜜˡʷ[i, j, 1]
    end
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, rtm::BreezeRTM, coupled_model)
    state = exchanger.state

    launch!(architecture(exchange_grid), exchange_grid, :xy,
            _interpolate_breeze_radiation_state!,
            state,
            rtm.downwelling_shortwave_flux,
            rtm.downwelling_longwave_flux)

    # RRTMGP fills interior columns only, while the flux kernels iterate into the halo. This wraps
    # the published state where the exchange grid is periodic; across a bounded edge the halo lies
    # outside the domain and there is nothing to publish there.
    fill_halo_regions!(state.ℐꜜˢʷ)
    fill_halo_regions!(state.ℐꜜˡʷ)

    return nothing
end

# σ is NumericalEarth's default: Breeze's `stefan_bolzmann_constant` is not reachable from the
# model, so land emission pairs with atmospheric absorption only while Breeze keeps that default.
function NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(rtm::BreezeRTM)
    FT = eltype(rtm.downwelling_shortwave_flux)
    ε = rtm.surface_properties.surface_emissivity
    # Whoever reads this state sees the direct albedo, the one the surface energy balance applies.
    # It is also the diffuse albedo unless the RTM was given the two separately.
    α = rtm.surface_properties.direct_surface_albedo
    return (σ = convert(FT, default_stefan_boltzmann_constant),
            surface_properties = (; land = SurfaceRadiationProperties(α, ε)))
end

# The air–sea analog: dispatch peels off no-ocean and prescribed-SST (no net fluxes) cases.
# A responsive ocean under a Breeze RTM raises a MethodError until its radiative heating
# is implemented.
NumericalEarth.EarthSystemModels.apply_air_sea_radiative_fluxes!(
        coupled_model :: NumericalEarth.EarthSystemModels.EarthSystemModel{<:BreezeRTM}) =
    apply_breeze_air_sea_radiative_fluxes!(coupled_model, coupled_model.ocean)

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ::Nothing) = nothing

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ocean) =
    apply_breeze_air_sea_radiative_fluxes!(coupled_model, ocean,
        NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(ocean))

apply_breeze_air_sea_radiative_fluxes!(coupled_model, ocean, ::Nothing) = nothing
