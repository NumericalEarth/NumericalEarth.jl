#####
##### Surface coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### The RTM reads the coupled surface through its `surface_properties`, bound at construction.
#####

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: ConstantField, Field
using Oceananigans.Grids: Center
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant

const BreezeRTM = Breeze.RadiativeTransferModel
const GrayBreezeRTM = Breeze.RadiativeTransferModel{<:Number, <:Any, <:Any, Nothing}

# A canopy column radiates as a blackbody at Tᵉᶠᶠ — σ (Tᵉᶠᶠ)⁴ is its total upwelling longwave —
# and reflects αᵉᶠᶠ, so it overrides configured optics. Any other surface keeps the configured
# optics and an explicitly configured temperature.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_properties(rtm::BreezeRTM, interfaces)
    land = interfaces.atmosphere_land_interface
    temperature = isnothing(land) ? nothing : land.temperature

    if temperature isa CanopyAirSpaceDiagnostics
        rtm = @set rtm.surface_radiation.surface_temperature = temperature.effective
        rtm = @set rtm.surface_radiation.surface_emissivity = ConstantField(one(eltype(temperature.effective)))
        rtm = @set rtm.surface_radiation.direct_surface_albedo = temperature.effective_albedo
        return @set rtm.surface_radiation.diffuse_surface_albedo = temperature.effective_albedo
    end

    isnothing(rtm.surface_radiation.surface_temperature) || return rtm
    Tˢ = NumericalEarth.EarthSystemModels.surface_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return @set rtm.surface_radiation.surface_temperature = Tˢ
end

# The exchanger keeps the configured surface optics, which the land reads as its ground's;
# a bound canopy overrides the RTM's own slots with its column optics.
function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(rtm::BreezeRTM, exchange_grid; kw...)
    ε = rtm.surface_radiation.surface_emissivity
    α = blue_sky_albedo(rtm)
    state = (; ℐꜜˢʷ = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(exchange_grid),
               surface_properties = (; land = SurfaceRadiationProperties(α, ε)))

    return ComponentExchanger(state, nothing)
end

# Downwelling fluxes are negative in both Breeze and the interface state.
@kernel function _interpolate_breeze_radiation_state!(state, ℐꜜˢʷ, ℐꜜˡʷ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        state.ℐꜜˢʷ[i, j, 1] = ℐꜜˢʷ[i, j, 1]
        state.ℐꜜˡʷ[i, j, 1] = ℐꜜˡʷ[i, j, 1]
    end
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, rtm::BreezeRTM, coupled_model)
    state = exchanger.state

    launch!(architecture(exchange_grid), exchange_grid, :xy,
            _interpolate_breeze_radiation_state!,
            state,
            rtm.downwelling_shortwave_flux,
            rtm.downwelling_longwave_flux)

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
