#####
##### Surface coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### The RTM reads the coupled surface through its `surface_properties`, bound at construction.
#####

using Oceananigans.Fields: ConstantField
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics

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

# A Breeze RTM needs no exchange state; without this method the generic constructor
# would pass the RTM's solver internals into the flux kernel, which cannot compile on GPU.
NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(::BreezeRTM, exchange_grid; kw...) = nothing

# Empty `surface_properties` keeps radiation out of the turbulent-flux kernel:
# with a Breeze RTM the land receives no interface radiative forcing.
NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(::BreezeRTM) =
    (surface_properties = NamedTuple(),)

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
