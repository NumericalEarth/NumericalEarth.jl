#####
##### Surface radiation coupling for the Breeze RRTMGP `RadiativeTransferModel`. The RTM
##### publishes its surface downwelling fluxes into the same interface radiation state a
##### `PrescribedRadiation` fills. Breeze stores fluxes positive-up, so its downwelling
##### components are negative where the interface contract wants positive-down magnitudes.
##### TODO: distinct direct/diffuse albedos need Breeze to expose the direct/diffuse SW↓ split.
#####

using Oceananigans.Fields: Center, Field
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant
using NumericalEarth.EarthSystemModels: radiating_temperature
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics

const BreezeRTM = Breeze.RadiativeTransferModel

# Binds `Teff` for a canopy, not the canopy-air node the atmosphere ventilates into.
# TODO: `Teff` already carries the reflected downwelling, so an RTM emissivity below one
# re-applies it — RRTMGP emits ε σ Teff⁴ + (1 - ε) ℐꜜˡʷ, short of σ Teff⁴ by (1 - ε)(LWu - ℐꜜˡʷ).
# Needs Breeze to accept a prescribed upwelling longwave.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_temperature(rtm::BreezeRTM, interfaces)
    isnothing(rtm.surface_properties.surface_temperature) || return rtm
    Tˢ = radiating_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return @set rtm.surface_properties.surface_temperature = Tˢ
end

# Without this method the two-argument `ComponentExchanger(state, regridder)` convenience
# swallows the call and stores the RTM itself as exchange state.
function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(::BreezeRTM, exchange_grid; kw...)
    state = (; ℐꜜˢʷ = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(exchange_grid))
    return ComponentExchanger(state, nothing)
end

@kernel function _interpolate_breeze_radiation_state!(state, ℐˢʷꜜ, ℐˡʷꜜ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        state.ℐꜜˢʷ[i, j, 1] = -ℐˢʷꜜ[i, j, 1]
        state.ℐꜜˡʷ[i, j, 1] = -ℐˡʷꜜ[i, j, 1]
    end
end

# The atmosphere grid is horizontally index-identical to the exchange grid under the Breeze
# coupling, so the surface faces copy straight across.
function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, rtm::BreezeRTM, coupled_model)
    launch!(architecture(exchange_grid), exchange_grid, :xy,
            _interpolate_breeze_radiation_state!,
            exchanger.state,
            rtm.downwelling_shortwave_flux,
            rtm.downwelling_longwave_flux)
    return nothing
end

function NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(rtm::BreezeRTM)
    FT = eltype(rtm.downwelling_shortwave_flux)
    ε = rtm.surface_properties.surface_emissivity
    # Equals `diffuse_surface_albedo` in the coupled configuration; always indexable.
    α = rtm.surface_properties.direct_surface_albedo
    return (σ = convert(FT, default_stefan_boltzmann_constant),
            surface_properties = (; land = SurfaceRadiationProperties(α, ε)))
end

# ℐˡʷꜛ is rebuilt from the live surface state: the RTM's own upwelling field is stale between
# scheduled solves. Shortwave keeps only the absorbed fraction — Breeze stores gross SW↓ with
# no upwelling field to read back.
@kernel function _apply_breeze_air_land_radiative_fluxes!(Es, Tˢ, ε, σ, ℐˡʷꜜ, ℐˢʷꜜ, α)
    i, j = @index(Global, NTuple)
    @inbounds begin
        εᵢⱼ = ε[i, j, 1]
        ℐˡʷꜛ = εᵢⱼ * σ * Tˢ[i, j, 1]^4 - (1 - εᵢⱼ) * ℐˡʷꜜ[i, j, 1]
        Es[i, j, 1] += ℐˡʷꜛ + ℐˡʷꜜ[i, j, 1] + (1 - α[i, j, 1]) * ℐˢʷꜜ[i, j, 1]
    end
end

# The generic method reads `PrescribedRadiation`-style `interface_fluxes`;
# a Breeze RTM carries its surface flux fields directly on the model.
function NumericalEarth.EarthSystemModels.apply_air_land_radiative_fluxes!(
        coupled_model :: NumericalEarth.EarthSystemModels.EarthSystemModel{<:BreezeRTM})

    land = coupled_model.land
    isnothing(land) && return nothing

    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    # A canopy (single or tiled) absorbs radiation inside its own solve; nothing is added here.
    al_interface.temperature isa CanopyAirSpaceDiagnostics && return nothing

    fluxes = land.fluxes
    hasproperty(fluxes, :surface_energy_flux) || return nothing
    Es = fluxes.surface_energy_flux

    rtm = coupled_model.radiation
    grid = land.grid
    arch = architecture(grid)
    σ = convert(eltype(grid), default_stefan_boltzmann_constant)
    Tˢ = rtm.surface_properties.surface_temperature
    ε = rtm.surface_properties.surface_emissivity

    # Equals `diffuse_surface_albedo` in the coupled configuration; always indexable.
    α = rtm.surface_properties.direct_surface_albedo

    launch!(arch, grid, :xy,
            _apply_breeze_air_land_radiative_fluxes!,
            Es,
            Tˢ,
            ε,
            σ,
            rtm.downwelling_longwave_flux,
            rtm.downwelling_shortwave_flux,
            α)
    return nothing
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
