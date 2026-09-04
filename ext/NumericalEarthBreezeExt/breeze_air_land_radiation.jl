#####
##### Surface energy balance coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### Each coupled step adds the net upward surface radiative flux, ℐˡʷꜛ - ℐꜜˡʷ - (1 - α) ℐꜜˢʷ,
##### to the slab's `surface_energy_flux` (positive = upward), reading the downwelling fluxes
##### the radiation exchanger publishes.
#####
##### ℐˡʷꜛ = ε σ Tₛ⁴ + (1 - ε) ℐꜜˡʷ rebuilds RRTMGP's own surface boundary from the live Tₛ, which
##### the RTM's stored upwelling longwave does not track between scheduled solves. The atmosphere
##### keeps absorbing the emission from the last solve, so the two sides disagree by ε σ ΔTₛ⁴
##### within a radiation interval.
#####
##### Shortwave takes (1 - α) of the downwelling rather than the RTM's own net, ℐꜜˢʷ - ℐꜛˢʷ: gray
##### optics computes no upwelling shortwave and never reads α, so the net would hand the land the
##### whole beam. Honoring α instead leaves α ℐꜜˢʷ returned to neither the atmosphere nor space
##### under gray optics, and collapses the direct and diffuse albedos under the scattering solvers.
##### TODO: read ℐꜜˢʷ - ℐꜛˢʷ under clear-sky and all-sky optics, where it is exact.

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Center, Field
using Oceananigans.Grids: inactive_node
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant

const BreezeRTM = Breeze.RadiativeTransferModel

# Bind the interfaces' diagnostic skin temperature — what the atmosphere actually sees;
# equal to land.temperature only for bulk formulations — into an RTM constructed without
# one. Explicit construction wins; with no land interface, Breeze errors at first solve.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_temperature(rtm::BreezeRTM, interfaces)
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

@kernel function _apply_breeze_air_land_radiative_fluxes!(Es, grid, Tˢ, ε, σ, ℐꜜˡʷ, ℐꜜˢʷ, α)
    i, j = @index(Global, NTuple)

    inactive = inactive_node(i, j, 1, grid, Center(), Center(), Center())

    @inbounds begin
        εᵢⱼ = ε[i, j, 1]
        ℐˡʷꜛ = εᵢⱼ * σ * Tˢ[i, j, 1]^4 + (1 - εᵢⱼ) * ℐꜜˡʷ[i, j, 1]
        ℐꜛ = ℐˡʷꜛ - ℐꜜˡʷ[i, j, 1] - (1 - α[i, j, 1]) * ℐꜜˢʷ[i, j, 1]
        Es[i, j, 1] += ifelse(inactive, zero(grid), ℐꜛ)
    end
end

# Downwelling comes from the radiation exchanger and Tˢ from the interface, so an RTM built with its
# own `surface_temperature` cannot force the land with a temperature the land does not carry.
function NumericalEarth.EarthSystemModels.apply_air_land_radiative_fluxes!(
        coupled_model :: NumericalEarth.EarthSystemModels.EarthSystemModel{<:BreezeRTM})

    land = coupled_model.land
    isnothing(land) && return nothing

    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    fluxes = land.fluxes
    hasproperty(fluxes, :surface_energy_flux) || return nothing
    Es = fluxes.surface_energy_flux

    rtm = coupled_model.radiation
    grid = land.grid
    arch = architecture(grid)
    σ = convert(eltype(grid), NumericalEarth.Radiations.default_stefan_boltzmann_constant)
    Tˢ = al_interface.temperature
    ε = rtm.surface_properties.surface_emissivity
    α = rtm.surface_properties.direct_surface_albedo

    state = coupled_model.interfaces.exchanger.radiation.state

    launch!(arch, grid, :xy,
            _apply_breeze_air_land_radiative_fluxes!,
            Es,
            grid,
            Tˢ,
            ε,
            σ,
            state.ℐꜜˡʷ,
            state.ℐꜜˢʷ,
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
