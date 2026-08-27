#####
##### Surface energy balance coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### Each coupled step adds the net upward surface radiative flux, ℐˡʷꜛ - ℐꜜˡʷ - (1 - α) ℐꜜˢʷ,
##### to the slab's `surface_energy_flux` (positive = upward), reading the downwelling fluxes
##### the radiation exchanger publishes. Longwave up is rebuilt from the live surface state,
##### ℐˡʷꜛ = ε σ Tₛ⁴ + (1 - ε) ℐꜜˡʷ, since the RTM's own upwelling longwave is stale between
##### scheduled solves. Shortwave keeps only the absorbed fraction (1 - α): Breeze stores gross
##### SW↓ with no upwelling field to read back. Exact for coincident direct/diffuse albedos —
##### the coupled configuration.
##### TODO: distinct direct/diffuse albedos need Breeze to expose the direct/diffuse SW↓ split.

using Oceananigans.Fields: Center, ConstantField, Field
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics

const BreezeRTM = Breeze.RadiativeTransferModel

function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_properties(rtm::BreezeRTM, interfaces)
    al_interface = interfaces.atmosphere_land_interface
    temperature = isnothing(al_interface) ? nothing : al_interface.temperature

    # A canopy owns its surface optics, overriding configured properties: σ T⁴ is the
    # column's total upwelling longwave — emission plus reflected downwelling — so a blackbody
    # at that temperature (ε = 1) reproduces it exactly, one broadband albedo in both shortwave slots.
    if temperature isa CanopyAirSpaceDiagnostics
        rtm = @set rtm.surface_properties.surface_temperature = temperature.effective
        rtm = @set rtm.surface_properties.surface_emissivity = ConstantField(one(eltype(temperature.effective)))
        rtm = @set rtm.surface_properties.direct_surface_albedo = temperature.effective_albedo
        return @set rtm.surface_properties.diffuse_surface_albedo = temperature.effective_albedo
    end

    Tˢ = NumericalEarth.EarthSystemModels.surface_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    isnothing(rtm.surface_properties.surface_temperature) || return rtm
    return @set rtm.surface_properties.surface_temperature = Tˢ
end

# RRTMGP copies scalar surface optics into its solver boundary conditions at construction
# only, so field-valued emissivity and albedo are republished each coupled step.
@kernel function _update_rrtmgp_surface_optics!(sfc_emis, sfc_alb_direct, sfc_alb_diffuse,
                                                emissivity, direct_albedo, diffuse_albedo, Nx)
    i, j = @index(Global, NTuple)
    c = i + (j - 1) * Nx
    @inbounds begin
        for band in axes(sfc_emis, 1)
            sfc_emis[band, c] = emissivity[i, j, 1]
        end
        for band in axes(sfc_alb_direct, 1)
            sfc_alb_direct[band, c]  = direct_albedo[i, j, 1]
            sfc_alb_diffuse[band, c] = diffuse_albedo[i, j, 1]
        end
    end
end

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, rtm::BreezeRTM)
    grid = coupled_model.interfaces.exchanger.grid
    properties = rtm.surface_properties
    launch!(architecture(grid), grid, :xy,
            _update_rrtmgp_surface_optics!,
            rtm.longwave_solver.bcs.sfc_emis,
            rtm.shortwave_solver.bcs.sfc_alb_direct,
            rtm.shortwave_solver.bcs.sfc_alb_diffuse,
            properties.surface_emissivity,
            properties.direct_surface_albedo,
            properties.diffuse_surface_albedo,
            grid.Nx)
    return nothing
end

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(::BreezeRTM, exchange_grid; kw...)
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

# The atmosphere and exchange grids are horizontally index-identical under the Breeze coupling.
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
    α = rtm.surface_properties.direct_surface_albedo
    return (σ = convert(FT, default_stefan_boltzmann_constant),
            surface_properties = (; land = SurfaceRadiationProperties(α, ε)))
end

@kernel function _apply_breeze_air_land_radiative_fluxes!(Es, Tˢ, ε, σ, ℐꜜˡʷ, ℐꜜˢʷ, α)
    i, j = @index(Global, NTuple)
    @inbounds begin
        εᵢⱼ = ε[i, j, 1]
        ℐˡʷꜛ = εᵢⱼ * σ * Tˢ[i, j, 1]^4 + (1 - εᵢⱼ) * ℐꜜˡʷ[i, j, 1]
        Es[i, j, 1] += ℐˡʷꜛ - ℐꜜˡʷ[i, j, 1] - (1 - α[i, j, 1]) * ℐꜜˢʷ[i, j, 1]
    end
end

# Downwelling comes from the radiation exchanger; emissivity, albedo and Tˢ from the RTM.
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
    σ = convert(eltype(grid), NumericalEarth.Radiations.default_stefan_boltzmann_constant)
    Tˢ = rtm.surface_properties.surface_temperature
    ε = rtm.surface_properties.surface_emissivity

    # Equals `diffuse_surface_albedo` in the coupled configuration; always indexable.
    α = rtm.surface_properties.direct_surface_albedo

    state = coupled_model.interfaces.exchanger.radiation.state

    launch!(arch, grid, :xy,
            _apply_breeze_air_land_radiative_fluxes!,
            Es,
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
