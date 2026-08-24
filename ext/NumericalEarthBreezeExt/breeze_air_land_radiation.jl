#####
##### Surface radiation coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### The RTM's surface-face downwelling fluxes are published to the interface radiation
##### state every step, so every atmosphere-land formulation — bulk, skin, canopy-air-space —
##### reads the same `(σ, α, ε, SW↓, LW↓)` contract it reads under a `PrescribedRadiation`.
##### Breeze stores fluxes positive-up (downwelling components negative) while the interface
##### contract wants positive-down magnitudes, so the exchange negates them.
#####
##### Formulations that do not internalize radiation (bulk, skin) then take the net upward
##### surface radiative flux into the slab's `surface_energy_flux`,
#####
#####     ℐˡʷꜛ + ℐˡʷꜜ + (1 - α) ℐˢʷꜜ,
#####
##### with ℐˡʷꜛ = ε σ Tₛ⁴ - (1 - ε) ℐˡʷꜜ rebuilt from the live surface state, since the RTM's
##### own upwelling longwave is stale between scheduled solves. Shortwave keeps only the
##### absorbed fraction (1 - α): Breeze stores gross SW↓ with no upwelling field to read back.
##### Exact for coincident direct/diffuse albedos — the coupled configuration.
##### TODO: distinct direct/diffuse albedos need Breeze to expose the direct/diffuse SW↓ split.
#####
##### A `CanopyAirSpace` (or `TiledLandInterface`) internalizes the two-face split in its own
##### solve and drives the slab through the skin→bulk conduction `Gᶜ`, so the add is a no-op
##### there — the radiation reaches the land through the interface state alone.
#####

using Oceananigans.Fields: Center, Field
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant
using NumericalEarth.EarthSystemModels: radiating_temperature
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics

const BreezeRTM = Breeze.RadiativeTransferModel

# Bind the interfaces' radiating temperature — the one whose ε σ T⁴ closes the surface
# upwelling longwave, equal to the atmosphere-facing node only for single-source
# formulations — into an RTM constructed without one. Explicit construction wins; with no
# land interface, Breeze errors at first solve.
# TODO: a canopy-air space folds the (1 - εᵛ) ground reflection into its `Teff`, so an RTM
# emissivity below one double-counts reflected longwave by ~(1 - ε). Exact once Breeze takes
# a per-surface emissivity, or a prescribed upwelling longwave, from the coupler.
function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_temperature(rtm::BreezeRTM, interfaces)
    isnothing(rtm.surface_properties.surface_temperature) || return rtm
    Tˢ = radiating_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return @set rtm.surface_properties.surface_temperature = Tˢ
end

# Exchange only the two downwelling fluxes the interface contract reads: passing the RTM's
# solver internals into the flux kernel (what the generic two-argument `ComponentExchanger`
# would do) cannot compile on GPU.
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

# The atmosphere grid is index-identical horizontally to the exchange grid under the Breeze
# coupling, so the surface faces `k = 1` copy straight across. Interior cells only, matching
# the atmosphere-land flux kernel's own `:xy` launch — RRTMGP leaves its flux halos unfilled.
# The first coupled `update_state!` runs before the atmosphere has ever solved, so the state
# it publishes is dark; the first `time_step!` fills it.
function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, rtm::BreezeRTM, coupled_model)
    launch!(architecture(exchange_grid), exchange_grid, :xy,
            _interpolate_breeze_radiation_state!,
            exchanger.state,
            rtm.downwelling_shortwave_flux,
            rtm.downwelling_longwave_flux)
    return nothing
end

# The RTM's own surface optics become the land's radiative properties, so a bulk or skin
# solve closes its budget against exactly what RRTMGP used. A `CanopyAirSpace` supplies its
# own per-source optics and reads only `σ` and the downwelling fluxes from here.
function NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(rtm::BreezeRTM)
    FT = eltype(rtm.downwelling_shortwave_flux)
    ε = rtm.surface_properties.surface_emissivity
    # Equals `diffuse_surface_albedo` in the coupled configuration; always indexable.
    α = rtm.surface_properties.direct_surface_albedo
    return (σ = convert(FT, default_stefan_boltzmann_constant),
            surface_properties = (; land = SurfaceRadiationProperties(α, ε)))
end

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

    # A canopy-air-space interface (single or tiled) internalizes the two-face longwave and
    # shortwave split in its soil-skin balance — the slab is driven by the skin→bulk
    # conduction, so no separate radiative contribution is added here. Mirrors the guard in
    # the generic `apply_air_land_radiative_fluxes!`.
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
