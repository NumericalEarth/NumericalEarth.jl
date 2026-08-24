#####
##### Surface energy balance coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### Each coupled step adds the net upward surface radiative flux, ℐˡʷꜛ + ℐˡʷꜜ + (1 - α) ℐˢʷꜜ,
##### to the slab's `surface_energy_flux` (positive = upward; downwelling stored negative).
##### Longwave up is rebuilt from the live surface state, ℐˡʷꜛ = ε σ Tₛ⁴ - (1 - ε) ℐˡʷꜜ, since
##### the RTM's own upwelling longwave is stale between scheduled solves. Shortwave keeps only
##### the absorbed fraction (1 - α): Breeze stores gross SW↓ with no upwelling field to read
##### back. Exact for coincident direct/diffuse albedos — the coupled configuration.
##### TODO: distinct direct/diffuse albedos need Breeze to expose the direct/diffuse SW↓ split.
#####

using Oceananigans.Fields: Center, Field
using NumericalEarth.EarthSystemModels: radiating_temperature
using NumericalEarth.EarthSystemModels.InterfaceComputations: CanopyAirSpaceDiagnostics

const BreezeRTM = Breeze.RadiativeTransferModel

function NumericalEarth.EarthSystemModels.materialize_earth_system_surface_properties(rtm::BreezeRTM, interfaces)
    Tˢ = radiating_temperature(interfaces)
    isnothing(Tˢ) && return rtm
    return bind_surface_properties(rtm, Tˢ, land_interface_temperature(interfaces),
                                   interfaces.exchanger.grid)
end

land_interface_temperature(interfaces) = interface_temperature(interfaces.atmosphere_land_interface)
interface_temperature(::Nothing) = nothing
interface_temperature(interface) = interface.temperature

# A single-source surface emits and reflects with the same ε and α the RTM applies, so its own
# fields are already exact; an explicitly constructed `surface_temperature` wins.
function bind_surface_properties(rtm, Tˢ, temperature, grid)
    isnothing(rtm.surface_properties.surface_temperature) || return rtm
    return @set rtm.surface_properties.surface_temperature = Tˢ
end

# A canopy carries its own optics, so both what it radiates and what it reflects are results of
# its solve rather than configuration. The coupler owns the fields it republishes each step,
# overriding explicitly constructed ones. One albedo field feeds both the direct and diffuse
# slots: the two-source split is broadband.
function bind_surface_properties(rtm, Tˢ, ::CanopyAirSpaceDiagnostics, grid)
    α = Field{Center, Center, Nothing}(grid)
    rtm = @set rtm.surface_properties.surface_temperature = Field{Center, Center, Nothing}(grid)
    rtm = @set rtm.surface_properties.direct_surface_albedo = α
    return @set rtm.surface_properties.diffuse_surface_albedo = α
end

# RRTMGP emits ε σ T⁴ + (1 - ε) ℐꜜˡʷ, but a canopy's `Teff` already carries the reflected
# downwelling, so binding it directly would apply the reflection twice. Bind instead the
# temperature that makes RRTMGP reproduce the canopy's upwelling exactly,
#
#     ε σ Tʳ⁴ + (1 - ε) ℐꜜˡʷ = σ Teff⁴.
#
# `ℐꜜˡʷ` is the previous solve's, so what survives is (1 - ε) times its change between solves
# rather than (1 - ε) times the full surface-to-sky contrast. The albedo needs no inversion:
# RRTMGP reflects α ℐꜜˢʷ, which is exactly what the canopy leaves unabsorbed once α is its own.
@kernel function _publish_canopy_radiative_properties!(Tʳ, α, Teff, αeff, ℐꜜˡʷ, ε, σ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Tᵉ = Teff[i, j, 1]
        εᵢⱼ = ε[i, j, 1]
        source = (σ * Tᵉ^4 - (1 - εᵢⱼ) * ℐꜜˡʷ[i, j, 1]) / εᵢⱼ
        invertible = (Tᵉ > 0) & (εᵢⱼ > 0) & (source > 0)
        Tʳ[i, j, 1] = ifelse(invertible, sqrt(sqrt(max(source, zero(source)) / σ)), Tᵉ)
        α[i, j, 1] = αeff[i, j, 1]
    end
end

NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, rtm::BreezeRTM) =
    update_radiative_properties!(rtm, coupled_model,
                                 land_interface_temperature(coupled_model.interfaces))

update_radiative_properties!(rtm, coupled_model, temperature) = nothing

function update_radiative_properties!(rtm, coupled_model, temperature::CanopyAirSpaceDiagnostics)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    launch!(architecture(grid), grid, :xy,
            _publish_canopy_radiative_properties!,
            rtm.surface_properties.surface_temperature,
            rtm.surface_properties.direct_surface_albedo,
            temperature.effective,
            temperature.effective_albedo,
            exchanger.radiation.state.ℐꜜˡʷ,
            rtm.surface_properties.surface_emissivity,
            convert(eltype(grid), NumericalEarth.Radiations.default_stefan_boltzmann_constant))
    return nothing
end

# A Breeze RTM needs no exchange state; without this method the generic constructor
# would pass the RTM's solver internals into the flux kernel, which cannot compile on GPU.
NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(::BreezeRTM, exchange_grid; kw...) = nothing

# Empty `surface_properties` keeps radiation out of the turbulent-flux kernel:
# with a Breeze RTM the radiative term enters via `apply_air_land_radiative_fluxes!` below.
NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(::BreezeRTM) =
    (surface_properties = NamedTuple(),)

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
    σ = convert(eltype(grid), NumericalEarth.Radiations.default_stefan_boltzmann_constant)
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
