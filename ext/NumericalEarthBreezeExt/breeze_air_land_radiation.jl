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

    fluxes = land.fluxes
    hasproperty(fluxes, :surface_energy_flux) || return nothing
    Es = fluxes.surface_energy_flux

    rtm = coupled_model.radiation
    grid = land.grid
    arch = architecture(grid)
    σ = convert(eltype(grid), NumericalEarth.Radiations.default_stefan_boltzmann_constant)
    Tˢ = al_interface.temperature   # land skin temperature (K)
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
