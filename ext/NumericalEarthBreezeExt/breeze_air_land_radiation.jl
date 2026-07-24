#####
##### Surface energy balance coupling for the Breeze RRTMGP `RadiativeTransferModel`.
#####
##### The RTM lives at `coupled_model.radiation`. We add the *net upward radiative flux*
##### at the surface to the slab's `surface_energy_flux` accumulator, using the "positive
##### flux = upward" sign convention (downwelling components are stored negative).
#####
##### Longwave is complete in the RTM's fields: `upwelling_longwave_flux` already bakes in
##### surface emission and reflection (ε σ Tₛ⁴ + (1-ε)·LW↓, from RRTMGP's emissivity
##### boundary condition), so `ℐˡʷꜛ + ℐˡʷꜜ` is the net upward longwave.
#####
##### Shortwave is NOT: RRTMGP reflects the surface albedo internally, but Breeze stores
##### only the *gross* downwelling shortwave (`downwelling_shortwave_flux = -SW↓`, total
##### direct + diffuse) — there is no upwelling-shortwave field to read back. Adding `ℐˢʷꜜ`
##### unmodified would deposit 100 % of SW↓ in the surface regardless of albedo. We instead
##### keep only the absorbed fraction, subtracting the reflected `α·SW↓`: the net upward
##### shortwave is `(1 - α)·ℐˢʷꜜ` (= -(1-α)·SW↓). `α` is the RTM's surface albedo; this is
##### exact when its direct and diffuse albedos coincide — a single `surface_albedo` or a
##### `CopernicusAlbedo()` (the coupled configuration).
##### TODO: an exact correction for *distinct* direct/diffuse albedos needs the direct/diffuse
##### split of SW↓, which Breeze does not expose — better fixed in Breeze by storing the
##### surface net (or upwelling) shortwave.
#####
##### So the net upward radiative flux at the surface face `k = 1` is
#####
#####    ℐˡʷꜛ + ℐˡʷꜜ + (1 - α)·ℐˢʷꜜ
#####
##### This runs in `update_state!` after the turbulent (sensible + latent) flux has been
##### written to `surface_energy_flux`, so the kernel adds the radiative term on top.
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

# A Breeze RTM needs no exchange state (the flux kernel takes the zero-radiation-state
# path, and Phase 4 reads the RTM's surface fluxes directly). Without this method the
# generic (state, regridder) constructor would store the RTM itself as state and pass
# its solver internals into the flux kernel — which cannot compile on GPU.
NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(::BreezeRTM, exchange_grid; kw...) = nothing

# The turbulent-flux kernel asks the radiation for "kernel properties" used to
# augment its interface energy balance. With a Breeze RTM the radiative
# contribution to the surface energy balance is handled separately by
# `apply_air_land_radiative_fluxes!` below, so we return an empty
# `surface_properties` here — `air_land_interface_radiation_state` already
# handles the "no land surface_properties" path by returning a zero radiation
# state.
NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(::BreezeRTM) =
    (surface_properties = NamedTuple(),)

@kernel function _apply_breeze_air_land_radiative_fluxes!(Es, ℐˡʷꜛ, ℐˡʷꜜ, ℐˢʷꜜ, α)
    i, j = @index(Global, NTuple)
    # Longwave is already net (emission + reflection baked into `ℐˡʷꜛ`). `ℐˢʷꜜ` is the
    # GROSS downwelling shortwave — RRTMGP stores no upwelling-SW field — so keep only the
    # absorbed fraction `(1 - α)·ℐˢʷꜜ`; the reflected `α·SW↓` is not deposited in the surface.
    @inbounds Es[i, j, 1] += ℐˡʷꜛ[i, j, 1] + ℐˡʷꜜ[i, j, 1] + (1 - α[i, j, 1]) * ℐˢʷꜜ[i, j, 1]
end

# Dispatch on `EarthSystemModel{<:BreezeRTM}`: the existing generic
# `apply_air_land_radiative_fluxes!` only handles `PrescribedRadiation`-style
# radiation (which carries `interface_fluxes.land` etc.); the Breeze RTM
# carries the surface flux fields directly on the model.
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

    # RRTMGP applies the surface albedo internally but Breeze stores only the gross
    # downwelling shortwave, so the kernel subtracts the reflected fraction `α·SW↓`.
    # `direct_surface_albedo` equals `diffuse_surface_albedo` for a single `surface_albedo`
    # or a `CopernicusAlbedo()` (the coupled configuration); RRTMGP always materializes it
    # to an indexable `Field`/`ConstantField`.
    α = rtm.surface_properties.direct_surface_albedo

    launch!(arch, grid, :xy,
            _apply_breeze_air_land_radiative_fluxes!,
            Es,
            rtm.upwelling_longwave_flux,
            rtm.downwelling_longwave_flux,
            rtm.downwelling_shortwave_flux,
            α)
    return nothing
end
